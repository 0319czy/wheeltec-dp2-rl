"""DP2 一阶纯 RL 站立续训与审计。观测、PWM 限幅和保护与台架固件对齐。"""
import argparse
import json
import pickle
from functools import partial
from pathlib import Path

import mujoco
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

from train_direct_pwm_bench import PWM_LIMIT, StandBench

DT = 0.01
ENC_RAD = 2 * np.pi / 60000
GYRO_RAD = np.pi / (180 * 16.4)
STRESS = {
    "nominal": (0.0, 0.0),
    "push": (0.2, 0.5),
    "stress": (0.36, 1.0),
}


class RobustStand(StandBench):
    def __init__(self, seed, push, noise, seconds=20, drift_weight=110, slew_weight=0.35, limit_cost=0.0, pwm_cost=0.15):
        super().__init__(seed, push)
        self.plant.max_steps = round(seconds / DT)
        self.noise = noise
        self.drift_weight = drift_weight
        self.slew_weight = slew_weight
        self.limit_cost = limit_cost
        self.pwm_cost = pwm_cost

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed, options=options)
        rng = self.plant.rng
        self.odom = 0.0
        self.previous = np.zeros(2)
        self.pitch_bias = rng.uniform(-0.0026, 0.0026) * self.noise
        self.gyro_bias = rng.uniform(-0.018, 0.018) * self.noise
        self.motor_trim = 1 + rng.uniform(-0.12, 0.12, 2) * self.noise
        self.deadband = rng.uniform(0.0, 0.035, 2) * self.noise
        # 保证起始观测仍在固件可解锁区间。
        pitch = self.plant.data.qpos[2]
        self.plant.data.qpos[2] = np.clip(pitch + self.pitch_bias, -0.04, 0.04) - self.pitch_bias
        mujoco.mj_forward(self.plant.model, self.plant.data)
        return self.read_sensors(advance=False)[0], {}

    def read_sensors(self, *, advance):
        q, v = self.plant.data.qpos, self.plant.data.qvel
        rng = self.plant.rng
        pitch = q[2] + self.pitch_bias + rng.normal(0, 0.00065) * self.noise
        rate = v[2] + self.gyro_bias + rng.normal(0, 0.008) * self.noise
        rate = np.rint(rate / GYRO_RAD) * GYRO_RAD
        wheel = np.rint(v[3:5] * 0.005 / ENC_RAD) * ENC_RAD / 0.005
        vx = 0.0335 * (wheel.mean() + rate)
        if advance:
            self.odom += vx * DT
        obs = np.array([pitch * 4, rate * 0.5, wheel[0] / 20, wheel[1] / 20, 2 * self.odom, 2 * vx],
                       dtype=np.float32)
        return np.clip(obs, -10, 10), pitch, wheel

    def step(self, action):
        pwm = np.clip(action, -PWM_LIMIT, PWM_LIMIT)
        torque_fraction = np.sign(pwm) * np.maximum(np.abs(pwm) - self.deadband, 0)
        self.plant.step(torque_fraction * self.motor_trim)
        obs, pitch, wheel = self.read_sensors(advance=True)
        q, v = self.plant.data.qpos, self.plant.data.qvel
        fault = ""
        if max(abs(pitch), abs(q[2])) > 0.09:
            fault = "pitch"
        elif max(abs(self.odom), abs(q[0])) > 0.20:
            fault = "travel"
        elif np.max(np.abs(wheel)) > 16:
            fault = "wheel_rate"

        near_limit = np.maximum(np.abs(action) - 0.8 * PWM_LIMIT, 0.0)
        reward = (1.0 - 65 * q[2] ** 2 - self.drift_weight * q[0] ** 2 - 12 * v[0] ** 2
                  - self.pwm_cost * np.mean(np.square(action))
                  - self.slew_weight * np.mean(np.square(pwm - self.previous))
                  - 0.15 * (pwm[0] - pwm[1]) ** 2
                  - self.limit_cost * np.mean(near_limit ** 2))
        self.previous = pwm
        if fault:
            reward -= 12
        info = {"fault": fault, "pitch": float(q[2]), "travel": float(q[0]),
                "odom": float(self.odom), "pwm": pwm}
        return obs, float(reward), bool(fault), self.plant.steps >= self.plant.max_steps, info


def evaluate(model, norm, seed, count, seconds, scenario):
    push, noise = STRESS[scenario]
    envs = [RobustStand(seed + ep, push, noise, seconds=seconds) for ep in range(count)]
    observations = np.stack([env.reset()[0] for env in envs])
    peak = np.zeros(count)
    lengths = np.zeros(count, dtype=int)
    power = pitch_square = sat = delta = 0.0
    previous = np.zeros((count, 2))
    active = list(range(count))
    faults = {}
    for k in range(round(seconds / DT)):
        if not active:
            break
        actions, _ = model.predict(norm.normalize_obs(observations[active]), deterministic=True)
        next_active = []
        for ep, action in zip(active, actions):
            observations[ep], _, terminated, truncated, info = envs[ep].step(action)
            lengths[ep] = k + 1
            peak[ep] = max(peak[ep], abs(info["travel"]))
            pwm = info["pwm"]
            power += np.dot(pwm, pwm) / 2
            change = pwm - previous[ep]
            delta += np.dot(change, change) / 2
            previous[ep] = pwm
            pitch_square += info["pitch"] ** 2
            sat += np.count_nonzero(np.abs(pwm) >= PWM_LIMIT - 1e-6) / 2
            if terminated or truncated:
                reason = info["fault"] or "full"
                faults[reason] = faults.get(reason, 0) + 1
            else:
                next_active.append(ep)
        active = next_active
    samples = lengths.sum()
    return {"scenario": scenario, "episodes": count, "seconds": seconds,
            "full": faults.get("full", 0), "faults": faults,
            "mean_seconds": float(lengths.mean() * DT), "worst_seconds": float(lengths.min() * DT),
            "p95_peak_travel_m": float(np.quantile(peak, 0.95)), "max_travel_m": float(np.max(peak)),
            "pitch_rms_deg": float(np.rad2deg(np.sqrt(pitch_square / samples))),
            "pwm_rms": float(np.sqrt(power / samples) * 6900),
            "pwm_change_rms": float(np.sqrt(delta / samples) * 6900),
            "pwm_saturated_pct": float(100 * sat / samples)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("norm", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--steps", type=int, default=32768)
    parser.add_argument("--envs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=40333)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seconds", type=float, default=20)
    parser.add_argument("--drift", type=float, default=170)
    parser.add_argument("--slew", type=float, default=0.6)
    parser.add_argument("--noise", type=float, default=0.7)
    parser.add_argument("--push", type=float, default=0.25)
    parser.add_argument("--pwm-cost", type=float, default=0.15)
    parser.add_argument("--limit-cost", type=float, default=0.0)
    parser.add_argument("--exploration", type=float, default=-2.4)
    parser.add_argument("--lr", type=float, default=8e-5)
    parser.add_argument("--scenario", nargs="+", choices=STRESS, default=tuple(STRESS))
    args = parser.parse_args()

    torch.set_num_threads(1)
    model = PPO.load(args.model, device="cpu")
    # 读取配套归一化参数，评估时冻结统计。
    with args.norm.open("rb") as handle:
        norm = pickle.load(handle)

    if args.out:
        args.out.mkdir(parents=True, exist_ok=False)
        print(f"Training {args.steps} steps from {args.model.name}", flush=True)
        vec = SubprocVecEnv([
            partial(RobustStand, args.seed + i * 19, args.push * (i % 4) / 3, args.noise,
                    seconds=16, drift_weight=args.drift, slew_weight=args.slew,
                    limit_cost=args.limit_cost, pwm_cost=args.pwm_cost)
            for i in range(args.envs)
        ], start_method="fork")
        try:
            norm = VecNormalize.load(args.norm, vec)
            norm.norm_reward = False
            norm.training = False
            model = PPO.load(args.model, env=norm, device="cpu", n_steps=256, batch_size=256,
                             n_epochs=6, learning_rate=args.lr, target_kl=0.018, seed=args.seed)
            # 续训使用新的优化器，避免旧任务的动量影响这一轮的小步更新。
            model.policy.optimizer.state.clear()
            with torch.no_grad():
                model.policy.log_std.fill_(args.exploration)
            model.learn(args.steps, reset_num_timesteps=False)
            model.save(args.out / "policy.zip")
            norm.save(args.out / "vecnormalize.pkl")
            config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
            (args.out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
            print("Checkpoint saved", flush=True)
        finally:
            vec.close()

    stats = []
    for scenario in args.scenario:
        result = evaluate(model, norm, args.seed + 70000, args.episodes, args.seconds, scenario)
        stats.append(result)
        print(json.dumps(result), flush=True)
    if args.out:
        (args.out / "evaluation.json").write_text(json.dumps(stats, indent=2) + "\n")


if __name__ == "__main__":
    main()
