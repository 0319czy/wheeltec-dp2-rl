"""MuJoCo 带杆二阶 DP2，纯 PPO 原地站立，动作直接映射到 PWM。"""
import argparse
import json
from functools import partial
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv

XML = Path(__file__).with_name("dp2_second_order.xml")
DT = 0.01
WHEEL_RADIUS = 0.0335
PWM_FULL = 6900
PWM_SCALE = 4500
TORQUE_MAX = 0.0402


class SecondOrderDP2(gym.Env):
    def __init__(self, seed, level=0, seconds=8, pwm_limit=PWM_SCALE, push=None, travel_limit=0.25, travel_cost=2.8, early_push=False):
        self.model = mujoco.MjModel.from_xml_path(str(XML))
        self.data = mujoco.MjData(self.model)
        self.rng = np.random.default_rng(seed)
        self.level = level
        self.steps_limit = round(seconds / DT)
        self.pwm_limit = pwm_limit
        self.travel_limit = travel_limit
        self.travel_cost = travel_cost
        self.early_push = early_push
        self.push = 0 if push is None and level < 2 else (0.2 if push is None else push)
        self.observation_space = gym.spaces.Box(-10, 10, (8,), np.float32)
        self.action_space = gym.spaces.Box(-1, 1, (2,), np.float32)
        self.friction_nominal = self.model.geom_friction[:, 0].copy()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        rng = self.rng
        mujoco.mj_resetData(self.model, self.data)
        angle = [0.010, 0.024, 0.039][self.level]
        body, pole = rng.uniform(-angle, angle, size=2)
        self.data.qpos[2] = body
        self.data.qpos[5] = pole - body
        self.motor_gain = rng.uniform(1 - 0.08 * self.level, 1 + 0.08 * self.level, 2)
        self.model.geom_friction[:, 0] = self.friction_nominal * rng.uniform(
            1 - 0.08 * self.level, 1 + 0.08 * self.level)
        self.bias = rng.uniform(-0.001, 0.001, 2) * self.level
        self.delay = int(rng.integers(0, self.level + 1))
        self.last_pwm = np.zeros(2)
        self.queue = [self.last_pwm.copy() for _ in range(self.delay)]
        self.steps = 0
        self.odom = 0.0
        self.next_push = rng.integers(25, 70) if self.early_push else rng.integers(120, 250)
        self.push_left = 0
        self.force = 0.0
        self.push_body = 1
        mujoco.mj_forward(self.model, self.data)
        return self.observe(), {}

    def observe(self):
        q, v = self.data.qpos, self.data.qvel
        # 摆杆铰链是相对角，观测使用相对重力竖直的绝对角、角速度。
        body = q[2] + self.bias[0]
        pole = q[2] + q[5] + self.bias[1]
        vx = WHEEL_RADIUS * (0.5 * (v[3] + v[4]) + v[2])
        obs = [body / 0.05, v[2] / 0.8, v[3] / 4, v[4] / 4,
               self.odom / 0.16, vx / 0.20, pole / 0.05, (v[2] + v[5]) / 0.8]
        return np.clip(np.asarray(obs, dtype=np.float32), -10, 10)

    def step(self, action):
        pwm = np.clip(action * PWM_SCALE, -self.pwm_limit, self.pwm_limit)
        self.queue.append(pwm)
        applied = self.queue.pop(0)
        self.data.ctrl[:] = applied * (TORQUE_MAX / PWM_FULL) * self.motor_gain
        self.data.xfrc_applied[:, 0] = 0.0
        if self.push_left:
            self.data.xfrc_applied[self.push_body, 0] = self.force
            self.push_left -= 1
        mujoco.mj_step(self.model, self.data, nstep=2)
        self.steps += 1
        if self.push > 0 and self.steps >= self.next_push:
            self.push_body = int(self.rng.choice([1, 4]))
            self.force = self.rng.choice([-1, 1]) * self.push * self.rng.uniform(0.7, 1.3)
            self.push_left = int(self.rng.integers(4, 8))
            self.next_push += self.rng.integers(140, 380)

        q, v = self.data.qpos, self.data.qvel
        body, pole = q[2], q[2] + q[5]
        self.odom += WHEEL_RADIUS * (0.5 * (v[3] + v[4]) + v[2]) * DT
        failure = ""
        if abs(body) > 0.14:
            failure = "body_tilt"
        elif abs(pole) > 0.19:
            failure = "pole_tilt"
        elif max(abs(q[0]), abs(self.odom)) > self.travel_limit:
            failure = "travel"
        elif max(abs(v[3]), abs(v[4])) > 18:
            failure = "wheel_speed"

        duty = pwm / PWM_SCALE
        slew = (pwm - self.last_pwm) / PWM_SCALE
        reward = (1.0 - 12 * abs(body) - 15 * abs(pole) - self.travel_cost * abs(q[0])
                  - 2.0 * v[0] ** 2 - 0.045 * (v[2] ** 2 + (v[2] + v[5]) ** 2)
                  - 0.055 * np.mean(duty ** 2) - 0.08 * np.mean(slew ** 2)
                  - 0.12 * (duty[0] - duty[1]) ** 2)
        self.last_pwm = pwm
        if failure:
            reward -= 8
        return self.observe(), float(reward), bool(failure), self.steps >= self.steps_limit, {
            "reason": failure, "body": body, "pole": pole, "travel": q[0], "pwm": pwm
        }


def evaluate(model, *, seed, episodes, seconds, level, push, pwm_limit, travel_limit):
    envs = [SecondOrderDP2(seed + i, level, seconds, pwm_limit, push, travel_limit) for i in range(episodes)]
    obs = np.stack([env.reset()[0] for env in envs])
    active = list(range(episodes))
    duration = np.zeros(episodes)
    peak_travel = np.zeros(episodes)
    angles = np.zeros(2)
    pwm_sq, pwm_delta_sq, samples = 0.0, 0.0, 0
    last_pwm = np.zeros((episodes, 2))
    faults = {}
    for step in range(round(seconds / DT)):
        if not active:
            break
        actions, _ = model.predict(obs[active], deterministic=True)
        alive = []
        for ep, action in zip(active, actions):
            obs[ep], _, ended, expired, info = envs[ep].step(action)
            duration[ep] = (step + 1) * DT
            peak_travel[ep] = max(peak_travel[ep], abs(info["travel"]))
            angles += [info["body"] ** 2, info["pole"] ** 2]
            pwm_sq += np.mean(info["pwm"] ** 2)
            pwm_delta_sq += np.mean((info["pwm"] - last_pwm[ep]) ** 2)
            last_pwm[ep] = info["pwm"]
            samples += 1
            if ended or expired:
                verdict = info["reason"] or "full"
                faults[verdict] = faults.get(verdict, 0) + 1
            else:
                alive.append(ep)
        active = alive
    return {
        "seed": seed, "episodes": episodes, "seconds": seconds, "level": level,
        "push_N": push, "pwm_limit": pwm_limit, "travel_limit_m": travel_limit,
        "full": faults.get("full", 0), "reasons": faults, "mean_seconds": float(duration.mean()),
        "min_seconds": float(duration.min()), "travel_peak_p95_m": float(np.quantile(peak_travel, 0.95)),
        "body_pitch_rms_deg": float(np.rad2deg(np.sqrt(angles[0] / samples))),
        "pole_pitch_rms_deg": float(np.rad2deg(np.sqrt(angles[1] / samples))),
        "pwm_rms": float(np.sqrt(pwm_sq / samples)), "pwm_jump_rms": float(np.sqrt(pwm_delta_sq / samples))
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--level", type=int, choices=range(3), default=0)
    parser.add_argument("--steps", type=int, default=245760)
    parser.add_argument("--chunk", type=int, default=61440)
    parser.add_argument("--envs", type=int, default=12)
    parser.add_argument("--seed", type=int, default=27121)
    parser.add_argument("--lr", type=float, default=0.0003)
    parser.add_argument("--exploration", type=float, default=-2.7)
    parser.add_argument("--seconds", type=float, default=8)
    parser.add_argument("--episodes", type=int, default=16)
    parser.add_argument("--pwm-limit", type=int, default=PWM_SCALE)
    parser.add_argument("--travel-limit", type=float, default=0.25)
    parser.add_argument("--push", type=float)
    parser.add_argument("--travel-cost", type=float, default=2.8)
    parser.add_argument("--early-push", action="store_true", help="训练时把首个推扰前移")
    parser.add_argument("--mix", action="store_true", help="混合当前难度与上一档")
    args = parser.parse_args()
    torch.set_num_threads(1)
    push = args.push if args.push is not None else (0.2 if args.level == 2 else 0)
    def assess(model, seed):
        return evaluate(model, seed=seed, episodes=args.episodes, seconds=args.seconds,
                        level=args.level, push=push, pwm_limit=args.pwm_limit, travel_limit=args.travel_limit)

    if args.out is None:
        if args.model is None:
            parser.error("--model is required for evaluation")
        model = PPO.load(args.model, device="cpu")
        print(json.dumps(assess(model, args.seed)), flush=True)
        return

    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "config.json").write_text(json.dumps(vars(args), default=str, indent=2) + "\n")
    vec = SubprocVecEnv([
        partial(SecondOrderDP2, args.seed + i * 11,
                max(0, args.level - 1) if args.mix and i % 3 == 0 else args.level,
                max(8, args.seconds), args.pwm_limit,
                0 if args.mix and i % 3 == 0 else push, args.travel_limit, args.travel_cost, args.early_push)
        for i in range(args.envs)
    ], start_method="fork")
    try:
        if args.model is None:
            model = PPO("MlpPolicy", vec, device="cpu", seed=args.seed, n_steps=256,
                        batch_size=256, n_epochs=7, gamma=0.995, gae_lambda=0.96,
                        learning_rate=args.lr, policy_kwargs={"net_arch": [64, 64]})
        else:
            model = PPO.load(args.model, env=vec, device="cpu", n_steps=256, batch_size=256,
                             n_epochs=7, learning_rate=args.lr, seed=args.seed, target_kl=0.02)
            model.policy.optimizer.state.clear()
        with torch.no_grad():
            model.policy.log_std.fill_(args.exploration)
        stats = []
        for index, completed in enumerate(range(0, args.steps, args.chunk), start=1):
            model.learn(min(args.steps - completed, args.chunk), reset_num_timesteps=False)
            model.save(args.out / f"policy_{index}.zip")
            result = assess(model, args.seed + 30000)
            result.update(checkpoint=index, total_steps=model.num_timesteps)
            stats.append(result)
            (args.out / "evaluation.json").write_text(json.dumps(stats, indent=2) + "\n")
            print(json.dumps(result), flush=True)
    finally:
        vec.close()


if __name__ == "__main__":
    main()
