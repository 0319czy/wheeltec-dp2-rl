"""微调无杆站立策略，动作直接映射到 PWM，不经过轮速 PI。"""
import argparse
from functools import partial
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from env_nopole_v2 import NoPoleBalanceV2

# 与纯 RL 台架固件一致。策略动作以满量程 PWM 为单位，由固件限幅。
PWM_LIMIT = 2500 / 6900
EPISODE_STEPS = 500


class StandBench(gym.Env):
    def __init__(self, seed, push=0.2):
        self.plant = NoPoleBalanceV2(
            seed=seed, push_strength=push, max_steps=EPISODE_STEPS, randomize=True
        )
        self.observation_space = gym.spaces.Box(-10, 10, (6,), np.float32)
        self.action_space = gym.spaces.Box(-1, 1, (2,), np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.plant.rng = np.random.default_rng(seed)
        self.plant.reset()
        self.plant.data.qpos[0] = 0
        self.plant.data.qpos[2] = self.plant.rng.uniform(-0.04, 0.04)
        mujoco.mj_forward(self.plant.model, self.plant.data)
        return self.plant._obs(), {}

    def step(self, action):
        pwm = np.clip(action, -PWM_LIMIT, PWM_LIMIT)
        observation, _, _, _ = self.plant.step(pwm)
        q, v = self.plant.data.qpos, self.plant.data.qvel
        fault = ""
        if abs(q[2]) > 0.09:
            fault = "pitch"
        elif abs(q[0]) > 0.20:
            fault = "travel"
        elif max(abs(v[3]), abs(v[4])) > 16:
            fault = "wheel_rate"
        reward = (
            1.0 - 65 * q[2] ** 2 - 110 * q[0] ** 2 - 12 * v[0] ** 2
            - 0.10 * np.mean(np.square(action))
        )
        if fault:
            reward -= 12
        return observation, float(reward), bool(fault), self.plant.steps >= EPISODE_STEPS, {"fault": fault}


def assess(model, norm, seed, push):
    env = StandBench(seed, push)
    durations, reasons = [], {}
    for _ in range(50):
        observation, _ = env.reset()
        for step in range(1, EPISODE_STEPS + 1):
            action, _ = model.predict(norm.normalize_obs(observation), deterministic=True)
            observation, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                reason = info["fault"] or "full"
                reasons[reason] = reasons.get(reason, 0) + 1
                durations.append(step * 0.01)
                break
    print(f"push={push}, {reasons}, mean={np.mean(durations):.3f}s, min={min(durations):.3f}s", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("norm", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--steps", type=int, default=163840)
    parser.add_argument("--fresh", action="store_true", help="从随机初始化开始训练")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)

    torch.set_num_threads(1)
    vec = SubprocVecEnv([partial(StandBench, 71 + i, 0.2 if i % 2 else 0) for i in range(8)],
                        start_method="fork")
    try:
        norm = VecNormalize.load(args.norm, vec)
        norm.norm_reward = False
        norm.training = False  # 冻结旧策略的观测归一化统计。
        if args.fresh:
            model = PPO("MlpPolicy", norm, device="cpu", n_steps=256, batch_size=256,
                        n_epochs=6, learning_rate=3e-4, target_kl=0.025, seed=101,
                        policy_kwargs={"net_arch": [64, 64], "log_std_init": -1.8})
        else:
            model = PPO.load(args.model, env=norm, device="cpu", n_steps=256, n_epochs=8,
                             learning_rate=3e-4, target_kl=0.025, verbose=0)
            with torch.no_grad():
                model.policy.log_std.fill_(-1.8)
        rounds = (args.steps + 40959) // 40960
        for i in range(rounds):
            model.learn(min(40960, args.steps - i * 40960), reset_num_timesteps=False)
            model.save(args.out / f"policy_{i + 1}.zip")
            norm.save(args.out / "vecnormalize.pkl")
            print(f"Checkpoint {i + 1}", flush=True)
            assess(model, norm, 5731, 0.0)
            assess(model, norm, 5731, 0.2)
    finally:
        vec.close()


if __name__ == "__main__":
    main()
