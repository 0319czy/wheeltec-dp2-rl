"""用教师策略的仿真轨迹训练低 PWM 跳变的纯 RL 策略。"""
import argparse
import json
import pickle
from pathlib import Path
from shutil import copyfile

import numpy as np
import torch
from stable_baselines3 import PPO

from train_direct_pwm_robust import PWM_LIMIT, STRESS, RobustStand


def collect(teacher, norm, seed):
    envs = [
        RobustStand(seed + i, *STRESS[scenario], seconds=16)
        for i, scenario in enumerate(["nominal", "push", "stress"] * 8)
    ]
    obs = np.stack([env.reset()[0] for env in envs])
    prior = obs.copy()
    ema = np.zeros((len(envs), 2), dtype=np.float32)
    first = np.ones(len(envs), dtype=bool)
    batches = []
    for _ in range(2100):
        now = norm.normalize_obs(obs)
        action, _ = teacher.predict(now, deterministic=True)
        action = np.clip(action, -PWM_LIMIT, PWM_LIMIT)
        ema = 0.48 * action + 0.52 * ema
        ema[first] = action[first]
        # 高角速度或大倾角时，优先保留教师的快速纠偏动作。
        recovery = np.maximum(np.abs(obs[:, 0]) / 0.16, np.abs(obs[:, 1]) / 0.6)
        steady = np.clip(1 - recovery, 0, 1).astype(np.float32)
        batches.append((now, norm.normalize_obs(prior), action, ema.copy(),
                        steady[:, None], (~first)[:, None]))
        prior = obs.copy()
        for i, (robot, pwm) in enumerate(zip(envs, action)):
            obs[i], _, ended, expired, _ = robot.step(pwm)
            first[i] = ended or expired
            if first[i]:
                obs[i] = robot.reset()[0]
                prior[i] = obs[i]
    return tuple(torch.from_numpy(np.concatenate(group)) for group in zip(*batches))


def train(args, data, name, fraction, penalty):
    result = args.out / name
    result.mkdir()
    student = PPO.load(args.model, device="cpu")
    actor = student.policy
    optimizer = torch.optim.Adam(
        [*actor.mlp_extractor.policy_net.parameters(), *actor.action_net.parameters()],
        lr=2e-4,
    )
    count = len(data[0])
    rng = torch.Generator().manual_seed(args.seed)
    for epoch in range(10):
        for idx in torch.randperm(count, generator=rng).split(512):
            now, prior, raw, filtered, steady, adjacent = (field[idx] for field in data)
            goal = raw + fraction * steady * (filtered - raw)
            action = actor.get_distribution(now).distribution.mean
            prev_action = actor.get_distribution(prior).distribution.mean
            tracking = (action - goal).square().mean()
            slew = ((action - prev_action).square() * steady * adjacent).mean()
            loss = tracking + penalty * slew
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if (epoch + 1) % 5 == 0:
            print(f"{name}: epoch {epoch + 1} / 10", flush=True)
    actor.optimizer.state.clear()
    student.save(result / "policy.zip")
    copyfile(args.norm, result / "vecnormalize.pkl")
    print(f"saved {name}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("norm", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--seed", type=int, default=89361)
    args = parser.parse_args()
    args.out.mkdir()
    torch.set_num_threads(1)
    with args.norm.open("rb") as handle:
        norm = pickle.load(handle)
    teacher = PPO.load(args.model, device="cpu")
    data = collect(teacher, norm, args.seed)
    print(f"Collected {len(data[0])} observations", flush=True)
    variants = {"moderate": (0.24, 0.15), "strong": (0.52, 0.38)}
    for name, (fraction, penalty) in variants.items():
        train(args, data, name, fraction, penalty)
    (args.out / "config.json").write_text(json.dumps({
        "teacher": str(args.model), "normalization": str(args.norm), "seed": args.seed,
        "observations": len(data[0]), "epochs": 10, "variants": variants,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
