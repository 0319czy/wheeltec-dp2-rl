"""用二阶 DP2 仿真回合的回报搜索纯 RL 输出层参数。"""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO

from train_second_order_pwm import SecondOrderDP2, PWM_SCALE

NETWORK = None
BASIS = None


def hidden(obs):
    w1, b1, w2, b2, _, _ = NETWORK
    return np.tanh(w2 @ np.tanh(w1 @ obs + b1) + b2)


def candidate(params):
    return NETWORK[4] + BASIS @ params[:-1], NETWORK[5] + params[-1]


def score(task):
    params, seed, level, push, seconds, trials = task
    action_weights, bias = candidate(params)
    outcomes = []
    for i in range(trials):
        strength = 0 if i == 0 else push
        env = SecondOrderDP2(seed + i, level, seconds, PWM_SCALE, strength, 0.20, early_push=True)
        observation, _ = env.reset()
        max_travel = 0
        movement = 0
        steps = 0
        for _ in range(env.steps_limit):
            action = np.clip(action_weights @ hidden(observation) + bias, -1, 1)
            observation, _, fallen, expired, info = env.step(action)
            steps += 1
            max_travel = max(max_travel, abs(info["travel"]))
            movement += abs(info["body"]) + abs(info["pole"])
            if fallen or expired:
                break
        duration = steps / env.steps_limit
        outcomes.append(duration + (0.18 if not fallen else 0) - 0.28 * max_travel - 0.6 * movement / steps)
    return float(np.mean(outcomes))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--seed", type=int, default=13151)
    parser.add_argument("--level", type=int, choices=(0, 1, 2), default=1)
    parser.add_argument("--trials", type=int, default=6)
    parser.add_argument("--seconds", type=float, default=8)
    parser.add_argument("--seed-banks", type=int, default=3)
    parser.add_argument("--initial-scale", type=float, default=0.035)
    parser.add_argument("--push", type=float, default=0.12)
    args = parser.parse_args()
    torch.set_num_threads(1)
    args.out.mkdir()
    (args.out / "config.json").write_text(json.dumps(vars(args), default=str, indent=2) + "\n")
    model = PPO.load(args.model, device="cpu")
    actor = model.policy.mlp_extractor.policy_net
    head = model.policy.action_net
    global NETWORK, BASIS
    NETWORK = tuple(t.detach().cpu().numpy().copy() for t in (
        actor[0].weight, actor[0].bias, actor[2].weight, actor[2].bias, head.weight, head.bias
    ))
    # 在网络自身的局部特征方向上搜索；为两轮输出添加相同的权重修正。
    zero = np.zeros(8)
    probes = np.eye(8) * 0.08
    features = np.column_stack([hidden(zero), *(hidden(s) - hidden(-s) for s in probes)])
    BASIS = np.linalg.qr(features)[0].astype(np.float32)
    rng = np.random.default_rng(args.seed)
    center = np.zeros(BASIS.shape[1] + 1, dtype=np.float32)
    elite = center.copy()
    scale = np.full_like(center, args.initial_scale)
    scale[-1] = args.initial_scale / 7
    history = []
    with ProcessPoolExecutor(max_workers=8) as pool:
        for generation in range(1, args.rounds + 1):
            proposals = rng.normal(size=(10, len(center))).astype(np.float32) * scale
            population = np.concatenate(([center, elite], center + proposals, center - proposals))
            seed = args.seed + 900 + 29 * (generation % args.seed_banks)
            scores = list(pool.map(score, [(x, seed, args.level, args.push, args.seconds, args.trials) for x in population]))
            winners = np.argsort(scores)[-4:]
            elite = population[winners[-1]].copy()
            center = 0.5 * elite + 0.5 * population[winners].mean(axis=0)
            scale *= 0.88 if scores[winners[-1]] > scores[0] + 0.03 else 0.96
            weight, bias = candidate(elite)
            with torch.no_grad():
                head.weight.copy_(torch.from_numpy(weight))
                head.bias.copy_(torch.from_numpy(bias))
            model.policy.optimizer.state.clear()
            model.save(args.out / f"policy_{generation}.zip")
            record = {"generation": generation, "seed": seed, "baseline": scores[0], "best": max(scores),
                      "proposals": len(scores), "episodes_each": args.trials}
            history.append(record)
            (args.out / "search.json").write_text(json.dumps(history, indent=2) + "\n")
            print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
