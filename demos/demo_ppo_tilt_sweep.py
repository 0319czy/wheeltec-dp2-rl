"""Demo: sweep initial body tilt and measure how long the shipped PPO policy survives.

Runs entirely in MuJoCo (CPU, headless). Writes media/demo_tilt_sweep.png.
This is a simulation demo: numbers describe the policy in simulation only.
"""
import json, sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'mujoco'))

from stable_baselines3 import PPO          # noqa: E402
from train_direct_pwm_robust import RobustStand  # noqa: E402

XML = ROOT / 'mujoco' / 'dp2_nopole.xml'
MODEL = ROOT / 'models/first-order/policy.zip'
NORM = ROOT / 'models/first-order/vecnormalize.pkl'
SECONDS = 10.0


def run_policy(model, norm, degrees, episodes=8, seed0=610000):
    surv = []
    for ep in range(episodes):
        env = RobustStand(seed0 + ep, push=0.0, noise=0.0, seconds=SECONDS)
        env.reset()
        import mujoco
        env.plant.data.qpos[2] = np.deg2rad(degrees)
        env.plant.data.qvel[2] = 0.0
        mujoco.mj_forward(env.plant.model, env.plant.data)
        obs, *_ = env.read_sensors(advance=False)
        steps = 0
        while True:
            action, _ = model.predict(norm.normalize_obs(np.asarray([obs])), deterministic=True)
            obs, _, failed, done, info = env.step(action[0])
            steps += 1
            if failed or done:
                break
        surv.append(steps * 0.01)
    return float(np.mean(surv)), float(np.min(surv))


def main():
    model = PPO.load(MODEL, device='cpu')
    import pickle
    with NORM.open('rb') as f:
        norm = pickle.load(f)
    angles = [0, 1, 2, 3, 4, 6, 8]
    rows = []
    for deg in angles:
        mean_s, min_s = run_policy(model, norm, deg)
        rows.append(dict(deg=deg, mean_s=round(mean_s, 2), min_s=round(min_s, 2)))
        print(f'{deg:>3}° -> 平均 {mean_s:5.2f}s  最短 {min_s:5.2f}s', flush=True)

    out = ROOT / 'results/demo/ppo_tilt_sweep.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(xml=str(XML), seconds=SECONDS, rows=rows), indent=2))

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.plot([r['deg'] for r in rows], [r['mean_s'] for r in rows], 'o-', label='mean survival')
    ax.plot([r['deg'] for r in rows], [r['min_s'] for r in rows], 's--', alpha=.6, label='worst episode')
    ax.axhline(SECONDS, color='grey', lw=.8, ls=':')
    ax.set_xlabel('initial body tilt (deg)')
    ax.set_ylabel('survival time (s)')
    ax.set_title('PPO policy vs initial tilt (MuJoCo, no push)')
    ax.set_ylim(0, SECONDS * 1.05)
    ax.grid(alpha=.3); ax.legend()
    media = ROOT / 'media/demo_tilt_sweep.png'
    fig.tight_layout(); fig.savefig(media, dpi=150)
    print('saved', media.relative_to(ROOT))


if __name__ == '__main__':
    main()
