"""Demo: 2D LiDAR scan geometry on top of a simulated episode.

Runs the shipped policy in MuJoCo, samples the robot pose every 0.5 s and casts
N10-like rays (0.5 deg resolution, 8 m range) against a rectangular room.
This is a geometry/visualisation demo, NOT SLAM: no mapping, no loop closure.
Writes media/demo_lidar_scan.png.
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mujoco
from stable_baselines3 import PPO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'mujoco'))
from train_direct_pwm_robust import RobustStand  # noqa: E402

ROOM = dict(x0=-2.0, x1=2.0, y0=-1.5, y1=1.5)
RAYS = np.deg2rad(np.arange(-135, 135, 0.5))
RANGE_MAX = 8.0


OBSTACLES = [(-0.35, 0.30, 0.55, 0.80),   # x0,y0,x1,y1  (a box in the room)
             (0.60, -0.90, 0.75, -0.45)]


def walls():
    r = ROOM
    segs = [((r['x0'], r['y0']), (r['x1'], r['y0'])),
            ((r['x1'], r['y0']), (r['x1'], r['y1'])),
            ((r['x1'], r['y1']), (r['x0'], r['y1'])),
            ((r['x0'], r['y1']), (r['x0'], r['y0']))]
    for x0, y0, x1, y1 in OBSTACLES:
        segs += [((x0, y0), (x1, y0)), ((x1, y0), (x1, y1)),
                 ((x1, y1), (x0, y1)), ((x0, y1), (x0, y0))]
    return segs


def cast(origin, angles):
    """Ray-cast 2D rays against axis-aligned room walls; returns distances."""
    ox, oy = origin
    dist = np.full(len(angles), RANGE_MAX)
    dx, dy = np.cos(angles), np.sin(angles)
    for (x0, y0), (x1, y1) in walls():
        if y0 == y1:      # horizontal wall
            t = (y0 - oy) / np.where(np.abs(dy) < 1e-9, 1e-9, dy)
            hit = (ox + t * dx >= min(x0, x1)) & (ox + t * dx <= max(x0, x1))
        else:             # vertical wall
            t = (x0 - ox) / np.where(np.abs(dx) < 1e-9, 1e-9, dx)
            hit = (oy + t * dy >= min(y0, y1)) & (oy + t * dy <= max(y0, y1))
        t = np.where(hit & (t > 0) & (t < dist), t, dist)
        dist = np.minimum(dist, t)
    return dist


def main():
    model = PPO.load(ROOT / 'models/first-order/policy.zip', device='cpu')
    with (ROOT / 'models/first-order/vecnormalize.pkl').open('rb') as f:
        norm = pickle.load(f)

    env = RobustStand(620001, push=0.0, noise=0.0, seconds=6.0)
    env.reset()
    obs, *_ = env.read_sensors(advance=False)
    pts, poses = [], []
    for step in range(600):
        action, _ = model.predict(norm.normalize_obs(np.asarray([obs])), deterministic=True)
        obs, _, failed, done, info = env.step(action[0])
        if step % 50 == 0 and not failed:
            x = float(env.plant.data.qpos[0])
            yaw = 0.0                     # planar model: heading locked to +x
            d = cast((x, 0.0), RAYS + yaw)
            m = d < RANGE_MAX
            pts.append(np.stack([x + d[m] * np.cos(RAYS[m] + yaw),
                                 d[m] * np.sin(RAYS[m] + yaw)], axis=1))
            poses.append((x, 0.0))
        if failed or done:
            break

    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    allp = np.vstack(pts) if pts else np.zeros((0, 2))
    if len(allp):
        ax.scatter(allp[:, 0], allp[:, 1], s=2.0, c='tab:blue', alpha=.55,
                   zorder=3, label='scan points')
    if poses:
        px, py = zip(*poses)
        ax.plot(px, py, 'r.-', lw=1, label='robot path')
    ax.add_patch(plt.Rectangle((ROOM['x0'], ROOM['y0']),
                               ROOM['x1'] - ROOM['x0'], ROOM['y1'] - ROOM['y0'],
                               fill=False, lw=1.0, color='k', zorder=4))
    for x0, y0, x1, y1 in OBSTACLES:
        ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0,
                                   fill=False, lw=1.0, color='0.35', zorder=4))
    ax.set_aspect('equal'); ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)')
    ax.set_title('2D scan demo: rays vs room walls (no SLAM)')
    ax.legend(loc='upper right', fontsize=8); ax.grid(alpha=.2)
    media = ROOT / 'media/demo_lidar_scan.png'
    fig.tight_layout(); fig.savefig(media, dpi=150)
    print(f'{len(pts)} scans, {sum(len(p) for p in pts)} points -> {media.relative_to(ROOT)}')


if __name__ == '__main__':
    main()
