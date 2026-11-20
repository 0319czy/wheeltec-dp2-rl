"""无杆版（一阶）平衡环境 v2：长时长 + 干扰 + 域随机化。

状态 6 维：[pitch, pitch_rate, wl, wr, pos_x, vel_x]
动作 2 维：左右轮力矩（归一化）
干扰：随机推力脉冲、初始倾角、电机增益/摩擦/延迟/观测噪声随机化
回合长度：默认 2000 步 = 20 秒
"""
import numpy as np
import mujoco
from pathlib import Path

XML = str(Path(__file__).resolve().parent / 'dp2_nopole.xml')
CTRL_DT = 0.01
TORQUE_MAX = 0.0402


class NoPoleBalanceV2:
    obs_dim = 6
    act_dim = 2

    def __init__(self, seed=0, push_strength=0.0, max_steps=2000, randomize=True, obs_noise=0.0):
        self.model = mujoco.MjModel.from_xml_path(XML)
        self.data = mujoco.MjData(self.model)
        self.rng = np.random.default_rng(seed)
        self.push_strength = push_strength     # 0=无干扰，1=强干扰
        self.max_steps = max_steps
        self.randomize = randomize
        self.obs_noise = obs_noise
        self.reset()

    def reset(self):
        rng = self.rng
        mujoco.mj_resetData(self.model, self.data)
        # 初始倾角：无干扰时 ±3°，有干扰时 ±6°
        p = rng.uniform(-0.10, 0.10) if self.randomize else rng.uniform(-0.03, 0.03)
        self.data.qpos[0] = rng.uniform(-0.2, 0.2)
        self.data.qpos[2] = p
        self.data.qvel[:] = 0.0
        self.steps = 0
        self.prev_action = np.zeros(2, dtype=np.float32)
        self.delay = int(rng.integers(0, 3)) if self.randomize else 0
        self.action_queue = [np.zeros(2, dtype=np.float32)] * (self.delay + 1)
        # 域随机化：电机增益、摩擦、身体质量
        self.motor_gain = rng.uniform(0.8, 1.2) if self.randomize else 1.0
        self.model.geom_friction[:, 0] = rng.uniform(0.6, 1.1) if self.randomize else 0.9
        self.model.body_mass[1] = 0.9 * (rng.uniform(0.9, 1.1) if self.randomize else 1.0)
        self.next_push = int(rng.integers(100, 300)) if self.push_strength > 0 else 10**9
        self.push_left = 0
        self.push_force = 0.0
        mujoco.mj_forward(self.model, self.data)
        return self._obs()

    def _obs(self):
        q = self.data.qpos; v = self.data.qvel
        o = np.array([q[2] * 4.0, v[2] * 0.5, v[3] / 20.0, v[4] / 20.0, q[0] / 0.5, v[0] / 0.5],
                     dtype=np.float32)
        if self.obs_noise:
            o = o + self.rng.normal(0, self.obs_noise, o.shape).astype(np.float32)
        return np.clip(o, -10, 10)

    def step(self, action):
        a = np.clip(np.asarray(action, dtype=np.float32), -1, 1)
        self.action_queue.append(a)
        delayed = self.action_queue.pop(0)
        torque = delayed * TORQUE_MAX * self.motor_gain
        steps = int(round(CTRL_DT / self.model.opt.timestep))
        for _ in range(steps):
            self.data.ctrl[:] = torque
            if self.push_left > 0:
                # 水平推力直接作用在车身上（模拟推一把）
                self.data.xfrc_applied[1, 0] = self.push_force
                self.push_left -= 1
            else:
                self.data.xfrc_applied[1, 0] = 0.0
            mujoco.mj_step(self.model, self.data)
        self.steps += 1
        if self.steps >= self.next_push and self.push_strength > 0:
            self.push_force = self.rng.choice([-1, 1]) * self.push_strength * self.rng.uniform(0.5, 1.5) * 3.0
            self.push_left = int(self.rng.integers(3, 9))
            self.next_push = self.steps + int(self.rng.integers(100, 350))

        q = self.data.qpos; v = self.data.qvel
        pitch, pos = q[2], q[0]
        rew = 2.0 * np.exp(-12.0 * pitch ** 2)
        rew -= 3.0 * (1.0 - np.exp(-12.0 * pitch ** 2))
        rew -= 0.03 * (pos ** 2) + 0.01 * (v[0] ** 2)
        rew -= 0.002 * float(np.sum((torque / TORQUE_MAX) ** 2))
        rew -= 0.01 * float(np.sum((a - self.prev_action) ** 2))
        self.prev_action = a.copy()
        upright = abs(pitch) < 0.35
        done = (not upright) or abs(pos) > 3.0 or self.steps >= self.max_steps
        return self._obs(), float(rew), done, dict(pitch=float(pitch), pos=float(pos))
