#!/usr/bin/env python3
"""项目自检单元测试(标准库 unittest,不依赖 pytest)。

运行:
  .venv/bin/python -m unittest discover -s tests -v

覆盖:
  1. sim.env_vla.success_func —— 12 条指令的成功判据(各给一个通过/不通过样例)
  2. sim.env_vla.wrap_angle —— 角度回绕
  3. vla.model.normalize_actions / denormalize —— 动作归一化往返与裁剪
  4. deploy.bridge.encode_frame —— vw / param / legacy 三种协议帧格式
"""
from __future__ import annotations

import math
import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from sim.env_vla import success_func, wrap_angle  # noqa: E402
from vla.model import denormalize, normalize_actions  # noqa: E402
from deploy.bridge import encode_frame  # noqa: E402


def make_info(**kw):
    """构造一个 info dict 的最小骨架(其余字段给中性默认值)。"""
    base = {
        "success": False,
        "fall": False,
        "pos": (0.0, 0.0),
        "start_pos": (0.0, 0.0),
        "start_yaw": 0.0,
        "cum_dyaw": 0.0,
        "dist": {0: 2.0, 1: 2.0, 2: 2.0},
        "heading_err_by_color": {0: 1.0, 1: 1.0, 2: 1.0},
    }
    base.update(kw)
    return base


class TestSuccessFunc(unittest.TestCase):
    """每条指令一个通过样例 + 一个不通过样例。"""

    def check(self, instr, ok, bad):
        self.assertTrue(success_func(instr, ok, [ok]), "instr %d 应判成功" % instr)
        self.assertFalse(success_func(instr, bad, [bad]), "instr %d 应判失败" % instr)

    def test_0_stand_still(self):
        self.check(0,
                   make_info(pos=(0.05, 0.0), cum_dyaw=0.1),
                   make_info(pos=(0.30, 0.0)))

    def test_1_forward(self):
        # start_yaw=0 → forward = Δx
        self.check(1, make_info(pos=(1.0, 0.0)), make_info(pos=(0.2, 0.0)))

    def test_2_backward(self):
        self.check(2, make_info(pos=(-0.6, 0.0)), make_info(pos=(0.6, 0.0)))

    def test_3_turn_left(self):
        self.check(3, make_info(pos=(0.4, 0.2), cum_dyaw=2.0),
                   make_info(pos=(0.4, 0.2), cum_dyaw=-2.0))

    def test_4_turn_right(self):
        self.check(4, make_info(pos=(0.4, 0.2), cum_dyaw=-2.0),
                   make_info(pos=(0.4, 0.2), cum_dyaw=2.0))

    def test_5_spin_left(self):
        # 原地左旋: dyaw 1.5–5.0 且相对起点位移 < 0.40 m
        self.check(5, make_info(pos=(0.05, 0.0), cum_dyaw=3.0),
                   make_info(pos=(0.60, 0.0), cum_dyaw=3.0))

    def test_6_spin_right(self):
        self.check(6, make_info(pos=(0.05, 0.0), cum_dyaw=-3.0),
                   make_info(pos=(0.60, 0.0), cum_dyaw=-3.0))

    def test_7_face_red(self):
        self.check(7, make_info(heading_err_by_color={0: 0.1, 1: 1.0, 2: 1.0}),
                   make_info(heading_err_by_color={0: 0.8, 1: 0.0, 2: 0.0}))

    def test_8_9_10_approach(self):
        self.check(8, make_info(dist={0: 0.40, 1: 2.0, 2: 2.0}),
                   make_info(dist={0: 0.60, 1: 2.0, 2: 2.0}))
        self.check(9, make_info(dist={0: 2.0, 1: 0.40, 2: 2.0}),
                   make_info(dist={0: 2.0, 1: 0.60, 2: 2.0}))
        self.check(10, make_info(dist={0: 2.0, 1: 2.0, 2: 0.40}),
                   make_info(dist={0: 2.0, 1: 2.0, 2: 0.60}))

    def test_11_detour(self):
        ok = make_info(dist={0: 0.8, 1: 0.45, 2: 2.0})
        hist = [make_info(dist={0: 1.2, 1: 0.6, 2: 2.0}), ok]
        self.assertTrue(success_func(11, ok, hist))
        # 红桶曾在 0.3 m 以内(离得太近) → 不通过
        close = make_info(dist={0: 0.3, 1: 0.6, 2: 2.0})
        bad = make_info(dist={0: 0.8, 1: 0.45, 2: 2.0})
        self.assertFalse(success_func(11, bad, [close, bad]))
        # 蓝桶没靠近 → 不通过
        self.assertFalse(success_func(11, make_info(dist={0: 1.0, 1: 0.9, 2: 2.0}),
                                      [make_info(dist={0: 1.0, 1: 0.9, 2: 2.0})]))

    def test_fall_always_fails(self):
        for instr in range(12):
            self.assertFalse(success_func(instr, make_info(fall=True), [make_info(fall=True)]))

    def test_unknown_instruction_raises(self):
        with self.assertRaises(ValueError):
            success_func(99, make_info(), [make_info()])


class TestWrapAngle(unittest.TestCase):
    def test_wrap(self):
        self.assertAlmostEqual(wrap_angle(0.0), 0.0)
        self.assertAlmostEqual(wrap_angle(math.pi - 0.01), math.pi - 0.01)
        self.assertAlmostEqual(wrap_angle(math.pi + 0.01), -math.pi + 0.01)
        self.assertAlmostEqual(wrap_angle(-math.pi - 0.01), math.pi - 0.01)
        self.assertAlmostEqual(wrap_angle(6 * math.pi + 0.5), 0.5)


class TestActionNormalization(unittest.TestCase):
    def test_roundtrip(self):
        import torch
        phys = torch.tensor([[0.2, -0.7], [-0.35, 1.0], [0.0, 0.0]], dtype=torch.float32)
        norm = normalize_actions(phys)
        back = denormalize(norm)
        torch.testing.assert_close(back, phys, atol=1e-6, rtol=0)

    def test_clamp_and_scale(self):
        import torch
        norm = normalize_actions(torch.tensor([[1.0, 2.0]], dtype=torch.float32))  # 超量程 → clamp 到 1
        torch.testing.assert_close(norm, torch.tensor([[1.0, 1.0]], dtype=torch.float32))
        self.assertAlmostEqual(float(denormalize(torch.tensor([[0.5, -0.5]]))[0, 0]), 0.5 * 0.35, places=6)


class TestBridgeProtocol(unittest.TestCase):
    def test_vw(self):
        self.assertEqual(encode_frame(0.25, 0.0, "vw"), "#V250,0;")
        self.assertEqual(encode_frame(-0.1, 0.5, "vw"), "#V-100,500;")

    def test_param(self):
        self.assertEqual(encode_frame(0.25, -0.3, "param"), "#vc=0.250;#wc=-0.300;")

    def test_legacy_chars(self):
        # 前进 A / 后退 E / 左转 G / 右转 C / 静止 Z
        self.assertEqual(encode_frame(0.3, 0.0, "legacy"), "A")
        self.assertEqual(encode_frame(-0.3, 0.0, "legacy"), "E")
        self.assertEqual(encode_frame(0.0, 0.5, "legacy"), "G")
        self.assertEqual(encode_frame(0.0, -0.5, "legacy"), "C")
        self.assertEqual(encode_frame(0.0, 0.0, "legacy"), "Z")


if __name__ == "__main__":
    unittest.main(verbosity=2)
