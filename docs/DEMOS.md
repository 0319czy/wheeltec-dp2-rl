# 可运行 demo（全部为仿真/几何演示，无需硬件）

三个 demo 都是**直接可跑**的脚本，输出图片与 JSON 落在 `media/` 与 `results/demo/`。
命令默认在仓库根目录执行，依赖见 [复现说明](REPRODUCE.md)。

## 1. PPO 策略 × 初始倾角扫描

```bash
python demos/demo_ppo_tilt_sweep.py
```

- 内容：把一阶策略（`models/first-order/policy.zip`）放到 MuJoCo 里，初始倾角从 0° 扫到 8°，
  每档 8 局、每局最多 10 s，统计平均/最短存活时间；
- 输出：`media/demo_tilt_sweep.png` + `results/demo/ppo_tilt_sweep.json`；
- **读数注意**：6°/8° 处 0.01 s 就"失败"，是台架俯仰保护阈值 0.09 rad（≈5.2°）先触发，
  不代表策略本身崩掉；真正有意义的是 0–4° 段。

实测输出：

```
  0° -> 平均 10.00s  最短 10.00s
  1° -> 平均 10.00s  最短 10.00s
  2° -> 平均 10.00s  最短 10.00s
  3° -> 平均  8.80s  最短  0.39s
  4° -> 平均  2.66s  最短  0.07s
```

这与"稳态可支撑最大倾角 ≈2.9°"的力学估算一致（见 [实机记录](REAL_ROBOT.md)）。

## 2. LQR 阶跃恢复 + 极点图

```bash
python demos/demo_lqr_step.py
```

- 内容：8 状态模型离散化后解 Riccati 得增益，仿真 2° 初始倾斜的闭环恢复；
- 输出：`media/demo_lqr_step.png`（左：倾角/摆杆角恢复曲线；右：z 平面极点）+ `results/demo/lqr_step.json`；
- 实测：`max|ΔK| vs 厂商 = 4.35e-05`，峰值倾角 1.80°，0.5 s 回到 0.042°，5 s 到 0.0004°。

## 3. 2D 雷达扫描几何演示

```bash
python demos/demo_lidar_scan.py
```

- 内容：跑一段策略仿真，每 0.5 s 取一次位姿，按 0.5° 分辨率、8 m 量程对房间墙壁和两个障碍盒做射线求交；
- 输出：`media/demo_lidar_scan.png`；
- **边界**：这是几何与可视化演示，**不是 SLAM**——没有建图、没有回环、没有位姿估计误差评估。

## 测试

```bash
python -m pytest -q tests
```

覆盖：MuJoCo 模型可加载、策略可推理、LQR 增益与厂商一致（<1e-3）、实机缓存指标可复算（≈10.2 Hz）。
CI 配置见 `.github/workflows/ci.yml`。
