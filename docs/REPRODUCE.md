# 复现说明

## 环境

| 组件 | 版本 |
| --- | --- |
| Python | 3.10 |
| MuJoCo | 3.17.0 |
| PyTorch | 2.9.1（CPU 版） |
| stable-baselines3 | 2.9.2 |
| Gymnasium | 1.4.0 |
| NumPy | 2.2.6 |

```bash
python3.10 -m venv .venv && source .venv/bin/activate
python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

MuJoCo 评估不需要显示器。

## 1. 一阶策略复评（20 局 × 60 s 强扰动）

```bash
python mujoco/train_direct_pwm_robust.py \
  models/first-order/policy.zip models/first-order/vecnormalize.pkl \
  --episodes 20 --seconds 60 --seed 99735 --scenario stress
```

**预期**：20/20 局完成；PWM 相邻步跳变 RMS ≈ 84.8（对照旧检查点 97.2）。
对照数据见 `results/first-order/slew/evaluation.json`。

## 2. 二阶策略复评

```bash
python mujoco/train_second_order_pwm.py --model models/second-order/policy.zip \
  --level 0 --push 0 --seconds 60 --episodes 20 --seed 172781 --pwm-limit 2500 --travel-limit 0.2

python mujoco/train_second_order_pwm.py --model models/second-order/policy.zip \
  --level 0 --push 0.08 --seconds 30 --episodes 32 --seed 172781 --pwm-limit 4500 --travel-limit 0.2
```

**预期**：第一组 20/20；第二组 32/32。

## 3. LQR 增益复算

```bash
python lqr/lqr_design.py
```

**预期**：输出 2×8 增益矩阵、闭环极点模，以及与厂商 MATLAB 结果的最大偏差。

## 4. 实机缓存复算

```bash
python docs/analysis/real_trace_stats.py
```

**预期**（与 `docs/REAL_ROBOT.md` 表格一致）：

```
样本 128，中位周期 9.98 ms
倾角 RMS 0.90°
倾角主频 10.19 Hz
PWM RMS 1906 / 1883
相邻步跳变 RMS 1112
末帧行程 198 mm
```

## 5. 固件编译（可选，需要 PlatformIO）

```bash
cd <原工程目录> && pio run -e dp2_bench_recovery45_continuous
```

本仓库只提供相关源码片段；完整 HAL/DMP/驱动工程未随仓库分发。

## 常见问题

- **MuJoCo 渲染报错**：本仓库的评估不需要渲染，勿设 `MUJOCO_GL`。
- **一阶结果差异大**：必须同时加载 `vecnormalize.pkl`，否则观测未归一化。
- **二阶 PWM 限幅**：两组复评限幅不同（2500 / 4500），混用会改变结论。
