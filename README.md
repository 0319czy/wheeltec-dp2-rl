# WHEELTEC B585 DP2 · 两轮倒立摆：MuJoCo 强化学习 + 固件部署 + LQR 建模

两轮倒立摆底盘（车体 + 双驱动轮，可拆摆杆）的控制与学习实验仓库。
内容分三块：**仿真训练（MuJoCo + PPO）**、**板载固件（STM32 HAL + 自研轻量 RTOS + 策略推理）**、**LQR 建模与对照**。

> **状态先说清楚**：仿真侧已完成多轮独立评估；实机侧**尚未通过闭环验收**。
> 实机已测得的故障数据（约 10 Hz 振荡、约 200 mm 单向漂移）与复现材料都在本仓库，
> 失败结论与后续候选同样如实记录（见 [实机记录](docs/REAL_ROBOT.md)）。

---

## 一、这份仓库能证明什么 / 不能证明什么

| 能力项 | 仓库中的证据 | 状态 |
| --- | --- | --- |
| MuJoCo 建模（一阶无杆 / 二阶带杆，含 N10 雷达载荷与装配偏差） | [`mujoco/`](mujoco/) · [`results/`](results/) | ✅ 可复现 |
| PPO 训练 + 域随机化 + 课程 | [`mujoco/train_direct_pwm_robust.py`](mujoco/train_direct_pwm_robust.py) · [`docs/TRAINING_LOG.md`](docs/TRAINING_LOG.md) | ✅ 日志齐全 |
| 策略直接输出 PWM、带台架限幅与保护 | [`firmware/src/rl_first_order.c`](firmware/src/rl_first_order.c) | ✅ 源码 + 编译产物说明 |
| 自研轻量 RTOS（TIM7 节拍 / PendSV 切换 / 任务与信号量） | [`firmware/src/rtos_kernel.c`](firmware/src/rtos_kernel.c) · [`firmware/src/rtos_tasks.c`](firmware/src/rtos_tasks.c) | ✅ 编译通过；实机待验 |
| C 推理与 Python 策略一致性校验 | [`firmware/src/rl_mj_infer.c`](firmware/src/rl_mj_infer.c) · [固件说明](firmware/README.md) | ✅ 14496 组对比，最大误差 3.35 PWM |
| LQR：8 状态离散化 + Riccati 求增益，并与厂商 MATLAB 结果比对 | [`lqr/`](lqr/) | ✅ 脚本 + 图 + 误差 |
| 实机数据采集与故障定位（100 Hz 控制缓存、频谱分析） | [`docs/REAL_ROBOT.md`](docs/REAL_ROBOT.md) · [`docs/real-trace-100hz.csv`](docs/real-trace-100hz.csv) | ✅ 原始数据在仓库 |
| 实车闭环站立验收 | — | ❌ **未通过**（见下） |
| PCB / CAN 电机 / Linux 字符设备驱动 / SLAM / YOLO / VLA | 不在本仓库 | ➖ 属其他项目：[0319czy/robomaster-cboard-omni](https://github.com/0319czy/robomaster-cboard-omni) |

---

## 二、仿真结果（MuJoCo，平面模型，100 Hz 控制）

策略直接输出左右轮 PWM，评估带台架限幅（±2500 PWM）、位移保护（0.20 m）、
俯仰保护（0.09 rad）与轮速保护（16 rad/s）。

| 策略 | 观测 → 动作 | 独立复评 |
| --- | --- | --- |
| [一阶无杆](models/first-order/policy.zip) | 6 → 2 | 60 s 强扰动 20/20 局完成；PWM 跳变 RMS 97.2 → 84.8 |
| [二阶带杆](models/second-order/policy.zip) | 8 → 2 | 60 s 无推扰 20/20；30 s 轻推扰 32/32；20 s 较难工况 9/24 |

![评估图](media/evaluation.png)

全部原始 JSON 在 [`results/`](results/)，指标定义与选型经过见 [训练日志](docs/TRAINING_LOG.md)，
调试过程见 [调试日志](docs/DEBUG_LOG.md)。

---

## 三、实机侧：链路与实测结论

**部署链路**（固件在 [`firmware/src/`](firmware/src/)）：

```
IMU(DMP) 俯仰角/角加速度 ─┐
编码器 → 轮速、里程      ├─→ 6 维观测 → C 推理(MLP 6-64-64-2) → PWM(±2500) → 电机
上位机命令/#参数         ─┘                 ↑ 保护：45° / 300 ms 心跳 / 0.20 m / 16 rad/s
```

**实测（2026-11-22，100 Hz 缓存 128 帧，数据在仓库）**：

| 指标 | 实测值 |
| --- | --- |
| 控制周期 | 中位 9.98 ms（≈100 Hz，抖动小） |
| 倾角 RMS | 0.90° |
| 倾角主频 | **10.19 Hz**（持续振荡） |
| PWM RMS / 相邻步跳变 RMS | 1906 / 1883 · 1112 |
| 单向行程漂移 | 约 198 mm，触发范围保护停机 |

**结论**：该策略**未通过实机验收**。后续做了偏差课程续训与蒸馏（小角度偏差从 0/12 提升到 12/12），
但大角度回正退化，且实机再次触发轮速保护。完整过程、失败原因与回滚记录见
[实机记录](docs/REAL_ROBOT.md)。

**已定位的物理边界**：按单轮力矩上限 0.0402 N·m、车身 0.9 kg、重心高 0.09 m 计算，
稳态可支撑的最大倾角约 **2.9°**；更大的初始倾斜在物理上无法靠力矩硬撑回正。
这解释了"大角度回正"训练反复无效，也是把评估口径改到 ±2.6° 的依据。

---

## 四、快速复现

```bash
python3.10 -m venv .venv && source .venv/bin/activate
python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt

# 一阶：20 局 × 60 s 强扰动复评
python mujoco/train_direct_pwm_robust.py \
  models/first-order/policy.zip models/first-order/vecnormalize.pkl \
  --episodes 20 --seconds 60 --seed 99735 --scenario stress

# 二阶：60 s 无推扰 + 30 s 轻推扰
python mujoco/train_second_order_pwm.py --model models/second-order/policy.zip \
  --level 0 --push 0 --seconds 60 --episodes 20 --seed 172781 --pwm-limit 2500 --travel-limit 0.2
python mujoco/train_second_order_pwm.py --model models/second-order/policy.zip \
  --level 0 --push 0.08 --seconds 30 --episodes 32 --seed 172781 --pwm-limit 4500 --travel-limit 0.2
```

LQR 增益复算：`python lqr/lqr_design.py`（脚本内直接与厂商 MATLAB 结果对比）。
实机缓存复算：`python docs/analysis/real_trace_stats.py`。

详细环境与预期输出见 [复现说明](docs/REPRODUCE.md)。

---

## 五、仓库结构

| 路径 | 内容 |
| --- | --- |
| [`mujoco/`](mujoco/) | MuJoCo 模型、环境、PPO 训练、平滑蒸馏、回报搜索 |
| [`models/`](models/) | 一阶/二阶候选策略与归一化参数 |
| [`results/`](results/) | 完整评估 JSON、分支配置 |
| [`firmware/src/`](firmware/src/) | 板载控制源码：RL 控制与保护、自研 RTOS 内核与任务、C 推理 |
| [`lqr/`](lqr/) | LQR 建模脚本、四联图、调参说明 |
| [`docs/`](docs/) | 训练日志、调试日志、实机记录、复现说明、原始缓存 |
| [`media/`](media/) | 结果图 |

## 六、已知限制（写在这里，避免误读）

1. 所有通过率均为 MuJoCo 仿真，**不能外推为实车能力**；
2. 实机未通过验收：10 Hz 振荡与单向漂移的机理仍在定位，控制时延随机化训练未奏效；
3. 未建模/未验证：轮地打滑的完整摩擦模型、电机死区与温漂、IMU 温度漂移、舵机云台；
4. 电阻/电容等电气参数与 PCB 设计不在本仓库。

## 许可

MIT，见 [LICENSE](LICENSE)。联系方式：2153420184@qq.com · [github.com/0319czy](https://github.com/0319czy)
