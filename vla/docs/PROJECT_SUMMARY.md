# 项目一页纸：DP2 平衡车上的自研小型 VLA（视觉 → 语言 → 动作）

> 一句话：在 WHEELTEC B585 DP2 二阶平衡车上，自建 MuJoCo 场景与专家标注流水线，训练
> **SmolVLM-256M + 自研动作头**的小型 VLA，10 Hz 输出高层速度指令 (v, ω)，交给车上 100 Hz 平衡环执行；
> 仿真闭环达到目标要求的 8 指令口径 **90.0%**，推理 **11.3 ms/步**，实机链路（相机→VLA→串口→固件）已全部就绪、待装车验收。

## 1. 问题与分层

- 车端 STM32F103（Cortex-M3，无 NPU）跑不动视觉大模型；100 Hz 平衡环也等不起网络推理。
- 因此分层：**笔记本（RTX 5060）跑 VLA 做 10 Hz 的「看 + 听 → (v, ω)」**；限幅、平衡、看门狗、急停留在固件里。
- 接口：USB/蓝牙串口 ASCII 帧 `#V250,0;`（v 单位 mm/s、ω 单位 mrad/s），固件解析后叠加到速度环目标。

## 2. 我做了什么

| 环节 | 内容 | 证据 |
|---|---|---|
| 仿真场景 | MuJoCo 复刻车厢/相机/三色桶，桶为真实碰撞体，随机布局±30°、光照、遮挡 | `sim/scene_vla.xml`、`sim/env_vla.py` |
| 数据 | 特权状态专家 + 5 种手工策略自动标注，**2500 回合 / 125,000 帧**，只保留成功回合（尝试成功率 100%，丢弃 7.0%），配对场景 93.4% | `sim/gen_dataset.py`、`scripts/ds_validate.py` |
| 模型 | SmolVLM-256M-Instruct（Idefics3）+ LoRA r16（视觉塔冻结）+ 动作头（576→256→2，tanh）+ aux 位置头；**输入 2 帧历史 × 256px**（每帧 16 image token） | `vla/model.py`、`runs/bc_v6b_lowlr/best.pt` |
| 训练 | 行为克隆 + aux 监督（0.1）+ 指令加权采样；峰值显存 2.4 GB | `vla/train_bc.py` |
| 评测 | 闭环评测器：12 指令 × 50 局、扰动/灰图/语言消融、逐回合失败原因、延迟统计 | `eval/closed_loop.py`、`eval/results/` |
| 部署 | 本地语音（faster-whisper GPU 117 ms）→ 指令归一化 → 上位机桥（10 Hz、mock/真串口）→ 固件补丁（已编译、33 条解析单测） | `deploy/`、`deploy/firmware-patch/` |
| 工程 | 推理优化（LoRA 合并 + 缓存指令 token + 快速预处理 + CUDA graph，**3.9×**）；一键验收/出车体检脚本；18 个单测 | `vla/fast_infer.py`、`scripts/acceptance.sh`、`scripts/preflight.sh`、`tests/` |

## 3. 关键结果

| 指标 | 结果 |
|---|---|
| 闭环成功率（**目标 8 条指令** 0–7，4×50 局 ×2） | **90.0%（360/400，eager）/ 89.2%（357/400，fast）** ✅ |
| 闭环成功率（12 指令扩展，含对象类） | 73.7%（442/600，eager）/ 73.2%（439/600，fast） |
| 常量类指令（站立/前后/转/旋） | 2/3/5/6 = 100%、左转 98%、右转 88%（已达底层+判据物理上限） |
| 视觉扰动（12 局/指令，同 seed） | base 77.1%｜亮度×0.7 **0.0pp**、×1.3 −1.4pp、10% 遮挡 +0.7pp、**灰图 −4.9pp**（视觉真在用） |
| 语言消融（指令全换成「向前走」） | 75.0% → **21.7%**（−53.3pp；站立/后退/转向全 0）→ 语言真在用 |
| 相机不一致（sim→real 预演） | 视场 ±10%：−2.8pp；画面偏移 5%：−3.5pp → 换任意 UVC 广角相机可接受 |
| 推理延迟 | **p50 11.3 ms / p95 12.4 ms**（双帧 256px）；优化前 44.1 ms；10 Hz 预算余量 ≥68 ms |
| 基线对比 | 随机 5.8%、恒零 11.7%、**纯 PPO（特权状态）19.2%** |

## 4. 没做到什么（诚实）

- 12 指令口径 73.7%，未到 80%：差距全在「面向/靠近某色桶」（30–66%），需要单目估距。
- 11 个变体（数据加权、512px、aux、DAgger、2/4 帧、延长续训、aux 条件化、oracle 距离/方位、512×2、动作块）都在 60–75% 区间 → 说明瓶颈不是训练量，而是观测可观测性/方法（要动作块执行+时序集成、更强骨干、或实机数据域适应）。
- 实机尚未跑通闭环：缺 USB 相机（需 UVC 广角）与固件烧写授权；链路脚本/预编译固件/协议单测已就绪。

## 5. 复现

```bash
bash vla/scripts/acceptance.sh quick    # 4 分钟：环境+单测+数据自检+闭环+延迟 → vla/docs/ACCEPTANCE_AUTO.md
bash vla/scripts/acceptance.sh full     # 25 分钟：再加 base50 与 5 场景鲁棒性
bash vla/scripts/preflight.sh           # 出车前硬件/物料体检
.venv/bin/python vla/scripts/check_camera.py          # 相机自检 + FOV 标定
.venv/bin/python vla/scripts/serial_probe.py --serial /dev/ttyACM0   # 串口只读遥测
```

细节：`vla/docs/ACCEPTANCE.md`（验收报告）、`vla/docs/MODEL_REPORT.md`（11 个变体全记录）、`vla/docs/REAL_ROBOT_PLAN.md`（实机方案）。
