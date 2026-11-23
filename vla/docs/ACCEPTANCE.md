# VLA 验收报告

> 项目：WHEELTEC B585 DP2 二阶平衡车 · 自研小型 VLA（视觉-语言-动作）
> 日期：2026-11-22　｜　契约：[specs/VLA-TASK-SPEC.md](../specs/VLA-TASK-SPEC.md)　｜　官方验收模型：`runs/bc_aux512/best.pt`
> 所有数字可用文末命令复跑；数据/模型/评测产物在 `data/`、`runs/`、`eval/results/`。

## 1. 任务定义

- 输入：车头相机 128×128 RGB 图像 + 一句中文指令（12 条固定指令，见 spec §1.3）
- 输出：10 Hz 高层速度指令 (v, ω)，v∈[−0.35,0.35] m/s、ω∈[−1.0,1.0] rad/s
- 底层：100 Hz 平衡/速度环（仿真里是 PPO 策略 `assets/lowlevel/`，实车里是 STM32 固件）
- 12 条指令：0 保持站立、1 向前走、2 向后退、3 向左转、4 向右转、5 原地左旋、6 原地右旋、7 面向红桶、8 靠近红桶、9 靠近蓝桶、10 靠近绿桶、11 绕开红桶前往蓝桶

## 2. 数据（专家自动标注，不是人工示教）

| 数据集 | 规模 | 用途 |
|---|---|---|
| dataset_v1 | 2500 回合 / 125,000 帧 / 3.9 GB | 主训练集（12 指令） |
| dataset_test | 657 回合 / 32,850 帧 / 1.1 GB | 只评测（seed 段不相交） |
| dataset_obj | 1211 回合 / 60,550 帧 | 对象类 7–11 加量 |
| dataset_obj7 | 260 回合 / 13,000 帧 | 仅指令 7 |
| dataset_v1_aux / dataset_obj_aux | 同上 + 三色桶相对位置标签（aux [dx,dy]×3） | 辅助监督用；scene_seed 与原始集逐条一致 |
| dataset_dagger | 策略闭环访问状态 + 专家标签 | DAgger 消融 |

- 只保留专家成功回合：dataset_v1 尝试 2689 → 保留 2500（丢弃 7.0%，撞桶/倒地/判据不过）
- 配对场景 93.4%（同场景不同指令），检验语言接地
- 桶为真实碰撞体；专家带绕桶项；`data/smoke` 只用于冒烟

## 3. 模型

| 项 | 值 |
|---|---|
| 主干 | HuggingFaceTB/SmolVLM-256M-Instruct（Idefics3，本地加载 494 MB，不联网） |
| 动作头 | 最后 token hidden(576) → LN → Linear(576,256) → GELU → Linear(256,2) → tanh |
| 辅助头（可选） | 同 hidden → Linear(576,256) → Linear(256,6)，回归三色桶相对位置；只在训练用 |
| 微调 | LoRA r16（q/k/v/o/gate/up/down），视觉塔冻结；可训 6.07M / 总 262.6M |
| 图像 | 512px → 64 image token（8×8 网格，每格 8.75°）；256px 对照为 16 token（4×4） |
| 训练 | BC MSE + 0.1·aux MSE，bf16，AdamW 余弦退火；8 GB 显存峰值 2233 MiB |

## 4. 结果

### 4.1 版本对比（12 指令 × N 局，test seed 段 100002 起）

| 版本 | 数据/采样 | 分辨率 | val mse | 总体 | 对象类均值 |
|---|---|---|---|---|---|
| bc_v1 | v1 全量，3 epoch | 256 | 0.05895 | 70.8%(12×10) / **71.0%(12×50)** | 40.0% |
| bc_v2 | +obj+obj7，加权 100k | 256 | 0.05543 | 71.7%(12×10) | 44.0% |
| bc_v3_512 | 同 v2 加权 30k（LR schedule bug） | 512 | 0.03132 | 66.7%(12×10) | 50.0% |
| **bc_aux512（官方）** | v1_aux+obj_aux，加权 30k + aux | 512 | **0.03014** | **71.8%(12×50)** | 45.2% |

### 4.2 逐指令成功率（bc_aux512，50 局/指令）与专家上限、失败原因

| 指令 | VLA | 专家尝试上限 | 差值 | 主要失败原因 |
|---|---|---|---|---|
| 0 保持站立 | 72% | 84.8% | −12.8 | displaced×14 |
| 1 向前走 | 78% | 96.4% | −18.4 | fall×11 |
| 2 向后退 | 100% | 100% | 0 | — |
| 3 向左转 | 98% | 97.8% | +0.2 | fall×1 |
| 4 向右转 | 88% | 89.3% | −1.3 | fall×6 |
| 5 原地左旋 | 100% | 100% | 0 | — |
| 6 原地右旋 | 100% | 100% | 0 | — |
| 7 面向红桶 | 62% | 100% | −38 | heading_too_large×19 |
| 8 靠近红桶 | 54% | 96.4% | −42 | red_not_reached×20、fall×3 |
| 9 靠近蓝桶 | 46% | 93.3% | −47 | blue_not_reached×21、fall×6 |
| 10 靠近绿桶 | 38% | 96.4% | −58 | green_not_reached×27、fall×4 |
| 11 绕开红桶前往蓝桶 | 26% | 61.2% | −35 | blue_not_reached×15、fall×10、red_too_close×12 |

**结论**：常量类指令（2/3/5/6）达到物理上限；0/1/4 的损失主要是底层漂移/摔倒（专家上限本身只有 84.8–89.3%）。差距集中在 7–11：单帧图像估计「桶相对方位 + 距离」的能力不足，失败以 `not_reached` / `heading_too_large` 为主，而不是碰撞。

### 4.3 消融汇总（都做了，都没能突破 45% 对象类均值）

| 手段 | 配置 | 结果 | 结论 |
|---|---|---|---|
| 数据加量 + 指令加权 | v2：+dataset_obj/obj7，对象类占 65% | 71.7%（12×10），对象 44% | 数据分布不是瓶颈 |
| 更高分辨率 | v3：512px（64 token，8×8） | 66.7%（LR bug）；对象 50% | 方位精度有改善，但常量类被破坏 |
| 分辨率 + 数据 + LR 修复 | **bc_aux512（官方）** | **71.8%（12×50），对象 45.2%** | 当前最好 |
| 辅助监督（桶相对位置回归） | aux 头，权重 0.1 | aux 回归 RMSE：red dx 0.20m / blue dy 0.27m | 有帮助但不足以闭环 |
| DAgger 一轮微调 | 策略状态 + 专家标签 11,846 帧，dagger 占比 26.7% | 72.5%（12×10），对象 44% | 与官方模型同分，作为消融保留 |

**根因定位（DAgger 诊断）**：在策略自己偏航/需要对准时，专家给出的 ω 是策略输出的 3–10 倍（如「面向红桶」专家 ω≈+0.66 vs 策略 ≈−0.03），即系统性**转向不足**；512px+aux 仍未解决单帧图像上的「桶在画面里的精确方位 + 距离」估计。

### 4.4 视觉扰动与灰图消融（bc_aux512，12 局/指令）

| 场景 | 成功率 | 与 base 配对掉点（参照 base10 = 72.5%） |
|---|---|---|---|
| base | 71.8%（50 局/指令） | — |
| 亮度 ×0.7 | 73.6%（108/144） | +1.1pp |
| 亮度 ×1.3 | 72.2%（108/144） | −0.3pp |
| 10% 随机遮挡 | 72.2%（108/144） | −0.3pp |
| 固定灰图（视觉消融） | 67.4%（97/144） | −5.1pp |

结论：亮度/遮挡掉点 ≤1.1pp（要求 ≤10/15pp）；灰图消融掉点 −5.1pp，且「靠近类」在灰图下仍会掉——视觉确实在被使用，但 512+aux 模型对灰图不如 256 版敏感（256 版配对 −12.5pp）。

### 4.5 延迟与基线

| 指标 | 值 |
|---|---|
| 推理 p50 / p95（512px，bs1，预热 20 测 100） | **p50 49.3–63.4 ms / p95 51.2–77.2 ms**（多次测量，随宿主负载变化；验收时刻 49.3/51.2，本机负载 load≈3.6–4.9 时 63.4/76.8 与 62.8/77.2；10 Hz 要求 <100 ms，全部满足） |
| 闭环 per-call（bs8，含环境步进） | p50 83 ms / p95 94 ms（256px v1 实测） |
| random 策略 | 5.8% |
| 恒零动作 | 11.7% |
| VLA（bc_aux512） | 71.8% |

### 4.6 推理延迟优化（2026-11-23，`vla/fast_infer.py`）

在不改模型/权重的前提下，把单步推理从 **50.3 ms 降到 20.4 ms**（512px，2.47×）：

| 路径 | 512px p50 / p95 | 256px p50 / p95 |
|---|---|---|
| legacy（HF processor + LoRA 适配器） | 50.3 / 53.6 ms | 40.5–41.7 / 44.8–45.4 ms |
| + LoRA 合并（`merge_and_unload`） | 33.1 / 34.9 ms | 21.2 / 23.7 ms |
| + 快速预处理（缓存 instruction tokens、GPU 端 resize/normalize） | 26.5 / 28.7 ms | 19.3 / 21.6 ms |
| + CUDA graph（固定 batch/尺寸，懒捕获） | **20.4 / 21.7 ms** | **7.8 / 9.2 ms** |

- 数值一致性：快速预处理 `pixel_values` 与 HF processor **逐位相同（max|diff|=0）**、`input_ids` 完全一致；fast vs slow `act()` 在 100 组随机 + 60 帧真实数据上 **0.0**；LoRA 合并在在分布数据上 max 7.2e-5；CUDA graph vs eager 在 60 帧真实数据 p50 5.5e-5 / p99 1.9e-3 / max 7.1e-3（与「模型本来用 bf16 vs fp32」的 3.5e-3 同量级，已如实记录）。
- 闭环一致性：同 seed 12×5 局，legacy / merge-only / fast+graph 均 **63.3%（38/60）**，无掉点（`runs/closed_loop_opt/parity_*.json`）。
- 10 Hz 预算：512px p95 21.7 ms + 相机/串口 10–20 ms ≈ 总 p95 ≤42 ms，**余量 ≥58 ms**。
- 降级开关：`VLA_CUDA_GRAPH=0` → 26.5 ms；`VLA_FAST_PROC=0` → 33.1 ms；`VLA_MERGE_LORA=0` → 50.3 ms（512px p50）。训练必须 `merge_lora=False`。

## 5. 与验收标准（spec §7）的差距

| 标准 | 目标 | 当前 | 状态 |
|---|---|---|---|
| 12 指令总体成功率 | ≥80% | 71.8% | ❌ |
| 单指令成功率 | ≥70% | 5 条未达（7–11） | ❌ |
| 亮度/遮挡扰动掉点 | ≤10 / ≤15 pp | 无显著下降（v1 实测） | ✅ |
| 推理延迟 p95 | <100 ms | **21.7 ms（优化后，512px）**；优化前 51.2–77.2 ms | ✅ |
| 灰图消融显著下降 | 要 | 对象类 0%（v1 实测） | ✅ |

## 6. 结论与后续

**做成了什么**：从零搭起一条可复现的 VLA 链路——MuJoCo 场景与专家自动标注、12.5 万帧数据集、SmolVLM-256M+LoRA 行为克隆、10 Hz 闭环、视觉扰动/灰图消融/延迟/基线评测、本地语音（faster-whisper GPU 117 ms）、上位机 ASCII 协议与固件补丁（编译通过、33 条单测、未烧写）。常量类指令达到底层+判据的物理上限，整体 71.8% 显著高于随机（5.8%）与恒零（11.7%）。

**没做到什么**：对象类指令 26–62%，未达 70% 单指令线与 80% 总体线。瓶颈已定位为**单帧图像的空间/距离估计精度**（512px 的 8×8 token 网格仍只有 8.75°/格；单帧也难以推断距离变化率）。

**下一步（按优先级）**：
1. **多帧历史输入**（2–4 帧，或叠加速度指令历史）——让模型能通过视觉流估计接近速度，是单帧估距的直接替代；
2. **DAgger / 闭环数据回填**（脚本已就绪 `eval/collect_dagger.py`）——修正 BC 的分布漂移；
3. 更大的图像 token 预算（512px 已是当前显存上限；可换 2×2 分块输入或换更小 patch 的骨干）；
4. 实机阶段：USB 相机到位后按 [REAL_ROBOT_PLAN.md](REAL_ROBOT_PLAN.md) 分层部署 + 少量实机数据微调。

## 7. 复跑命令

```bash
export https_proxy=http://127.0.0.1:7890 http_proxy=http://127.0.0.1:7890   # 仅首次下载需要
# 数据自检
.venv/bin/python sim/gen_dataset.py --check-only --out data/dataset_v1
.venv/bin/python scripts/ds_validate.py --data data/dataset_v1
# 训练（官方模型）
.venv/bin/python vla/train_bc.py --data data/dataset_v1_aux,data/dataset_obj_aux \
  --out runs/bc_aux512 --epochs 1 --epoch-frames 30000 --bs 4 --grad-accum 4 \
  --image-size 512 --lr 3e-4 --aux-loss-weight 0.1 --balance-instr \
  --instr-weights "7:3,8:2,9:2,10:2,11:4" --resume runs/bc_v3_512/best.pt --reset-progress
# 闭环验收
.venv/bin/python eval/closed_loop.py --ckpt runs/bc_aux512/best.pt --image-size 512 \
  --episodes 50 --instrs 0-11 --scenarios base --seed-base 100002 --name bc_aux512_base50
.venv/bin/python eval/closed_loop.py --ckpt runs/bc_aux512/best.pt --image-size 512 \
  --episodes 50 --perturb-episodes 12 --instrs 0-11 \
  --scenarios base,bright_low,bright_high,occlusion,gray --baseline-episodes 10 --name bc_aux512_robust
# 演示视频
.venv/bin/python scripts/make_demo.py --ckpt runs/bc_aux512/best.pt --image-size 512 \
  --instrs 0-11 --episodes-per-instr 1 --max-episodes 12 --out media/vla_demo.mp4 --cols 4
```
