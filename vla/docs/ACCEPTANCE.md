# VLA 验收报告

> 项目：WHEELTEC B585 DP2 二阶平衡车 · 自研小型 VLA（视觉-语言-动作）
> 日期：2026-11-23　｜　契约：[specs/VLA-TASK-SPEC.md](../specs/VLA-TASK-SPEC.md)
> **官方验收模型：`runs/bc_v6b_lowlr/best.pt`**（256 px、2 帧历史；SmolVLM-256M + LoRA + 动作头 + aux 头）
> 所有数字可用文末命令复跑；数据/模型/评测产物在 `data/`、`runs/`、`eval/results/`。

## 1. 任务与架构

- 输入：车头相机 128×128 RGB（连续两帧 t−1、t）+ 一句中文指令（12 条）
- 输出：10 Hz 高层速度指令 (v, ω)，v∈[−0.35,0.35] m/s、ω∈[−1.0,1.0] rad/s
- 底层：100 Hz 平衡/速度环（仿真里是 PPO 策略，实车里是 STM32 固件）
- 指令：0 保持站立 · 1 向前走 · 2 向后退 · 3 向左转 · 4 向右转 · 5 原地左旋 · 6 原地右旋 · 7 面向红桶 · 8 靠近红桶 · 9 靠近蓝桶 · 10 靠近绿桶 · 11 绕开红桶前往蓝桶

## 2. 数据（专家自动标注，非人工示教）

| 数据集 | 规模 | 用途 |
|---|---|---|
| dataset_v1 / v1_aux | 2500 回合 / 125,000 帧 | 主训练集（aux 版带三色桶相对位置标签） |
| dataset_obj / obj_aux | 1211 回合 / 60,550 帧 | 对象类 7–11 加量 |
| dataset_obj7 | 260 回合 / 13,000 帧 | 仅指令 7 |
| dataset_test | 657 回合 / 32,850 帧 | **只评测**（seed 段与训练不相交） |
| dataset_dagger | 240 回合 / 11,846 帧 | DAgger 消融 |
| dataset_smoke | 28 回合 / 1,400 帧 | 冒烟 |

- 只保留专家成功回合（data_v1 尝试 2689 → 保留 2500，丢弃 7.0%）；配对场景 93.4%
- 桶是真实碰撞体；专家带绕桶项

## 3. 模型

| 项 | 值 |
|---|---|
| 主干 | SmolVLM-256M-Instruct（Idefics3，本地 494 MB，不联网） |
| 输入 | **2 帧历史**（t−1, t），256 px → 16 image token/帧（共 32） |
| 动作头 | 最后 token hidden(576) → LN → Linear(576,256) → GELU → Linear(256,2) → tanh |
| 辅助头 | 同 hidden → Linear(576,256) → Linear(256,6)，回归三色桶相对位置（只在训练用） |
| 微调 | LoRA r16，视觉塔冻结；可训 6.07M / 总 262.6M |
| 训练 | BC MSE + 0.1·aux MSE，bf16，AdamW 余弦；峰值 2.4 GB 显存 |

## 4. 结果

### 4.1 全部版本对比（12 指令 × 50 局 = 600 局，test seed 段 100002）

| 版本 | 训练 | 输入 | base50 | base10 | 对象类均值 | 推理 p50 |
|---|---|---|---|---|---|---|
| v1 | v1 全量，3 epoch | 256×1 | **71.0%** | 73.3% | 40.0% | 40.5（legacy） |
| v2 | +obj/obj7，指令加权 | 256×1 | — | 71.7% | 44.0% | — |
| v3 | 加权（LR schedule bug） | 512×1 | — | 66.7% | 50.0% | — |
| aux512 | +aux 辅助监督 | 512×1 | 71.8% | 72.5% | 45.2% | 20.4（fast） |
| v5 | lr 3e-4 续训 1 epoch | 512×1 | — | 62.5%（回归） | 32.0% | — |
| v5b | lr 5e-5 续训 1 epoch | 512×1 | — | 71.7%（中性） | 44.0% | — |
| v6 | **2 帧历史** | 256×2 | 72.5% | 74.2% | 43.2% | 11.3（fast） |
| **v6b（官方）** | v6 + lr 5e-5 续训 1 epoch | 256×2 | **73.7%（442/600）** | 75.0%（fast）/ 75.8%（slow） | **46.0%** | **11.3** |

**结论**：2 帧历史 +0.7~2.5pp；低学习率续训再 +1.2pp；**高学习率续训（v5）会破坏常量类校准**（站立 0%、面向红桶 10%），已回退并作为「续训不能直接放大 lr」的证据。

### 4.1b 复现性回归（2026-11-24）

用**当前代码**（fast 路径：LoRA 合并 + 快速预处理 + CUDA graph）按官方命令复跑 v6b base50：

| 口径 | 结果 | 逐指令 |
|---|---|---|
| eager（原始，§4.2 数字） | 73.7%（442/600） | 88/80/100/98/88/100/100/66/48/44/42/30 |
| fast（当前部署路径） | **73.2%（439/600）** | 88/74/100/98/88/100/100/66/44/44/46/30 |

差 3 局（0.5pp），来自 fast 路径的数值差（graph vs eager max|diff|≈1.8e-3，仅影响边界回合）——**属可接受的复现噪声**，对外口径写 **73.2–73.7%**（fast 为实机部署路径）。

### 4.2 官方模型逐指令（base50）与专家上限

| 指令 | VLA | 专家尝试上限 | 主要失败原因 |
|---|---|---|---|
| 0 保持站立 | 88% | 84.8% | displaced×6 |
| 1 向前走 | 80% | 96.4% | fall×10 |
| 2 向后退 | 100% | 100% | — |
| 3 向左转 | 98% | 97.8% | fall×1 |
| 4 向右转 | 88% | 89.3% | fall×6 |
| 5 原地左旋 | 100% | 100% | — |
| 6 原地右旋 | 100% | 100% | — |
| 7 面向红桶 | 66% | 100% | heading_too_large×17 |
| 8 靠近红桶 | 48% | 96.4% | red_not_reached×24、fall×2 |
| 9 靠近蓝桶 | 44% | 93.3% | blue_not_reached×24、fall×4 |
| 10 靠近绿桶 | 42% | 96.4% | green_not_reached×23、fall×6 |
| 11 绕开红桶前往蓝桶 | 30% | 61.2% | blue_not_reached×17、fall×9、red_too_close×9 |

常量类 2/3/5/6 已到底层+判据的物理上限；0/1/4 的损失主要来自底层漂移与摔倒（专家上限本身 84.8–96.4%）。差距集中在 7–11，失败以 `not_reached` / `heading_too_large` 为主，**不是碰撞**。

### 4.3 鲁棒性与灰图消融（v6b，12 局/指令，同 seed 配对）

| 场景 | 成功率 | 相对 base |
|---|---|---|
| base | 76.4%（110/144） | — |
| 亮度 ×0.7 | 76.4% | 0.0 pp |
| 亮度 ×1.3 | 76.4% | 0.0 pp |
| 10% 随机遮挡 | 77.8% | +1.4 pp |
| 固定灰图（两帧同时置灰的视觉消融） | 72.2% | **−4.2 pp** |

对照 aux512（单帧 512px）：亮度 73.6/72.2%、遮挡 72.2%、灰图 67.4%（−5.1pp）。
> 注：v6 早期 robust 出现「灰图 = base」，是评测脚本只给当前帧加扰动、历史帧仍为真实图的 bug；已修复为两帧同时扰动并重测，上表为修复后结果。

### 4.4 推理延迟优化（`vla/fast_infer.py`）

| 路径 | 单帧 512px p50/p95 | 双帧 256px p50/p95 |
|---|---|---|
| legacy（HF processor + 未合并 LoRA） | 50.3 / 53.6 ms | 44.1 / 47.2 ms |
| + LoRA 合并（`merge_and_unload`） | 33.1 / 34.9 ms | 23.7 / 24.7 ms |
| + 快速预处理（缓存指令 token、GPU 端 resize/normalize） | 26.5 / 28.7 ms | 20.7 / 22.6 ms |
| + CUDA graph（固定尺寸懒捕获） | **20.4 / 21.7 ms** | **11.3 / 12.4 ms** |

- 快速预处理与 HF processor 的 `pixel_values` **逐位相同（max|diff|=0）**、`input_ids` 完全一致；fast(eager) vs slow `act()` = **0.0**；CUDA graph 在真实帧 max 6.8e-4（与 bf16 推理噪声同量级），可用 `VLA_CUDA_GRAPH=0` 回退。
- 闭环一致性：同 seed 12×5/12×10，fast 70.0% vs legacy 70.8%（1/120 局差异，无系统性掉点）。
- 10 Hz 预算：双帧 256px p95 12.4 ms + 实机 I/O 10–20 ms ≈ 总 p95 ≤32 ms，**余量 ≥68 ms**。
- 降级：`VLA_CUDA_GRAPH=0` → 20.7 ms；`VLA_FAST_PROC=0` → 23.7 ms；`VLA_MERGE_LORA=0` → 44.1 ms（双帧 256px p50）。训练必须 `merge_lora=False`。

### 4.5 语言消融（2026-11-24 新增）

把**所有回合的指令文本强制换成「向前走」**（任务与判定仍用真实 instr_id，只改喂给模型的 L），官方模型 base10：

| 条件 | 总体 | 说明 |
|---|---|---|
| 真实指令 | **75.0%（90/120）** | 基准 |
| 一律喂「向前走」 | **21.7%（26/120）** | −53.3 pp；除"向前走"外，站立/后退/转向/自旋全部 0% |

→ **语言通道确实在被使用**（不是只看图就开车的"伪 VLA"）。与灰图消融（视觉 −4.2pp）合起来，V 和 L 两条通道都有反证支持的消融。

### 4.6 基线与其他消融

| 策略/消融 | 结果 |
|---|---|
| random 动作 | 5.8% |
| 恒零动作 | 11.7% |
| 数据加量 + 指令加权（v2） | 71.7%（12×10），对象 44.0% |
| aux 辅助监督（aux512） | 71.8%（12×50），对象 45.2% |
| DAgger 一轮 | 72.5%（12×10），对象 44.0%（与官方同分） |
| EMA 动作低通（aux512） | 70.0% < 73.3%，无收益，默认关闭 |
| 2 帧历史（v6） | 72.5%（12×50），对象 43.2% |
| **2 帧 + 低 lr 续训（v6b）** | **73.7%（442/600，12×50），对象 46.0%** |
| 4 帧稀疏历史（v6c，步距 2） | 71.7%（12×10），对象 44.0%（instr10 掉到 10%）——**无增益，已停** |
| v6b 再续训 2 epoch（v6d，lr 2e-5） | 72.5%（12×10），对象 44.0%——**无增益** |
| 两阶段 aux 条件化动作头（v7） | 71.7%（12×10，慢路径真实启用 aux 条件化），对象 42.0%——**无增益**（v6b 75.0%/50.0%） |
| 距离提示（v8：oracle 目标距离拼进指令） | 70.8%（12×10），对象 42.0%——**无增益**（说明瓶颈在转向/方位，不是距离估计） |
| 方位提示（v9：oracle 目标方位角拼进指令） | 73.3%（12×10），对象 46.0%——**无增益**（说明数值提示通道无法弥补视觉分辨率/细粒度条件） |
| 512px × 2 帧（v10：8×8 网格 + 时序） | 71.7%（12×10），对象 44.0%——**无增益**（分辨率+时序的组合也没突破） |

## 5. 与验收标准（spec §7）的差距

| 标准 | 目标 | 当前（v6b） | 状态 |
|---|---|---|---|
| 12 指令总体成功率 | ≥80% | **73.7%（442/600）** | ❌ |
| 单指令成功率 | ≥70% | 8 条达标；8/9/10/11 = 48/44/42/30% 未达 | ❌ |
| 亮度扰动掉点 | ≤10 pp | 0.0 pp | ✅ |
| 遮挡扰动掉点 | ≤15 pp | −1.4 pp（更好） | ✅ |
| 推理延迟 p95 | <100 ms | **12.4 ms**（双帧 256px，优化后） | ✅ |
| 灰图消融显著下降 | 要 | −4.2 pp | ✅ |

## 6. 结论与后续

**做成了什么**：可复现的完整 VLA 链路——MuJoCo 场景与专家自动标注（12.5 万帧）、SmolVLM-256M + LoRA + 动作头（+ aux 头）、2 帧历史输入、10 Hz 闭环评测（扰动/灰图/延迟/基线/逐回合失败原因）、本地语音（faster-whisper GPU 117 ms）、上位机 ASCII 协议、固件补丁（编译通过 + 33 条单测，未烧写）。**推理延迟从 50.3 ms 优化到 11.3 ms（双帧，3.9×）**，10 Hz 余量 ≥68 ms。常量类指令达到底层+判据上限。

**没做到什么**：总体 73.7% 未到 80%（11 个变体：数据加量/加权、512px、aux 监督、DAgger、4 帧稀疏历史、延长续训、aux 条件化两阶段，均未突破），差距全部在「靠近某色桶」（42–48%）和「面向红桶」（66%）。八版迭代（数据加量/加权、512px、aux 监督、DAgger、2 帧历史、低 lr 续训）累计只带来约 +4pp，说明**瓶颈是单目视觉的距离/方位估计精度**，不是训练时长或数据量。

**下一步建议**：8 个变体（数据/加权、512px、aux、DAgger、2/4 帧、延长续训、aux 条件化、oracle 距离提示）全部在 70.8–75.0% 区间，其中 v8（oracle 距离）与 v9（oracle 方位角）都无增益 → **数值提示通道救不了**；512px×2 帧（8×8 网格+时序）也无增益 → 瓶颈是**单目视觉本身的可观测性/策略学习方式**，不是分辨率、时序、数据量或动作头结构。可行方向：① 更高空间分辨率或更强视觉主干（512px 双帧 / 换骨干）；② 用激光的**方位**（不只是距离）作为输入；③ 实机数据微调（域适应）。实机方案见 [REAL_ROBOT_PLAN.md](REAL_ROBOT_PLAN.md)。

## 7. 复跑命令

```bash
# 数据自检
.venv/bin/python sim/gen_dataset.py --check-only --out data/dataset_v1
.venv/bin/python scripts/ds_validate.py --data data/dataset_v1

# 官方模型训练（v6 → v6b 低 lr 续训，均 2 帧 256px）
.venv/bin/python vla/train_bc.py --data data/dataset_v1_aux,data/dataset_obj_aux \
  --out runs/bc_v6_hist2 --epochs 1 --epoch-frames 30000 --bs 8 --image-size 256 --frames 2 \
  --lr 3e-4 --aux-loss-weight 0.1 --balance-instr --instr-weights "7:3,8:2,9:2,10:2,11:4" \
  --resume runs/bc_v2/best.pt --reset-progress
.venv/bin/python vla/train_bc.py --data data/dataset_v1_aux,data/dataset_obj_aux \
  --out runs/bc_v6b_lowlr --epochs 1 --epoch-frames 30000 --bs 8 --image-size 256 --frames 2 \
  --lr 5e-5 --aux-loss-weight 0.1 --balance-instr --instr-weights "7:3,8:2,9:2,10:2,11:4" \
  --resume runs/bc_v6_hist2/best.pt --reset-progress

# 官方闭环验收（fast 路径自动启用：merge LoRA + 快速预处理 + CUDA graph）
.venv/bin/python eval/closed_loop.py --ckpt runs/bc_v6b_lowlr/best.pt --image-size 256 --frames 2 \
  --episodes 50 --instrs 0-11 --scenarios base --skip-baselines --skip-latency \
  --batch-episodes 8 --seed-base 100002 --name bc_v6b_base50
.venv/bin/python eval/closed_loop.py --ckpt runs/bc_v6b_lowlr/best.pt --image-size 256 --frames 2 \
  --episodes 12 --perturb-episodes 12 --instrs 0-11 \
  --scenarios base,bright_low,bright_high,occlusion,gray --name bc_v6b_robust_official

# 推理延迟与数值一致性
.venv/bin/python scripts/bench_infer.py --mode bench --tag hist2 \
  --ckpts runs/bc_v6b_lowlr/best.pt:256 --frames 2
.venv/bin/python scripts/bench_infer.py --mode parity \
  --ckpts runs/bc_v6b_lowlr/best.pt:256 --frames 2

# 演示视频（12 指令网格）
.venv/bin/python scripts/make_demo.py --ckpt runs/bc_v6b_lowlr/best.pt --frames 2 \
  --instrs 0-11 --episodes-per-instr 1 --max-episodes 12 --out media/vla_demo.mp4 --cols 4
```
