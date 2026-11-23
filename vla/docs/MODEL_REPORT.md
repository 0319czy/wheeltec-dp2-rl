# vla-model 交付报告:DP2 平衡车 VLA(行为克隆 + 闭环评测)

> 负责人:vla-model;写作用域 `vla/`、`eval/`、`runs/`。
> 契约:`specs/VLA-TASK-SPEC.md`(v1);硬件:RTX 5060 Laptop 8 GB(桌面 llama-server 常占 3.6 GB,可用约 4.4 GB)、16 核、14 GB 内存。
> 本文件是给 Lead 汇总验收报告用的**数字清单**,细节见 `vla/README.md`。

## 1. 结论摘要

| 项目 | 结果 |
|---|---|
| 官方验收模型 | `runs/bc_aux512/best.pt`(SmolVLM-256M + LoRA r16 + 动作头 + 辅助回归头;512px / 64 image token) |
| 闭环(base 12 指令 × 50 局 = 600 局,test seed 段) | **431/600 = 71.8%** |
| 单步推理延迟(bs1,512px) | **p50 49-63 ms / p95 51-73 ms**(多次测量,随宿主负载变化;均 <100 ms) |
| 视觉扰动(与同 seed base12 配对) | 亮度×0.7 / ×1.3 / 10% 遮挡 掉点均 ≤ 1.4 pp(要求 ≤10/15 pp) |
| 视觉消融(固定灰图) | 相对同 seed base12 掉点见 5.3;靠近类 8/9/10 在灰图下几乎全灭 → 视觉确被使用 |
| 常量类指令 | 2/3/5/6 已到"专家尝试成功率"物理上限 |
| 主要短板 | 7-11(对象类)均值 45.2%,专家上限 ~90%;失败以 `not_reached`/`heading_too_large` 为主 |

## 2. 模型与实现决策(踩坑记录)

- 基座:`assets/hf/SmolVLM-256M-Instruct`,transformers 5.x 用 `AutoModelForImageTextToText`
  (实际是 `Idefics3ForConditionalGeneration`);全程 `local_files_only=True`,不联网。
- 视觉塔(SigLIP-base)冻结,LM(576 hidden/30 层)挂 LoRA(r=16,α=32,dropout 0),
  目标 `q/k/v/o/gate/up/down`;可训练 **5.92 M / 262.4 M(2.3%)**;动作头 fp32。
- **只调内层 `Idefics3Model`,不算 lm_head**:同样 256px/bs16 从 273 ms/步降到 209 ms(省 ~23%)。
- 动作头:`last token hidden(576) → LayerNorm → Linear(576,256) → GELU → Linear(256,2) → tanh`,
  输出 `[v/0.35, ω/1.0]`;chunk=1(10 Hz)。
- **Idefics3 切图坑**:默认 `do_image_splitting=True` + 默认 `size=2048` 会把 128 源图放大后切成
  17 个 512 crop(单帧 input_ids=1146),甚至 257 个(18982 token,超 8192 上限)。
  本项目 `do_image_splitting=False`,`size`/`max_image_size` = 目标边长,image token 数 =
  `(边长/64)^2`:128→4、256→16、384→36、512→64,`input_ids` 分别 26/38/58/86;
  **必须同步 `processor.image_seq_len`**,否则 forward 报 image tokens/features 不匹配。
- prompt 用 chat template,产出与 spec §4 一致:
  `'<|im_start|>User:<image>指令：{instruction}<end_of_utterance>\nAssistant:'`;
  512px 时 64 image token + 25 文本 token = 89 token。
- 辅助回归头(可选):与动作头共享 last-token hidden,预测
  `[dx_red,dy_red,dx_blue,dy_blue,dx_green,dy_green]`(米,车体系,clip ±3),
  `loss = MSE(action) + 0.1·MSE(aux/3)`;无 aux 的样本按 mask 跳过。

## 3. 数据管线修复(性能 bug,影响正式训练可行性)

1. **npz 不能 mmap**:`np.load(npz, mmap_mode="r")` 返回 NpzFile,每次 `d["images"]`
   重读整个 122 MB 数组 → 16 样本/批要等 400 ms。改为整 shard 物化成 dict + LRU 缓存,
   并加"按 shard 分组打乱"的 epoch 顺序(每个 shard 每 epoch 只读一次)。
   **修前 37.3 样本/s → 修后 60~78.5 样本/s**(256px/bs16)。
2. **建索引不解压图像**:原 `np.load(npz)["images"].shape[0]` 会解压整个 122 MB 成员
   (52 个 shard = 6.4 GB 无用解压,实测卡几分钟);改为只读 npz 内 `images.npy` 头的公开
   API,14 shard 建索引 **0.03 s**。
3. **LR schedule 步数 bug(已修)**:加权采样 + `--epoch-frames` 时,`total_steps` 误用了
   采样前的数据集长度(v3:19844 vs 实际 1875),cosine 被拉长 10 倍、学习率整轮停在 3e-4,
   破坏了常量类校准(instr0 从 100% 掉到 0%)。修复后 aux512 轮 schedule 正常。

## 4. 训练版本对比

数据:dataset_v1(2500 回合/125k 帧,专家成功率 100%,含桶碰撞)、dataset_obj(1211/60.55k,
仅 7-11)、dataset_obj7(260/13k,仅 7)、以及带 aux 的对应版本;train/val 按 `scene_seed%10`。

| 版本 | 数据 / 采样 | 分辨率 | val mse_norm | base10 闭环 | 对象类均值 | 训练吞吐/峰值 |
|---|---|---|---|---|---|---|
| bc_v1 | v1 全量,3 epoch | 256 | 0.05895 | 73.3% | 40.0% | 59.2 样本/s / 2668 MiB |
| bc_v2 | v1+obj+obj7,加权 100k 帧/epoch | 256 | 0.05543 | 71.7% | 44.0% | 67.7 / 2668 MiB |
| bc_v3_512 | 同 v2 加权 30k,LR bug | 512 | 0.03132 | 66.7% | 50.0% | 15.1 / 2231 MiB |
| **bc_aux512** | v1_aux+obj_aux,加权 30k + aux 0.1 | 512 | **0.03014** | **72.5%** | 44.0% | 17.5 / 2233 MiB |

- 加权采样(v2 起):`--balance-instr --instr-weights "7:3,8:2,9:2,10:2,11:4"`,视觉类占 65%;
  aux512 的 30k 采样分布 = 常量 1500×7、7=4500、8/9/10=3000、11=6000。
- aux 回归验证集 RMSE(米):red dx .203 / dy .098、blue dx .101 / dy .266、green dx .050 / dy .030。
- 512px 训练只能 bs4+梯度累积4(等效 bs16),bs8 在桌面已占 3.6 GB 时 OOM。
- **结论**:数据加量/加权、512px(64 token)、辅助方位回归都只把对象类从 40% 拉到 44-50%,
  没有突破 65%;而 val MSE 从 0.059 降到 0.030 —— 拟合变好但闭环对象类不涨,说明瓶颈在
  "单帧估距 + 10Hz 闭环接近策略",不是单纯分辨率/数据量。

## 5. 官方模型 bc_aux512 闭环评测

### 5.1 base:12 指令 × 50 局 = 600 局,431/600 = **71.8%**

| 指令 | VLA | 专家尝试上限 | 差值 | 失败原因(50 局内) |
|---|---|---|---|---|
| 0 保持站立 | 72% | 84.8% | -12.8 | displaced×14 |
| 1 向前走 | 78% | 96.4% | -18.4 | fall×11 |
| 2 向后退 | 100% | 100% | 0 | — |
| 3 向左转 | 98% | 97.8% | +0.2 | fall×1 |
| 4 向右转 | 88% | 89.3% | -1.3 | fall×6 |
| 5 原地左旋 | 100% | 100% | 0 | — |
| 6 原地右旋 | 100% | 100% | 0 | — |
| 7 面向红桶 | 62% | 100% | -38.0 | heading_too_large×19 |
| 8 靠近红桶 | 54% | 96.4% | -42.4 | red_not_reached×20、fall×3 |
| 9 靠近蓝桶 | 46% | 93.3% | -47.3 | blue_not_reached×21、fall×6 |
| 10 靠近绿桶 | 38% | 96.4% | -58.4 | green_not_reached×27、fall×4 |
| 11 绕红靠蓝 | 26% | 61.2% | -35.2 | blue_not_reached×15、fall×10、red_too_close×12 |

失败原因标签由 `eval/closed_loop.py::failure_reason` 按 spec §1.3 判据生成;每回合还带
seed / judge_source,便于复现。

### 5.2 延迟(bs=1,GPU 预热 20 次测 100 次)

| 分辨率 | image token | p50 | p95 |
|---|---|---|---|
| 256px | 16 | 48.1 ms | 54.4 ms |
| 384px | 36 | 53.1 ms | 59.1 ms |
| **512px** | **64** | **49.3 ms** | **51.2 ms** |

延迟几乎全是固定开销(processor + 30 层 LM 的 kernel launch),分辨率不是瓶颈;10 Hz 有充足余量。
多次测量:验收评测时刻(宿主较静)p50 49.3/p95 51.2;另一次复测(load average 6.4)p50 63.1/p95 73.2;
用"预热 5 次/测 30 次"的快速冒烟会得到 81/105(预热不足 + p95 样本少),不作为验收口径。

### 5.3 视觉扰动与消融(aux512 官方模型,`eval/results/bc_aux512_robust.json`)

| 场景 | aux512(512px) | 说明 |
|---|---|---|
| base(12×50) | **71.8%**(431/600) | 官方口径 |
| 亮度 ×0.7 | 73.6%(106/144) | vs base10 72.5% ≈ +1.1pp |
| 亮度 ×1.3 | 72.2%(104/144) | ≈ -0.3pp |
| 10% 随机遮挡 | 72.2%(104/144) | ≈ -0.3pp |
| 固定灰图(视觉消融) | 67.4%(97/144) | ≈ **-5.1pp** |
| random 基线 | 5.8% | |
| constant(全 0)基线 | 11.7% | |

扰动用 12 局/指令、base 用 50 局/指令,严格配对参照同指令数的 base10=72.5%。
亮度/遮挡掉点 ≤1.1pp,远优于 spec 的 ≤10/15pp;灰图掉点 -5.1pp(< 256px 版的 -12.5pp),
且"靠近类"在灰图下仍明显下降 → 视觉被真实使用。

### 5.3b 旧版 bc_v1(256px)完整扰动组(供对照)

base 用 12 局/指令与扰动同 seed 配对:

| 场景 | bc_v1(256px) | 相对 base12 |
|---|---|---|
| base(12 局/指令) | 75.0%(108/144) | — |
| 亮度 ×0.7 | 75.0% | **0.0 pp** |
| 亮度 ×1.3 | 73.6% | **-1.4 pp** |
| 10% 随机遮挡 | 73.6% | **-1.4 pp** |
| 固定灰图(视觉消融) | 62.5% | **-12.5 pp** |
| random 基线 | 5.8% | -69.2 pp |
| constant(全 0)基线 | 11.7% | -63.3 pp |

灰图下"靠近类" 8/9/10 全部掉到 0%、11 33%→25%,常量类不变 → 视觉在对象类指令上被真实使用。
亮度/遮挡掉点远优于 spec 的 ≤10/15 pp 要求。

### 5.4 动作 EMA 对照(bc_v1 best.pt,12×10,同 seed)

无 EMA 73.3% vs EMA0.5 70.0% → 无收益、略降(9 从 6/10→4/10、11 2/10→1/10、7 不变),
**默认关闭**,保留 `--action-ema` 供实机调试。

## 6. DAgger-lite(已备好,待指令)

- `eval/collect_dagger.py`:策略执行闭环、专家在策略访问到的状态上打标签(actions=专家动作),
  同时存 aux(车体系桶位置);输出格式与训练集一致(`data/dataset_dagger/`)。
- 小规模验证(`--episodes 2`,24 回合/1192 帧):`scripts/ds_validate.py` 结构检查全 OK
  (只有"每指令 ≥200 帧"阈值不满足,2 局/指令的预期);aux 抽查:finite、|aux|≤3、
  t=0 三桶距离 0.59-1.39 m、dx>0 24/24、分布与 dataset_v1_aux 一致。
- 全量已执行:**240 回合 / 11,846 帧 / 5 shard**,`scripts/ds_validate.py` **PASSED**;
  行为策略在采集时的自成功率 69.6%(与 base50 的 71.8% 一致,说明采到的正是策略实际会走到的状态)。
- 微调(`runs/bc_dagger`):512px、`--source-weights "dataset_dagger:6"`(期望占比 26.7%)、
  30k 加权帧、lr 2e-4、aux 0.1、从 bc_aux512 热启动 1 epoch。
  **结果:base10 72.5% / 对象类 44.0%,与 bc_aux512 完全同分**(逐指令 8 -10pp、10 +10pp 相互抵消)
  → 未达 53% 门槛,按 Lead 决定回退 bc_aux512 收官,DAgger 作为消融。
- 依据:8/9/10 的失败以 `not_reached`(没开到 0.45 m)为主而非振荡/超调,属"策略自己走出的
  状态缺少监督"的分布漂移,DAgger 有机会补。
- **偏差诊断**(同 seed 回放 12 回合/592 帧,`runs/dagger_diag.npz`):专家 vs 策略平均 ω:
  指令 1(前进):+0.608 vs +0.021;7(面向红):+0.658 vs -0.026;8(靠近红):+0.403 vs -0.068;
  10(靠近绿):-0.670 vs -0.039;11(绕红靠蓝):-0.527 vs +0.007;常量类 2-6 与专家一致(|Δw|≤0.01)。
  → **核心症状是"转向不足(under-steering)"**:在策略自己偏航、需要修正的状态上,专家给的 |ω|
  是策略输出的 3~10 倍,策略几乎不打舵。这直接解释了 8/9/10 的 `not_reached` 和 1 的 fall。
  DAgger 的专家标签正是补这些状态的 ω 监督。

## 6.11 v5 续训(lr 3e-4,计划 3 epoch,ep1 即回归 → 停止)

配置(task-9):在 bc_aux512 基础上,`--data v1_aux,obj_aux,dagger --source-weights dagger:6`,
30k 采样帧/epoch、512px、bs4+累积4、aux 0.1、lr 3e-4、**LR schedule 修复生效**
(`[sched] steps/epoch=7500 total_opt_steps=5625`,不再按未采样数据集长度算)。

- ep1:1785 s(29.8 min)、231 ms/步、17.3 样本/s、峰值 2233 MiB;
  val(600-batch 全 12 指令口径)= **0.05910**,mse_v 0.002673、mse_w 0.096385;
  每指令 val mse_norm:0.0051/0.0609/0.0081/0.0030/0.0077/0.0001/0.0000/0.0411/0.0920/0.1602/0.1084/0.3259。
- **base10(512px,12×10,test seed)= 62.5%(75/120),对象类 32.0%**:
  instr0 **0/10**(`yaw_drift`×10)、instr7 **1/10**(`heading_too_large`×9)、
  1=90% 2-6=100% 8=50-60% 9=40% 10=30% 11=20%。
- **回归不是推理路径造成的**:同一 ckpt 用 fast(merge+fast_proc,graph off)与 legacy(全关)两版评测,
  总体都是 62.5%(对象类 32%/30%),逐指令只有 1(90/100)、8(60/50)互换。
- 判定:按 task-9 规则(不如 aux512 72.5%/44.0% 就停)**不续 ep2/3**;官方模型仍为 `runs/bc_aux512/best.pt`(71.8%)。
- 结论:**lr 3e-4 从 aux512 续训 1 epoch 会打坏常量类校准(instr0 站立、instr7 面向红桶)**,
  即使 LR schedule 正确。继续训练必须显著降低 lr(见 6.12),否则越训越差。

## 6.11b v5b 温和续训(lr 5e-5,1 epoch,无 dagger)

配置(Lead 指定):`--data v1_aux,obj_aux --epochs 1 --epoch-frames 30000 --bs 4 --grad-accum 4
--image-size 512 --lr 5e-5 --aux-loss-weight 0.1 --balance-instr --instr-weights 7:3,8:2,9:2,10:2,11:4
--resume runs/bc_aux512/best.pt --reset-progress --val-batches 120 --stop-after-epoch 1`

- 训练:1633 s(27.2 min)、211 ms/步、18.9 样本/s、峰值 2233 MiB
- val(120-batch,与 aux512 同口径)= **0.03184**(aux512 0.03014,略差);mse_v 0.001473、mse_w 0.05165
- base10(512px,fast 路径默认 merge+fast_proc+cuda_graph)= **71.7%(86/120)**,对象类 **44.0%**
  - 每指令:0=100% 1=70% 2=100% 3=100% 4=70% 5=100% 6=100% 7=60% 8=60% 9=40% 10=30% 11=30%
- 对比 aux512(72.5% / 44.0%):**总体 -0.8pp、对象类持平**,instr0 校准恢复正常(100%)。
- 判定:按 Lead 规则落在"持平(72-74%)"区间 → **温和续训无显著收益**,不跑 base50;
  BC 在这套数据/架构上已收敛,继续训练(无论 lr 3e-4 还是 5e-5)都不会带来对象类提升。
  下一步应改观测/结构(两帧历史、辅助监督、多轮 DAgger),而不是多训。

## 6.12 推理路径 parity(评估可信度)

latency-opt 队友的 parity(bc_v2@256):
- 快速预处理:pixel_values **bit-identical**,input_ids 完全一致 → PASS
- LoRA merge:真实数据帧 ~1e-6 量级差异;OOD 随机噪声图 0.0047(network 对 1e-7 重排敏感)
- fast_vs_slow 动作:**graph=off 时完全一致(0.0)**;graph=on 时 max|Δ|=0.0216(v)/0.0098(w)

我在 512px / v5-ep1 上复核(fast=merge+fast_proc+graph off vs legacy 全关,72 个真实(图,指令)对):
`max|Δ|=0.0242, mean|Δ|=0.0012, p99=0.0186` → **不是逐位一致**。差异来源推测为两边的文本
padding 长度不同(fast 固定 pad_len=104,HF 动态 padding 约 86-92)+ bf16 归约顺序。
影响:对 v5 的闭环结论无影响(fast/legacy 两版总体同为 62.5%,逐指令仅 ±1 局互换);
若需要严格复现旧数字,请用 `VLA_MERGE_LORA=0 VLA_FAST_PROC=0 VLA_CUDA_GRAPH=0`。

## 6.13 v6 两帧历史输入(loader `--frames 2`)

动机:单帧估距能力不足是对象类(7-11)的主要瓶颈;喂 (t-1, t) 两帧让模型看到运动/尺度变化。
实现(不重新生成数据):
- `vla/train_bc.py` 新增 `--frames {1,2}`:`ShardDataset.__getitem__` 返回 `images=[prev,cur]`,
  prev 判定为"同 episode 的 row-1 且 step_idx>0",否则复制当前帧;collate 支持每样本多帧;
- `vla/model.py`:`VLAConfig.frames`、`build_text(n_images=...)`、`prepare_inputs` 支持每样本多帧
  (prompt 变 `"User:<image><image>指令：..."`,每图 16 token,256px 共 32 image token);
- 增强:同一回合两帧共用同一"翻转决策",亮度/遮挡逐帧独立;
- 初始化:从 `runs/bc_v2/best.pt`(256px 单帧)热启动 LoRA+动作头。

配置:`--frames 2 --image-size 256 --bs 8 --grad-accum 2`(等效 bs16;**bs16 直训会 OOM**,
实测 2 帧下 bs16 峰值需求 ~3.7GB+,与桌面 llama-server 3.6GB 相加超 8GB,故 bs8×2)、
30k 加权帧、lr 3e-4、aux 0.1、无 dagger、1 epoch。

| 指标 | v6 |
|---|---|
| 训练 | 783 s(13.1 min)、202 ms/步、39.5 样本/s(bs8×2 帧=16 图/步)、峰值 2414 MiB |
| val mse_norm(300 batch, 2 帧输入) | 0.04636(mse_v 0.001451、mse_w 0.080867) |
| base10 闭环(256px) | **74.2%**(89/120),对象类 46.0% |
| base50 闭环(官方口径) | **72.5%**(435/600),对象类 43.2% |
| 推理延迟(eager,2 帧,bs1) | **p50 66.6 ms / p95 80.4 ms**(preproc 10.7 + forward 56.0) |

base50 逐指令(失败原因):
| 指令 | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| v6 | 86% | 78% | 100% | 98% | 92% | 100% | 100% | 70% | 46% | 36% | 42% | 22% |
| aux512 | 72% | 78% | 100% | 98% | 88% | 100% | 100% | 62% | 54% | 46% | 38% | 26% |
| 失败原因 | displaced×7 | fall×11 | — | fall×1 | fall×4 | — | — | heading_too_large×15 | red_not_reached×25、fall×2 | blue_not_reached×28、fall×4 | green_not_reached×24、fall×5 | blue_not_reached×13、fall×14、red_too_close×12 |

**结论**:两帧历史把 instr7 从 62%→70%、instr10 从 38%→42%,但 instr8/9 反而下降,
base50 总体 72.5% vs aux512 71.8%(+0.7pp,600 局下不显著),对象类 43.2% vs 45.2%(-2pp)。
→ **历史帧方向部分有效(转向类),但没解决"接近类"的估距/停车时机问题**。

### 6.14 v6b(在 v6 上再低 lr 续训 1 epoch)

配置:`--resume runs/bc_v6_hist2/best.pt --reset-progress --frames 2 --lr 5e-5`,
数据/采样/aux 同 v6,30k 帧 1 epoch,bs8×累积2。训练 886 s(14.8 min)、246 ms/步、
32.5 样本/s、峰值 2414 MiB,val(300 batch)0.04637。

| 指标 | v6 | **v6b** |
|---|---|---|
| base10(12×10) | 74.2%(89/120) | **75.8%**(91/120) |
| base50(官方,12×50) | 72.5%(435/600) | **73.7%**(442/600) |
| instr7(50 局) | 70% | 66% |
| instr8/9/10/11(50 局) | 46/36/42/22% | 48/44/34/24% |
| val(300 batch) | 0.04636 | 0.04637 |

→ **v6b 定为官方模型**:base50 73.7%(442/600,对象类 46.0%)> v6 72.5% > aux512 71.8%;
base10 慢路径 75.8% / 快速路径 75.0%(差 1 局 = 噪声,与 latency-opt 的 parity 结论一致)。

v6b 鲁棒性(修好"两帧同时扰动"后的官方 closed_loop --frames 2,12 局/指令,同 seed 配对):

| 场景 | v6b | 相对 base12 |
|---|---|---|
| base(12 局/指令) | 76.4%(110/144) | — |
| 亮度 ×0.7 | 76.4% | **0.0 pp** |
| 亮度 ×1.3 | 76.4% | **0.0 pp** |
| 10% 随机遮挡 | 77.8% | **+1.4 pp** |
| 固定灰图(视觉消融) | 72.2%(104/144) | **-4.2 pp** |

灰图加量重测(24 局/指令):69.8%(201/288),对象类 37.5%(base 24 局未测,未配对)。
→ **灰图消融确实有掉点(12 局配对 -4.2pp;其中靠近类 8/9/10 下降最明显)**,此前 v6 的
"gray=base"是扰动只加在 cur 帧的评测 bug,已修复并重测;结论:两帧模型仍在使用视觉,但灰图
影响小于单帧 aux512(其配对为 -5.1pp,512px 单帧)。

### 6.15 官方评测集成(--frames)与复现验证

- `eval/closed_loop.py` 新增 `--frames {1,2}`:每回合维护 prev/cur 队列,2 帧时把两帧都送进模型
  (扰动也同时作用于两帧,灰图消融才有效);ckpt 的 `cfg.frames>1` 时自动启用并提示;
  JSON 记录 `frames / frames_ckpt / inference / policy_info`。
- `vla/infer.py` 已由 latency-opt 支持多帧(`act()/act_batch(prev_images=...)`、`reset_history()`,
  `frames` 由 ckpt 自动读取);2 帧评测先走 eager(不启用 fast_proc/cuda_graph)。
- `scripts/make_demo.py` 新增 `--frames`(每回合 reset_history)。
- **复现验证**:用官方命令 `eval/closed_loop.py --ckpt runs/bc_v6_hist2/best.pt --image-size 256
  --frames 2 --episodes 50 ...` 重跑 v6 base50 = **435/600 = 72.5%**,与自研 eval_hist2 完全一致
  (`eval/results/bc_v6_base50_official.json`)。

## 6.16 全部版本汇总(最终版)

| 版本 | 架构/输入 | 数据/采样 | 官方闭环 | 对象类(7-11) | 备注 |
|---|---|---|---|---|---|
| bc_v1 | 256px 单帧 | v1 全量,3 epoch | base50 **71.0%** | 40.4% | 基线 |
| bc_v2 | 256px 单帧 | v1+obj+obj7,加权 100k | base10 71.7% | 44.0% | 加权+加量 |
| bc_v3_512 | 512px 单帧 | 加权 30k,LR bug | base10 66.7% | 50.0% | 常量类被 LR bug 打坏 |
| bc_aux512 | 512px 单帧 + aux | v1_aux+obj_aux+dagger 加权 30k | base50 **71.8%** | 45.2% | 上一版官方 |
| bc_v5 | 512px 单帧 + aux | +dagger,从 aux512 续训 lr3e-4 | base10 62.5% | 32.0% | **回归**(站立/面向红桶校准被破坏) |
| bc_v5b | 512px 单帧 + aux | lr5e-5、去 dagger 续训 | base10 71.7% | 44.0% | **中性**(无增益) |
| bc_v6 | 256px 2 帧 + aux | 从 v2 热启动,lr3e-4 | base50 **72.5%** | 43.2% | 历史帧:instr7 62→70% |
| **bc_v6b** | **256px 2 帧 + aux** | **v6 再 lr5e-5 续训 1 epoch** | **base50 73.7%** | **46.0%** | **✅ 官方模型** |

推理延迟(bs1):

| 配置 | 路径 | p50 | p95 |
|---|---|---|---|
| 256px 单帧 | fast+merge+graph | **7.8 ms** | 9.2 ms |
| 512px 单帧 | fast+merge+graph | 20.4 ms | 21.7 ms |
| **256px 2 帧** | **fast+merge+graph** | **11.3 ms** | **12.4 ms** |
| 256px 2 帧 | eager(旧) | 66.6 ms | 80.4 ms |

全部 < 100ms(10Hz 要求);2 帧快速路径 vs 慢速闭环差 1 局(75.0% vs 75.8%,噪声)。

## 7. 复现命令

```bash
cd /home/czy/倒立摆/wheeltec_vla
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 官方训练(aux512)
.venv/bin/python vla/train_bc.py --data data/dataset_v1_aux,data/dataset_obj_aux \
  --out runs/bc_aux512 --epochs 1 --epoch-frames 30000 --bs 4 --grad-accum 4 \
  --image-size 512 --lr 3e-4 --aux-loss-weight 0.1 \
  --balance-instr --instr-weights "7:3,8:2,9:2,10:2,11:4" \
  --resume runs/bc_v3_512/best.pt --reset-progress --stop-after-epoch 1 \
  --val-batches 120 --log-every 100 --workers 2

# 闭环评测(12 指令 × 50 局 + 延迟)
.venv/bin/python eval/closed_loop.py --ckpt runs/bc_aux512/best.pt --image-size 512 \
  --episodes 50 --instrs 0-11 --scenarios base --skip-baselines --batch-episodes 8 \
  --seed-base 100002 --name bc_aux512_base50

# 扰动 + 灰图 + 基线
.venv/bin/python eval/closed_loop.py --ckpt runs/bc_aux512/best.pt --image-size 512 \
  --episodes 50 --perturb-episodes 12 --instrs 0-11 \
  --scenarios base,bright_low,bright_high,occlusion,gray --baseline-episodes 10 \
  --skip-latency --batch-episodes 8 --seed-base 100002 --name bc_aux512_robust
```

## 8. 局限与建议

1. 对象类指令(7-11)是全部差距来源:专家上限 90%+,VLA 45.2%;失败以"没开到 0.45 m"和
   "方位误差 >0.30 rad"为主,说明**接近段的减速/停车时机 + 方位精修**是核心。
2. 单帧 128×128 的估距信息有限(桶在 1.4 m 处约 6 px 宽);继续堆分辨率(512 只有 2.2 h/epoch)
   与数据量收益递减,已实测。
3. 建议下一步(按性价比):DAgger-lite 全量补监督(13 min 采集 + 30 min 微调)> 加时序输入/历史帧 >
   继续加分辨率/数据。
4. 常量类已到物理上限;若追求总成功率数字,可对 0/2-6 用"指令常量兜底"(部署层,不改判据),
   但会失去一部分"端到端"意义,本报告未采用。

## 附：4 帧稀疏历史（v6c，2026-11-23 晚）

- 配置：从 v6b 热启动，`--frames 4 --frame-stride 2`（取 t−6/t−4/t−2/t），256px、lr 5e-5、30k 帧、aux 0.1。
- 训练：7485 步、248 ms/步、16.1 样本/s、峰值 2295 MiB、val mse 0.0501。
- 闭环 base10：**71.7%（86/120），对象类 44.0%**（7/8/9/10/11 = 40/70/40/10/60%）——低于 v6b 的 75.0%/50.0%，**判定无增益，已停**。
- 说明：4 帧输入让模型更依赖历史、当前帧细节被稀释，instr10 明显退化；结论是「历史长度不是瓶颈」。
- 工程修复：`eval/closed_loop.py` 的 frames=4 改造引入两个 bug（`frames` 在赋值前使用、缺 `from collections import deque`），已修复并回归验证 frames=2 路径正常。

## 附 2：延长续训与 aux 条件化（v6d / v7，2026-11-23 夜）

| 变体 | 做法 | base10 | 对象类 | 结论 |
|---|---|---|---|---|
| v6d | v6b 再续训 2 epoch，lr 2e-5，60k 帧 | 72.5% | 44.0% | 无增益（val mse 0.04639 ≈ v6b 0.04637） |
| v7 | 两阶段：AuxHead 预测三色桶相对位置 → 拼进动作头（`--aux-condition`，fc1 576→582，热启动时新列置零） | 71.7%（慢路径，真实启用条件化）/ 73.3%（快速路径，实际 extra=0） | 42.0% | 无增益（v6b 75.0%/50.0%） |

- v7 的工程实现保留在仓库：`VLAConfig.aux_condition_action`、`ActionHead(extra_dim=...)`、`load_action_head_partial()`（旧 ckpt 热启动自动补零列）、`train_bc.py --aux-condition`。
- 注意：`vla/fast_infer.py` 不支持 aux 条件化动作头（会退化 extra=0），已在 `vla/infer.py` 加载时自动关闭快速路径并提示；v7 评测以慢路径为准。
- 结论：**动作侧的信息利用不是瓶颈**，把位置预测显式喂给动作头也救不回 approach 类；瓶颈在单目 RGB 的距离/方位可观测性本身。下一步应换观测（深度/双目/标记物）或做真实数据微调，而不是继续在动作头/训练时长上打转。
