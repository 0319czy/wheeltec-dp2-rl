# vla-model 交付报告:DP2 平衡车 VLA(行为克隆 + 闭环评测)

> 负责人:vla-model;写作用域 `vla/`、`eval/`、`runs/`。
> 契约:`specs/VLA-TASK-SPEC.md`(v1);硬件:RTX 5060 Laptop 8 GB(桌面 llama-server 常占 3.6 GB,可用约 4.4 GB)、16 核、14 GB 内存。
> 本文件是给 Lead 汇总验收报告用的**数字清单**,细节见 `vla/README.md`。

## 1. 结论摘要

| 项目 | 结果 |
|---|---|
| 官方验收模型 | `runs/bc_aux512/best.pt`(SmolVLM-256M + LoRA r16 + 动作头 + 辅助回归头;512px / 64 image token) |
| 闭环(base 12 指令 × 50 局 = 600 局,test seed 段) | **431/600 = 71.8%** |
| 单步推理延迟(bs1,512px) | **p50 49.3 ms / p95 51.2 ms**(<100 ms 达标) |
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
