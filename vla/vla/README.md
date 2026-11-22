# vla/ — DP2 平衡车 VLA 模型(行为克隆 + 闭环评测)

> 契约:[`specs/VLA-TASK-SPEC.md`](../specs/VLA-TASK-SPEC.md) 第 2/3/4/7 节。
> 负责人:vla-model;写作用域 `vla/`、`eval/`、`runs/`。

## 1. 架构

```
128x128 RGB 图像 + 中文指令
        │
        ▼
SmolVLM-256M-Instruct(本地 assets/hf/):视觉塔 SigLIP-base 冻结 + LM(576 隐藏/30 层)挂 LoRA
        │  取最后一个 token 的 hidden state(576 维)
        ▼
动作头:LayerNorm -> Linear(576,256) -> GELU -> Linear(256,2) -> tanh
        │
        ▼
(v/0.35, w/1.0) 反归一化 -> (v m/s, w rad/s),10 Hz 输出给底层平衡环
```

- 基座:**SmolVLM-256M-Instruct**(`assets/hf/`,13 个文件 494 MB),`local_files_only=True`,全程不联网。
- 训练:BC + MSE,bf16,AdamW,LoRA(r=16, α=32,dropout 0)作用在 LM 的
  `q_proj/k_proj/v_proj/o_proj/gate_proj/up_proj/down_proj`;视觉塔冻结(**无视觉梯度回传**);
  动作头 fp32 从头训练。可训练参数 **5.92 M / 262.4 M(2.3%)**。
- 动作空间(spec §2):`v ∈ [-0.35,0.35] m/s`、`ω ∈ [-1,1] rad/s`,输出 `a_norm=[v/0.35, ω/1.0]`,
  `tanh` 限幅后反归一化;chunk=1(10 Hz 单步)。
- 数据格式(spec §3):`data/<name>/shards/*.npz`;`scene_seed % 10 ∈ 0..7` 训练、`8..9` 验证。

## 2. 关键实现细节(踩坑记录)

### 2.1 transformers 5.x 没有 `AutoModelForVision2Seq`
改用 `AutoModelForImageTextToText`,实际加载到 `Idefics3ForConditionalGeneration`。
动作只需要 hidden state,**不需要 lm_head**,因此训练/推理都直接调内层 `Idefics3Model`:
- 全模型 forward(算 4.9 万词表 logits):256px/bs16 一步 **273 ms**
- 只调内层模型(不算 logits):同样配置一步 **209 ms**(省 ~23%)

### 2.2 Idefics3 默认切图会爆炸
`do_image_splitting=True` 时处理器会先把 128×128 图放大到 `size.longest_edge`(默认 2048),
再按 `max_image_size` 切成 (2048/512)² = **17 个 512×512 crop**,单帧 `input_ids`=1146;
若 `max_image_size=128` 则切 257 张,`input_ids`=18982(超过 8192 上限)。
本项目设置 **`do_image_splitting=False`**,并把 `size`/`max_image_size` 都设为目标边长,
保证每帧只有 1 张 global 图:

| 输入边长 | image token 数 | input_ids 总长(单条指令) | pixel_values |
|---|---|---|---|
| 128 | 4 | 26 | [1,1,3,128,128] |
| 256 | 16 | 38 | [1,1,3,256,256] |
| 384 | 36 | 58 | [1,1,3,384,384] |
| 512 | 64 | 86 | [1,1,3,512,512] |

image token 数必须等于 `(边长/16)^2 / 16 = (边长/64)^2`,同步写 `processor.image_seq_len`,
否则 forward 报 `Image features and image tokens do not match`。

### 2.3 prompt 模板(用 chat template,与 spec §4 一致)
```python
messages = [{"role": "user", "content": [
    {"type": "image"},
    {"type": "text", "text": "指令：{instruction}"},
]}]
text = processor.apply_chat_template(messages, add_generation_prompt=True)
```
实测产出及 token 化:
```
'<|im_start|>User:<image>指令：靠近红桶<end_of_utterance>\nAssistant:'
→ input_ids = [<|im_start|>, User, :, <64 个 image token>, 指令, ：, 靠近红桶, <end_of_utterance>, \n, Assistant, :]
```
512px 时整条 prompt 64 image token + 25 文本 token = **89 token**,decode 可完整还原(见
`runs/probe/probe_processor.py` 输出)。`Assistant:` 之后的最后一个 token 的 hidden 作为动作特征。

### 2.4 DataLoader 必须"整 shard 物化 + 分片局部性"
`np.load("xxx.npz", mmap_mode="r")` 返回的是 `NpzFile`,**不做内存映射**:每次
`d["images"]` 都会重读整个 122 MB 数组(约 25 ms/样本),16 样本一批要 400 ms。
修法:
1. `_npz()` 一次性物化成 dict 并做 LRU 缓存(每 shard 约 123 MB,默认缓存 3 个);
2. 每 epoch 按 shard 分组打乱(先随机 shard 顺序、再 shard 内随机),保证每个 shard
   每 epoch 只读一次,而不是随全局 shuffle 反复抖动读盘。

实测(runs/synth_data,256px,bs16):**修前 37.3 样本/s(data=404 ms/批)→ 修后 60.0 样本/s
(data=5~24 ms/批)**;12.5 万帧/epoch 约 **35 min**。

### 2.5 分辨率/批量取舍(8 GB 显存,桌面已占约 3.6 GB,可用约 4.4 GB)
实测(合成数据,bs16/bf16,LoRA,含 fwd+bwd+AdamW):

| 图像边长 | image token | batch | 单步耗时 | 吞吐 | GPU 峰值(本进程) |
|---|---|---|---|---|---|
| 128 | 4 | 16 | ~184 ms | ~87 样本/s | 1473 MiB |
| 256 | 16 | 16 | ~209 ms | ~77 样本/s | 2947 MiB |
| 384 | 36 | 8 | ~230 ms | ~35 样本/s | 2914 MiB |
| 512 | 64 | 4 | ~225 ms | ~18 样本/s | 2622 MiB |

源图只有 128×128,512 上采样纯浪费算力;256 档在吞吐与空间分辨率(4×4 token 覆盖 70° FOV)
之间折中,是默认配置。若闭环发现转向类指令精度不够,再切 384。

## 3. 命令

```bash
cd /home/czy/倒立摆/wheeltec_vla

# 训练(默认 256px / bs16 / LoRA r16 / 6 epoch)
.venv/bin/python vla/train_bc.py --data data/dataset_v1 --out runs/bc_v1 --epochs 6 --bs 16

# 合成数据自测(不需要 sim/ data/):生成 100 回合假数据到 runs/synth_data
.venv/bin/python runs/make_synth_data.py --out runs/synth_data --episodes 100
.venv/bin/python vla/train_bc.py --data runs/synth_data --out runs/synth_smoke --epochs 2 --bs 16

# 推理 + 延迟(预热 20 次,测 100 次,要求 p95 < 100 ms)
.venv/bin/python vla/infer.py --ckpt runs/bc_v1/best.pt --bench --n 100 --batch 8

# 闭环评测(12 指令 x 50 局 + 亮度/遮挡/灰图扰动 + random/constant 基线)
.venv/bin/python eval/closed_loop.py --ckpt runs/bc_v1/best.pt --episodes 50 --instrs 0-11
```

`eval/closed_loop.py` 只依赖 spec §1.4 的 `VLAEnv`/`success_func`,默认 `--env-module sim.env_vla`;
`sim/env_vla.py` 就绪前可用 `--env-module mock`(内置合成假环境)自测链路,正式验收必须用默认值。

## 4. 产物

```
runs/<name>/
  best.pt           # 验证集 action MSE 最优(只存 LoRA ~23 MB + 动作头)
  last.pt
  train_log.json    # 每 epoch loss/val MSE/耗时/显存 + 数据统计 + 完整配置
eval/results/*.json # 闭环评测:每指令成功率/扰动掉点/延迟 p50/p95/消融基线
```

checkpoint 里带完整配置(`VLAConfig`):图像边长、image_seq_len、LoRA 超参、prompt 模板、
动作归一化系数、训练 args,推理时自动复现,不依赖外部参数。

## 4.4 数据集混采能力(`--data` 多目录 + 加权采样)

- `--data a,b,c`:多个数据目录合并建索引(shard 顺序、来源目录都会记录)。
- `--balance-instr`:每条指令先按"帧数相同"为基准,再乘 `--instr-weights`(反制长尾)。
- `--epoch-frames N`:每个 epoch 只采 N 帧(控制单 epoch 时长,LR schedule 按 N/bs 计算)。
- `--source-weights "data/dataset_dagger:6"`:按**目录来源**给行加权(DAgger / 补集数据占比低时抬高),
  采样分布会打印到日志(含各目录的期望占比)。
- 采样后仍按 shard 分组,保证每个 shard 每 epoch 只读一次。

## 4.5 辅助监督(备选方案,已实现待启用)

若 256/512px 的行为克隆仍卡在对象类指令,可启用**桶相对位置辅助回归头**:
- 数据侧:`sim/gen_dataset.py --aux` 会给 shard 加 `aux: float32 [N,6]`,
  顺序 `[dx_red, dy_red, dx_blue, dy_blue, dx_green, dy_green]`(米,机器人坐标系)。
- 模型侧:`VLAConfig.aux_dim=6` → `VLAModel` 多一个 `AuxHead`(LayerNorm→Linear(576,256)
  →GELU→Linear(256,6)),与动作头共享同一个 last-token hidden。
- 训练侧:`--aux-loss-weight 0.1` 启用,`loss = MSE(动作,归一化) + w * MSE(aux/3.0)`,
  shard 无 aux 时按 mask 跳过;验证集额外报 `aux_mse_m`(每维 m²);checkpoint 自动带 `aux_state`。
- 已用 `data/aux_smoke`(12 回合/600 帧)验证:数据管线/前向/反向/存取全部通过。

## 5. 自测工具(不参与验收)

- `runs/make_synth_data.py`:合成"棋盘地面 + 三个彩色桶"的 128×128 假场景,按 spec §1.3
  专家控制器生成可学的 (图像, 指令, 动作) → 用来验证训练链路/视觉梯度,而 `data/` 由 sim-data 负责。
- `eval/synth_scene.py`:假场景渲染 + 专家 + 成功判据 + `MockVLAEnv`(与 spec §1.4 同接口)。

## 6. 结果

### 6.1 合成数据(runs/synth_data:100 回合 = 5000 帧;仅验证链路,非验收)
配置:256px / LoRA r16 / bs16 / 3 epoch / 4000 训练帧 + 1000 验证帧。

| epoch | train tail loss(MSE,归一化) | val mse_norm | val mse_v | val mse_w |
|---|---|---|---|---|
| 1 | 0.126 | **0.159** | 0.0045 | 0.282 |
| 2 | 0.109 | 0.208 | 0.0044 | 0.381 |
| 3 | 0.094 | 0.226 | 0.0014 | 0.441 |

按指令拆开看(val mse_norm,epoch1):常量类 0-6 为 0.000-0.008(已学到),
闭环视觉类 7-11 为 0.21/0.54/0.43/0.35/0.44(合成场景只有 100 个,样本不够泛化)。
→ 卷积/注意力/动作头链路与视觉梯度都通,但闭环视觉指令必须等真实数据集。
显存峰值 2668 MiB(本进程),单步 246-255 ms(共享 GPU 下;干净探针 209 ms)。

### 6.2 推理延迟(256px,bs1,GPU 预热 10 次测 50-100 次)

| 配置 | p50 | p95 | 备注 |
|---|---|---|---|
| bs=1 | 48.6-63.1 ms | 51.5-73.2 ms | 满足 10 Hz(<100 ms);随宿主负载变化 |
| bs=8 | 91 ms/批 | 97.5 ms/批 | 11.5 ms/样本(闭环评测批量用)|

### 6.2b 图像分辨率 A/B(data/smoke 真实数据,30 回合/1300 训练帧/200 验证帧,6 epoch,LoRA r16)

| 边长 | image token | bs | 单步 | 吞吐 | GPU 峰值 | val mse_norm | val mse_v | val mse_w | 延迟 p50/p95(bs1) | 闭环成功率(3 局/指令,36 局) |
|---|---|---|---|---|---|---|---|---|---|---|
| **128** | 4 | 16 | 204 ms | **78.5 样本/s** | **1558 MiB** | **0.0129** | 0.00147 | **0.0138** | 74.6 / 88.0 ms | **72.2%** (26/36) |
| 256 | 16 | 16 | 305 ms | 52.4 样本/s | 2668 MiB | 0.0188 | 0.00143 | 0.0258 | 72.5 / 93.5 ms | 66.7% (24/36) |
| 384 | 36 | 8 | 1050 ms* | — | 2547 MiB | 0.032(未跑完) | — | — | — | — |

\* 384 那轮与 sim-data 生成数据集抢 CPU,单步 400~1000 ms,6 epoch 没跑完即中止;
它在这份小数据上验证集 MSE 反而更差,说明"更高分辨率≠更好",源图本来只有 128×128。

说明:smoke 验证集只含 5/6/7/8 四条指令(按 scene_seed 划分),闭环每指令仅 3 局,
单条指令 ±33pp 噪声,128 vs 256 的 72.2% vs 66.7% 不构成显著差异;但 128 训练快 1.5×、
显存省 40%,且逐 epoch 验证 MSE 下降更顺(0.052→0.029→0.024→0.025→0.019→0.013),
因此正式训练默认取 128px,256px 作为对照保留。

### 6.2c 训练吞吐基准(data/dataset_test:14 shard / 32850 帧,100 step 实测)

| 配置 | 单步 | 吞吐 | 本进程峰值 |
|---|---|---|---|
| 128px / bs16 | 375 ms* | 42.7 样本/s* | 1558 MiB |
| 256px / bs16 | 440 ms* | 36.3 样本/s* | 2668 MiB |

\* 测这组时 sim-data 正在生成 dataset_v1 + 其他队友在跑 VAD,CPU 抢得厉害,
数值约为空闲时的一半(空闲实测 128px/bs16 = 78.5 样本/s、256px/bs16 = 52.4 样本/s)。
**按空闲值估算:12.5 万帧/epoch 在 128px/bs16 下约 27-28 min,256px/bs16 约 40 min。**
8 GB 卡上稳定 batch:128px/bs16(1558 MiB)、256px/bs16(2668 MiB);128px/bs32 在
llama-server 已占 3.6 GB 时 OOM,故正式训练用 bs16。

### 6.2d 场景扰动与消融可在命令行跑通(data/smoke,ab_128 模型,4 指令 x 2 局)

`--scenarios base,bright_low,bright_high,occlusion,gray` 实测:
base 87.5% / bright_low 87.5% / bright_high 87.5% / occlusion 87.5% / **gray 62.5%**(-25pp)。
样本极小只验证链路,正式数字见 6.3。

### 6.3 真实数据 data/dataset_v1(2500 回合 / 125,000 帧 / 50 shard,专家成功率 100%,已启用桶碰撞)

正式训练配置(Lead 拍板):
```
.venv/bin/python vla/train_bc.py --data data/dataset_v1 --out runs/bc_v1 \
    --epochs 3 --bs 16 --image-size 256 --lr 1e-3 --patience 2 --seed 0 \
    --val-batches 120 --log-every 100
```
- train/val:按 `scene_seed % 10`(0-7 / 8-9)= **100,150 / 24,850 帧**;6259 步/epoch,总 18,777 优化步
- 每个 epoch 存 `ep{N}.pt` + `optim_ep{N}.pt`(便于中途评测),验证集最优另存 `best.pt`
- 静默下 256px/bs16 ≈ 63 样本/s,单 epoch ≈ 26 min;GPU 峰值 2668 MiB

(结果表见 6.4;ep1 中途结果如下)

### 6.4 训练结果(runs/bc_v1)

**ep1(1 epoch = 6259 步,1754 s)**
- 验证集 action MSE(归一化)= **0.06480**,mse_v = 0.004284 m²/s²,mse_w = 0.094628 rad²/s²
- 每指令 mse_norm:0/2-6 已≈0;1=0.053、7=0.019、8=0.106、9=0.161、10=0.111、11=0.357
- 闭环(12 指令 × 10 局,base,test seed 段)= **70.8%**(85/120):
  0:100% 1:70% 2:100% 3:100% 4:90% 5:100% 6:100% 7:30% 8:60% 9:40% 10:20% 11:40%
- 开销:273 ms/步、58.7 样本/s、GPU 峰值 2668 MiB

**ep2/ep3(续训,余弦后半程)**

| epoch | 验证 mse_norm | mse_v | mse_w | 每 epoch 用时 |
|---|---|---|---|---|
| 1 | 0.06480 | 0.004284 | 0.09463 | 1754 s |
| 2 | 0.06313 | 0.003819 | 0.09508 | 1692 s |
| **3(best)** | **0.05895** | **0.002745** | 0.09549 | 1786 s |

ep3 每指令 mse_norm:0/2-6 全为 0.0000-0.0001;1=0.054、7=0.017、8=0.100、9=0.147、10=0.092、11=0.318。
→ 常量类已收敛到底;剩余误差集中在"靠近类"(8/9/10)和 11,与闭环一致。

**最终闭环(12 指令 × 50 局,base,test seed 段 100002):426/600 = 71.0%**

| 指令 | 0保持 | 1前进 | 2后退 | 3左转 | 4右转 | 5左旋 | 6右旋 | 7面向 | 8靠近红 | 9靠近蓝 | 10靠近绿 | 11绕红靠蓝 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 成功率 | 86% | 74% | 100% | 98% | 92% | 100% | 100% | 58% | 46% | 40% | 38% | 20% |

- 单步延迟(bs1,256px,GPU 预热 20 测 100):**p50 = 48.1 ms,p95 = 54.4 ms**(满足 <100 ms)
- 动作 EMA(0.5)对照(12×10,同 seed 序列):见 `eval/results/bc_v1_base10_ema05.json` 与
  `bc_v1_base10_noema.json`,结论见报告。
- 512px 可行性微基准(data/dataset_v1,30 步):**bs4+梯度累积4 = 250-353 ms/步、15.6 样本/s、
  峰值 2231 MiB**;bs8+累积2 **OOM**(需 ~3.7 GB,加上桌面 llama-server 3.6 GB 超 8 GB)。
  → 512px 可跑但吞吐只有 256px 的 ~1/4(12.5 万帧/epoch ≈ 2.2 h),不适合全量重训。
- 扰动/消融/基线(`eval/results/bc_v1_robust.json`):
  base 用 12 局/指令(与扰动同 seed、144 局)配对比较:

| 场景 | 成功率 | 相对 base12 |
|---|---|---|
| base(12 局/指令) | **75.0%**(108/144) | — |
| 亮度 ×0.7 | 75.0% | **0.0 pp** |
| 亮度 ×1.3 | 73.6% | **-1.4 pp** |
| 10% 随机遮挡 | 73.6% | **-1.4 pp** |
| 固定灰图(视觉消融) | 62.5% | **-12.5 pp** |
| random 基线 | 5.8% | -69.2 pp |
| constant(全 0)基线 | 11.7% | -63.3 pp |

  逐指令灰图消融:靠近类 8/9/10 全部掉到 **0%**、11 从 33%→25%,而常量类 2-6 不变
  (本来就与图像无关)→ 证明视觉在"靠近桶"任务上被真正使用。
  亮度/遮挡掉点 ≤1.4pp,远优于 spec 要求的 ≤10/15pp。

### 6.5 v2:按指令加权 + 对象类加量数据(Lead 决策)

数据:`data/dataset_v1`(125k)+`dataset_obj`(60.55k,仅 7-11)+`dataset_obj7`(13k,仅 7)
= **198,550 帧**(train 158,750 / val 39,800,按 scene_seed%10);
加权采样 `--balance-instr --instr-weights "7:3,8:2,9:2,10:2,11:4" --epoch-frames 100000`
→ 每个 epoch 常量类各 5,000、**7=15,000、8/9/10 各 10,000、11=20,000**(视觉类占 65%),
从 `bc_v1/best.pt` 热启动(`--reset-progress`,lr 5e-4)训 1 epoch(6250 步,29 min)。

| 指标 | bc_v1(3 epoch,v1 数据) | bc_v2(1 epoch,加权+加量) |
|---|---|---|
| 验证集 mse_norm(各自 val) | 0.05895 | **0.05543** |
| base10 闭环(12×10,test seed) | 73.3% | 71.7% |
| 对象类(7-11)均值 | 40.0%(60/60/40/30/10) | 44.0%(70/60/40/30/20) |

→ 加权+加量只把指令 7(+10pp)、11(+10pp)拉起,对象类均值 +4pp、总体未提升;
结合 256px 只有 4×4 image token 的事实,判据瓶颈是**空间分辨率**而不是数据量。
下一步做 512px(64 token,8×8 网格)微调,见 6.6。

### 6.6 512px 微调(bc_v3_512)

- 数据/采样同 v2,`--epoch-frames 30000 --bs 4 --grad-accum 4`(等效 bs16),lr 3e-4,
  从 `bc_v2/best.pt` 热启动 1 epoch;实测 512px 训练 **~250-290 ms/步(bs4)、峰值 2231 MiB**,
  推理 **p50 53.8 ms / p95 57.3 ms**(bs1,<100 ms 达标)。
- 直接拿 256px 权重在 512 上零样本推理会掉到 58.3%(token 数 16→64 分布不匹配),必须在 512 上训练。

512px 微调 ep1(30k 加权帧)结果:
- 验证集(合并 val,120 batch 子集)mse_norm **0.03132**(v2 合并 val 为 0.05543,v/mse_v 0.001377、mse_w 0.0514)
- base10 闭环 **66.7%**(80/120),对象类(7-11)均值 **50.0%**(7=90% 8=60% 9=40% 10=30% 11=30%)
  vs v2 的 71.7% / 44.0%:对象类 +6pp,但**常量类指令 0 从 100% 掉到 0%**(失败原因全部是 `displaced`,
  即模型不再输出 0 速度)→ 总成功率反而降 5pp。
- 失败原因分布(新标签):instr0 `displaced`×10;instr1 `fall`×2;instr4 `fall`×3;
  instr8 `red_not_reached`×4;instr9 `blue_not_reached`×5 + `fall`×1;
  instr10 `green_not_reached`×6 + `fall`×1;instr11 `fall`×3 + `blue_not_reached`×3 + `red_too_close`×1。
- 该轮踩到一个 LR schedule bug 并已修:加权采样 + `--epoch-frames` 时 `total_steps` 误用未采样的
  数据集长度(19844 vs 实际 1875),导致 cosine 被拉长、学习率整轮几乎停在 3e-4;修复后
  `steps/epoch` 按 `epoch_frames/bs` 计算。

判定(Lead 规则 45%≤对象类<65%):转**辅助监督路线**(512px + aux 回归头 0.1),见 6.8。

### 6.8 辅助监督 512px(bc_aux512)

数据:`data/dataset_v1_aux`(125k,与 v1 逐条同 seed/图像/动作)+`data/dataset_obj_aux`(60.5k),
shard 多一个 `aux=[dx_red,dy_red,dx_blue,dy_blue,dx_green,dy_green]`(米,车体系);
采样同 v2 权重/30k 帧;`--aux-loss-weight 0.1`;从 `bc_v3_512/best.pt` 热启动、修好 LR schedule 后训 1 epoch。

| 指标 | bc_v1(256) | bc_v2(256 加权) | bc_v3_512 | **bc_aux512** |
|---|---|---|---|---|
| val mse_norm | 0.05895 | 0.05543 | 0.03132 | **0.03014** |
| base10 闭环 | 73.3% | 71.7% | 66.7%* | **72.5%** |
| 对象类(7-11)均值 | 40.0% | 44.0% | 50.0% | 44.0% |
| 训练吞吐/峰值 | 59.2 样本/s / 2668MiB | 67.7 / 2668 | 15.1 / 2231 | 17.5 / 2233 |

\* v3 那轮受 LR schedule bug 影响(学习率整轮≈3e-4),常量类被破坏(instr0 0/10)。

aux 回归头验证集 RMSE(米):red dx 0.203 / dy 0.098、blue dx 0.101 / dy 0.266、green dx 0.050 / dy 0.030。

**结论**:三种手段(数据加量+加权、512px 64 token、辅助方位回归)都没把对象类从 40-50% 推到 65%;
但 aux512 总体最好、验证 MSE 最低、常量类正常。8/9/10 的失败以 `not_reached`(没开到 0.45m)为主,
fall 少数 → 瓶颈更像"单帧估距 + 10Hz 闭环接近策略"的能力边界,继续加分辨率/数据收益递减。

**bc_aux512 正式评测(官方验收模型,12 指令 × 50 局,600 局)**:431/600 = **71.8%**;
延迟 bs1 **p50 49.3 ms / p95 51.2 ms**。逐指令与专家上限、失败原因:

| 指令 | VLA(50 局) | 专家尝试上限 | 差值 | 失败原因 |
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

对象类均值 45.2%;常量类 2/3/5/6 已到专家上限。**8/9/10 失败以 `not_reached` 为绝对主导**
(而不是 fall/振荡),说明"接近类"主要卡在偏航对准与减速时机,属于可用 DAgger-lite 补监督的分布漂移问题。

**robust(aux512,`eval/results/bc_aux512_robust.json`)**:亮度×0.7 = 73.6%、×1.3 = 72.2%、
10% 遮挡 = 72.2%、固定灰图 = 67.4%(视觉消融,相对 base10 72.5% 约 -5.1pp)、
random 5.8%、constant 11.7%。亮度/遮挡掉点 ≤1.1pp。

### 6.9 DAgger-lite 偏差诊断

用 aux512 策略在同 seed 场景回放(12 回合/592 帧),比较"策略实际动作"与"专家标签":

| 指令 | 策略 mean ω | 专家 mean ω | Δω |
|---|---|---|---|
| 0 保持 | -0.025 | 0 | -0.025 |
| 1 前进 | +0.021 | **+0.608** | -0.587 |
| 2-6 常量类 | 与专家一致 | — | \|Δω\| ≤0.01 |
| 7 面向红 | -0.026 | **+0.658** | -0.685 |
| 8 靠近红 | -0.068 | +0.403 | -0.472 |
| 9 靠近蓝 | -0.042 | +0.137 | -0.178 |
| 10 靠近绿 | -0.039 | **-0.670** | +0.631 |
| 11 绕红靠蓝 | +0.007 | -0.527 | +0.534 |

→ **系统性转向不足(under-steering)**:在策略自己偏航、需要修正的状态上,专家给的 |ω| 是策略的
3-10 倍。这解释了 8/9/10 的 `not_reached` 与 1 的 fall,也说明 DAgger(在策略访问状态上用专家动作
做标签)能补到关键监督。

### 6.10 DAgger 微调结果(消融,非最终模型)

DAgger 数据:`data/dataset_dagger/` 240 回合 / 11,846 帧 / 5 shard(12 指令 × 20 局,策略执行 +
专家标签 + aux),`scripts/ds_validate.py` **PASSED**;行为策略自成功率 69.6%。
微调:512px,`--data v1_aux,obj_aux,dagger --source-weights "dataset_dagger:6"`(dagger 期望占比 26.7%),
30k 加权帧、lr 2e-4、aux 0.1、从 bc_aux512 热启动 1 epoch。

| 指标 | bc_aux512 | bc_dagger |
|---|---|---|
| base10 闭环 | 72.5% | **72.5%** |
| 对象类均值 | 44.0% | **44.0%** |
| 每指令(7-11) | 70/60/40/20/30 | 70/50/40/30/30 |

→ DAgger 一轮微调**没有提升**(对象类 44% < 53% 门槛),按 Lead 决定**回退 bc_aux512 为最终模型**,
DAgger 作为消融写入报告。原因推断:单向 BC 的 under-steering 已被 26.7% 的 DAgger 样本部分纠正,
但 1 epoch、30k 帧、冻结视觉塔的容量不足以把这个行为改过来;DAgger 需要多轮迭代(每轮用新策略重新采集)。

### 6.7 VLA vs 专家上限(expert 在生成数据时的"尝试成功率"= 同一判据下专家自身也有失败):

| 指令 | 0保持 | 1前进 | 2后退 | 3左转 | 4右转 | 5左旋 | 6右旋 | 7面向 | 8靠近红 | 9靠近蓝 | 10靠近绿 | 11绕红靠蓝 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| VLA(12×50) | 86% | 74% | 100% | 98% | 92% | 100% | 100% | 58% | 46% | 40% | 38% | 20% |
| 专家尝试上限 | 85% | 96% | 100% | 98% | 89% | 100% | 100% | 100% | 96% | 93% | 96% | 61% |
| 差距 | +1 | -22 | 0 | 0 | +3 | 0 | 0 | **-42** | **-50** | **-53** | **-58** | **-41** |

  → 常量类已到"底层动力学+判据"的物理上限(甚至略超,因 eval seed 段不同);
  71.0% 的总差距几乎全在 7-11,不能用底层漂移解释。

