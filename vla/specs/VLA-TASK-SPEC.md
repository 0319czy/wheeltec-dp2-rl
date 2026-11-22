# DP2 平衡车 VLA 任务契约 v1

> 本文件是仿真数据、模型训练、评测、部署三方的唯一接口契约。任何一方改动必须先改本文件并由 Lead 确认。
> 项目根：`/home/czy/倒立摆/wheeltec_vla`；python：`.venv/bin/python`（Python 3.12 + torch 2.9.1+cu128）。
> 参考项目 `/home/czy/倒立摆/wheeltec_rl` 只读，禁止修改（Lead 已把必要资产复制到 `assets/lowlevel/`）。

## 0. 目标与分层架构

真实机器人是 STM32F103 主控的二阶平衡车，不可能跑 VLA，也不可能 100 Hz 推理。因此采用业界通行的分层架构：

```
笔记本(上位机)                                 车(下位机)
相机图像 + 文本/语音指令
        │
        ▼
  VLA 模型 (10 Hz)  ──(v, ω) 速度指令──►  底层平衡/速度环 (100 Hz)  ──► 左右轮 PWM
        ▲                                        │
        └──────────── 遥测(倾角/轮速/电压) ◄──────┘
```

- **VLA 只负责“看懂场景 + 听懂指令 → 给 10 Hz 的速度指令 (v, ω)”**。
- 平衡、电流限幅、死区保护由 100 Hz 底层完成（仿真里是已有 PPO 策略，实车里是固件）。
- v: 前进速度 m/s（正=向前），ω: 偏航角速度 rad/s（正=左转/逆时针）。

## 1. 仿真环境契约（`sim/`，负责人：sim-data）

### 1.1 物理与底层
- MuJoCo 模型 `sim/scene_vla.xml`，以 `assets/lowlevel/dp2_nopole3d.xml`（一阶 3D 底盘，6-DOF freejoint）为基础扩展。
- 物理 timestep `0.005 s`；底层策略每 `0.01 s` 决策一次（100 Hz）；VLA 每 `0.1 s` 决策一次（10 Hz，每个 VLA step = 10 个底层 step）。
- 底层 = `assets/lowlevel/balance_model.zip` + `assets/lowlevel/vecnormalize.pkl`（SB3 PPO，9 维观测，含 v_cmd/w_cmd）。加载时必须用 VecNormalize 的 `normalize_obs`（`training=False`），**否则策略输出恒 0**（已踩坑）。
- 当前选用的一对来自 `wheeltec_rl/runs/mujoco_cmd3d/fix_mixed-v0.3-w1.2`（v6C 那一对不能边走边转、已弃用；对比见 `docs/lowlevel-selection.md`）。Lead 扫了 23 个候选（固定 7 条指令 × 5 s），只有这一对 7 项判据全过：站立 0.01 m/0.11 rad、前进 0.93–1.11 m、后退 0.56 m、左转 +2.0~2.5 rad/0.30 m、右转 −1.9~−2.0 rad/0.34 m、左旋 +3.0 rad/0.20 m、右旋 −3.1 rad/0.01 m。
- 底层接口 = `assets/lowlevel/env_cmd3d.py` 的 `Cmd3D`：设置 `env.v_cmd/env.w_cmd` 后调用 `env.step(action)`。注意 VecNormalize 的 obs 顺序是 `[pitch*2, pitch_rate*0.4, yaw_rate*0.5, wl/20, wr/20, vx*2, vy*2, v_cmd*2, w_cmd*0.5]`。

### 1.2 场景
- 地面：8 m × 8 m 平面，棋盘/网格纹理（MuJoCo `builtin="checker"`），材质外观可随机（亮度/偏移）。
- 目标物：3 个圆柱形“桶”，半径 0.07 m、高 0.22 m，颜色为 **红 (1,0.15,0.15) / 蓝 (0.15,0.3,1) / 绿 (0.15,0.85,0.2)**。
  - 桶是**真实碰撞体**（刚体、不可推动）：机器人撞到桶会停/翻车，这是任务的一部分；数据生成时**丢弃碰撞或倒地失败的回合**（只保留专家成功回合做 BC）。
- 三个桶的颜色在每回合内**随机排列**（同一位置可能是红也可能是蓝），迫使模型必须看图像+读指令，不能背位置。
  - 位置：以起点为原点，极坐标半径 `0.65–1.4 m`，**方位角 ±30°**（保证开局落在相机 ±35° 水平视场内，否则目标不可观测、任何无记忆 VLA 只能猜），相互间距 ≥ 0.45 m。
  - 「绕开红桶前往蓝桶」这一条把红桶放在机器人→蓝桶连线的横向偏移 0–0.25 m、距离 0.7–1.0 m 处，这样绕行才真的需要视觉避让。
  - （早期版本用过 ±170°：目标 79% 概率不在画面内，属于不可观测任务，已废弃。）
- 相机：固定在车体上的“头”相机，`128×128` RGB，FOV≈70°，距地面约 0.18 m，朝车头前方、略微下俯，能同时看到桶与地面对比纹理。
- 可选：4 面低矮围墙（高 0.3 m）作为视觉参照并防止跑出场景；若实现，不得影响任务判据。
- 随机化（每回合）：桶位/颜色排列、起点位姿（x,y ±0.1 m，yaw ±0.2 rad）、光照（`option`/`light` 强度 0.7–1.3）、相机图像噪声（σ≈2/255）、地面纹理偏移。

### 1.3 指令集与专家控制器（12 条，固定 id）

| id | 文本 | 专家行为（10 Hz 输出） | 成功判据（回合结束时判定） |
|---|---|---|---|
| 0 | 保持站立 | v=0, ω=0 | 位移 < 0.10 m 且 \|Δyaw\| < 0.20 rad |
| 1 | 向前走 | v=0.25, ω=0 常量 | 车头方向前进位移 ∈ [0.5, 1.9] m |
| 2 | 向后退 | v=-0.20, ω=0 常量 | 车头反方向位移 ∈ [0.35, 1.5] m |
| 3 | 向左转 | v=0.25, ω=+0.6 常量 | Δyaw ∈ [1.0, 3.5] rad 且前进位移 > 0.15 m |
| 4 | 向右转 | v=0.25, ω=-0.6 常量 | Δyaw ∈ [-3.5, -1.0] rad 且前进位移 > 0.15 m |
| 5 | 原地左旋 | v=0, ω=+1.0 常量 | Δyaw ∈ [1.5, 5.0] rad 且位移 < 0.40 m |
| 6 | 原地右旋 | v=0, ω=-1.0 常量 | Δyaw ∈ [-5.0, -1.5] rad 且位移 < 0.40 m |
| 7 | 面向红桶 | v=0，ω = clip(1.5·θ_err, ±1.0)（θ_err=红桶方位角） | 终止时 \|θ_err\| < 0.30 rad 且未倒地 |
| 8 | 靠近红桶 | d>0.95 时 v=0.32；d≤0.95 时 v=clip(1.0·(d-0.40),0,0.32)；ω=clip(1.5·θ_err,±1.0) | 终止时 d_red < 0.45 m |
| 9 | 靠近蓝桶 | 同 8，目标蓝桶 | 终止时 d_blue < 0.45 m |
| 10 | 靠近绿桶 | 同 8，目标绿桶 | 终止时 d_green < 0.45 m |
| 11 | 绕开红桶前往蓝桶 | 朝蓝桶 v/ω 同上，另加红桶斥力：若 d_red<0.8，ω += 0.8·sign(叉积)·(0.8-d_red)/0.8 的转向项 | 终止时 d_blue < 0.50 m 且全程 min d_red > 0.40 m |

- 回合固定 **50 个 VLA step（5.0 s）**，超出即结束；倒地（pitch > 0.4 rad 或 z < 0.012）提前结束并记失败。
- **偏航角必须逐帧展开（unwrap）后累加**：Δyaw 用相邻帧差 `atan2(sin,cos)` 再累加，不能用首末位姿差——累计约 3.1 rad 时会在 ±π 处回绕成 −3.0，判据会误判（已实测踩坑）。
- 专家闭环使用特权状态（机器人位姿 + 桶位），**只作为数据标注者**；数据集只保存图像与文本，不保存位姿给模型（评测时也不给）。
- 桶位在世界系，机器人从原点出发；需要把桶位置换算为相对机器人的方位角 θ_err 与距离 d。

### 1.4 环境 API（供 eval 用）

`sim/env_vla.py` 提供：

```python
class VLAEnv:
    INSTRUCTIONS: list[str]          # 12 条文本，index = id
    obs_dim_image: (128,128,3)
    def reset(self, instr_id: int | None = None, seed: int | None = None) -> np.ndarray  # 返回图像
    def step(self, action_vw: np.ndarray) -> tuple[np.ndarray, bool, dict]  # (image, done, info)
    # info: {"dist": {0:..,1:..,2:..} 按桶色, "heading_err_by_color": {...}, "success": bool,
    #        "fall": bool, "pos": (x,y), "yaw": .., "vx": .., "yawrate": .., "instr_id": int, "t": int}
```

- `success_func(instr_id, info, history)` 放在同一文件，与 §1.3 判据一一对应。
- 同一个 `seed` 必须可复现同一场景（桶位/颜色/光照）。

## 2. 动作空间与归一化

- 物理量：`v ∈ [-0.35, 0.35] m/s`，`ω ∈ [-1.0, 1.0] rad/s`。
- 归一化（模型输出，tanh 后）：`a_norm = [v/0.35, ω/1.0] ∈ [-1,1]`。
- 数据集保存**未归一化物理量**，训练侧自行归一化；`stats.json` 记录 mean/std 供核对。
- 动作 chunk：第一版 chunk=1（10 Hz 单步）；若时序抖动明显再加 chunk=4（由模型负责人决定并写进 report）。

## 3. 数据集格式（`data/<dataset_name>/`）

```
data/dataset_v1/
  shards/shard_0000.npz   # 每个 shard = 50 个 episode
  index.jsonl             # 每个 episode 一行
  stats.json
  README.md               # 生成命令、规模、直方图/样例图路径
```

`shard_XXXX.npz` 内：
| 键 | dtype | shape | 说明 |
|---|---|---|---|
| images | uint8 | [N,128,128,3] | RGB，相机原图 |
| actions | float32 | [N,2] | (v, ω) 物理量 |
| instr_ids | int16 | [N] | 0–11 |
| episode_ids | int32 | [N] | 全局 episode 序号 |
| step_idx | int16 | [N] | 0–49 |
| scene_seeds | int32 | [N] | 场景种子（用于划分） |

`index.jsonl` 每行（顺序同 episode_ids 递增）：
```json
{"episode_id":0,"scene_seed":123,"instr_id":1,"instruction":"向前走","success":true,"T":50,"num_step":50,"bucket_colors":{"pos":["red","blue","green"]}}
```

`stats.json`：总帧数、各指令帧数、actions 的 mean/std/min/max、每个 shard 的文件名。

划分：`scene_seed % 10` → 0–7 训练、8–9 验证；测试集另用 `--split test` 生成（不同 seed 段，≥600 episodes 或 ≥30k 帧）。
**配对强制**：同一 `scene_seed` 下至少生成 2 个不同指令的 episode（占比 ≥40%），用于检验语言接地，`index.jsonl` 里用 `pair_group` 字段标注。

## 4. 模型接口（`vla/`、`eval/`，负责人：vla-model）

- 主干：`HuggingFaceTB/SmolVLM-256M-Instruct`（首次用 `assets/hf/` 本地缓存下载，代理见 §6）。
- 输入：128×128 RGB 图像 + 指令文本；prompt 模板：
  ```
  User:<image>指令：{instruction}\nAssistant:
  ```
- 输出：取最后 token（或自建 `<act>` 特殊 token）的 hidden state → 2 层 MLP → 2 维，`tanh` 后按 §2 反归一化。
- 训练：行为克隆 MSE（可加动作平滑/时序差分项），AdamW，bf16；8 GB 显存要求下可用 LoRA 或冻结视觉塔，训练脚本必须打印显存/耗时。
- 推理：`vla/infer.py` 提供 `VLAPolicy.load(ckpt).act(image, instruction) -> (v, ω)`，10 Hz 闭环下单步 < 100 ms（GPU）。
- 评测：`eval/closed_loop.py` 调用 `sim/env_vla.py`，输出 `eval/results/*.json`：
  - 8 条指令 × 50 局（instr 0–7 或全 12 条），每指令成功率、总成功率；
  - 视觉扰动（亮度 ×0.7/×1.3、随机 10% 遮挡块）下成功率；
  - 单步延迟 p50/p95；
  - 对比基线：`random`、`constant`（全输出 0）、以及“只看指令不看图”的 ablation（把图像替换为固定灰图）。
- 通过标准见 §7。

## 5. 语音输入（`deploy/asr/`，负责人：Lead）

- 本地 `faster-whisper`（`small` 或 `base`，中文），模型缓存 `assets/hf/whisper/`。
- 接口：`asr.transcribe(audio_16k_mono_float32) -> str`，再做指令归一化（同义词/数字/错字纠错）映射到 12 条指令文本；映射表 `deploy/asr/command_map.json`。
- 文本通道：同一 `command_map`，网页/命令行输入。

## 6. 环境与网络

- Python：`/home/czy/倒立摆/wheeltec_vla/.venv/bin/python`（已装 torch 2.9.1+cu128、transformers、mujoco、sb3、gymnasium、faster-whisper）。
- HuggingFace：**直连不通，必须** `export https_proxy=http://127.0.0.1:7890 http_proxy=http://127.0.0.1:7890`；GitHub 直连不通，依赖走 PyPI 清华源。
- 显存 8 GB（桌面占用约 3.7 GB），内存 14 GB。训练时 `num_workers ≤ 2`，避免把内存打满。
- 硬件：无相机、无上位机算力；实机阶段需外接 USB 摄像头（待采购）。

## 7. 验收标准（仿真）

1. 全 12 条指令、每条 50 局：总成功率 ≥ 80%，单条 ≥ 70%（常量类指令应 ≥ 95%）。
2. 视觉扰动：亮度 ×0.7 与 ×1.3 下成功率下降 ≤ 10 个百分点；10% 随机遮挡下降 ≤ 15 个百分点。
3. 单步推理延迟 p95 < 100 ms（GPU）。
4. 消融：固定灰图 + 同一指令，成功率显著低于用真图（证明视觉在使用）。
5. 纯 PPO 基线无法区分指令（对照组，仅说明任务不能只靠底层策略完成）。

## 8. 目录与写作用域

| 路径 | 负责人 | 说明 |
|---|---|---|
| `specs/` | Lead | 契约、决策记录 |
| `sim/` | sim-data | 场景、专家、数据生成、环境 API |
| `assets/lowlevel/` | Lead | 底层策略与模型（已复制，勿改） |
| `data/` | sim-data 写，vla-model 只读 | 数据集 |
| `vla/` | vla-model | 模型、训练、推理 |
| `eval/` | vla-model | 闭环评测、结果 JSON |
| `scripts/` | Lead | 一键脚本 |
| `deploy/` | Lead | 上位机桥、语音、固件协议 |
| `docs/`、`README.md` | Lead | 文档 |
| `runs/` | vla-model | 训练产物 |

## 9. 常用命令

```bash
cd /home/czy/倒立摆/wheeltec_vla
export https_proxy=http://127.0.0.1:7890 http_proxy=http://127.0.0.1:7890

# 相机/场景自检（输出 media/camera_check.png）
.venv/bin/python sim/smoke_render.py --out media/camera_check.png

# 小规模数据生成（30 回合，验收格式）
.venv/bin/python sim/gen_dataset.py --episodes 30 --out data/smoke

# 正式数据（2500 回合 ≈ 12.5 万帧）
.venv/bin/python sim/gen_dataset.py --episodes 2500 --out data/dataset_v1 --workers 1

# 行为克隆
.venv/bin/python vla/train_bc.py --data data/dataset_v1 --out runs/bc_v1 --epochs 6 --bs 16

# 闭环评测
.venv/bin/python eval/closed_loop.py --ckpt runs/bc_v1/best.pt --episodes 50 --instrs 0-11
```

## 10. 已知风险

- 底层速度跟踪不是理想的（指令 1.0 rad/s 实测约 0.6 rad/s；v6C 版甚至不能边走边转，已弃用），因此常量类指令的成功判据按 §1.3 的实际可达范围写死；闭环类指令（7–11）专家必须闭环。
- SmolVLM 的图像处理器可能把 128² 上采样到 512²，训练侧应设置较小的 `max_image_size`（256 左右）并记录实测显存；若 8 GB 放不下，用 LoRA + 冻结视觉塔。
- 内存 14 GB：数据生成用 `--workers 1`，训练 `num_workers 0–2`。
- 语音识别对“红桶/蓝桶”易错，必须在 `command_map` 里做模糊匹配 + 反馈确认。
- 实机相机未采购，实机阶段依赖采购进度；仿真验收不受影响。
