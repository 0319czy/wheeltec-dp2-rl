# sim/ — DP2 平衡车 VLA 仿真数据链路

契约：`specs/VLA-TASK-SPEC.md` §1–§3（唯一接口依据）。本目录只负责**仿真场景、底层闭环、专家标注、数据集生成**，不碰 `vla/`、`eval/`、`deploy/`、`specs/`。
运行环境：`/home/czy/倒立摆/wheeltec_vla/.venv/bin/python`；MuJoCo 渲染强制 **EGL**（本机 osmesa 不可用，`env_vla.py` 在 import mujoco 前设置 `MUJOCO_GL=egl`）。

```
sim/
  scene_vla.xml      一阶底盘 + 车头相机 + 3 桶 + 棋盘地面 + 4 面矮墙 + 天空渐变 + 2 盏方向光
  env_vla.py         VLAEnv（10 Hz API，128x128 RGB）、success_func、场景随机化、Δyaw 逐帧展开
  expert.py          12 条指令专家（特权状态标注者，不进数据集）
  gen_dataset.py     shards/index.jsonl/stats.json/README.md/samples.png + 自检断言
  smoke_render.py    media/camera_check.png 多回合拼图 + 图像/桶像素统计
```

## 常用命令

```bash
cd /home/czy/倒立摆/wheeltec_vla

# 相机/场景自检（6 回合拼图 + 统计，写 media/camera_check.png/.json）
.venv/bin/python sim/smoke_render.py --out media/camera_check.png --rounds 6

# 小规模数据（30 次尝试 → data/smoke，约 11–17 s；失败尝试会被丢弃）
.venv/bin/python sim/gen_dataset.py --episodes 30 --out data/smoke

# 正式数据（2600 次尝试，保证 >=2500 条成功回合约 12.5 万帧，约 17 min，约 4.1 GB）
.venv/bin/python sim/gen_dataset.py --episodes 2600 --out data/dataset_v1 --workers 1 --min-success 2500

# 测试集（不同 seed 段：100002 起，spec §3；保证 >=600 条成功回合 / 30k 帧）
.venv/bin/python sim/gen_dataset.py --episodes 700 --out data/dataset_test --split test --workers 1 --min-success 600

# 对象类指令加量（只生成 7–11；obstacle 场景比例约 46%，instr11 占 30%）
.venv/bin/python sim/gen_dataset.py --episodes 1400 --min-success 1200 --split trainval \
    --instrs "7,8,9,10,11" --instr-weights "1,2,2,2,3" --out data/dataset_obj --workers 1

# 单指令补集：只生成指令 7（面向红桶），每个场景唯一 seed
.venv/bin/python sim/gen_dataset.py --episodes 260 --min-success 220 --instrs "7" --seed-base 300000 \
    --out data/dataset_obj7 --workers 1

# 消融：关闭桶碰撞体（视觉桶，机器人可穿过）
.venv/bin/python sim/gen_dataset.py --episodes 60 --out /tmp/smoke_nocollide --no-collide

# 额外：把首个成功回合录成 GIF（demo 素材，约 50 帧）
.venv/bin/python sim/gen_dataset.py --episodes 30 --out /tmp/smoke_gif --save-gif media/expert_demo.gif
```

## 环境 API（spec §1.4）

```python
from sim.env_vla import VLAEnv, success_func, INSTRUCTIONS
env = VLAEnv()                       # 单进程；EGL 渲染 128x128
img = env.reset(instr_id=8, seed=123)  # -> uint8 (128,128,3)
while True:
    img, done, info = env.step(action)  # action = (v[m/s], w[rad/s])，内部 clip 到 [-0.35,0.35]x[-1,1]
    if done:
        break
```

- 时间基准：物理 5 ms；底层 PPO 10 ms（100 Hz）；VLA 100 ms（10 Hz）；1 回合 = 50 VLA step = 5 s。
- 底层：`assets/lowlevel/balance_model.zip` + `assets/lowlevel/vecnormalize.pkl`（SB3 PPO，9 维 obs）。
  **必须**用 VecNormalize 的 `normalize_obs`（`training=False`）后再 predict，否则策略输出恒 0（已踩坑）。
- `info` 字段：`dist`、`heading_err_by_color`（**按颜色 id 0=红/1=蓝/2=绿** 索引）、`success`、`fall`、`pos`、`yaw`、`vx`、`yawrate`、`instr_id`、`t`；
  额外提供 `pitch`、`z`、`cum_dyaw`、`start_pos`、`start_yaw`、`scene_seed`、`scene_mode`、`bucket_positions`、`bucket_colors`。
- `cum_dyaw` 用**逐帧 `atan2(sin,cos)` 差值累加**，不是首末位姿差：原地左旋 5 s 时累计约 +3.3 rad，但末位姿 yaw 已回绕到 −3.09，用首末差会把“左旋”误判成“右旋”。
- `success_func(instr_id, info, history)` 与 spec §1.3 判据一一对应；`info["success"]` 每步都算（含义是“若此刻结束是否成功”），评测只看 done 时那一步。
- 同一 `seed` → 同一场景（桶位/颜色/光照/起点）**且同一动作序列 → 同一图像轨迹**（已测，逐位一致）。
- **图像噪声约定**：`reset()` 与 `step()` 返回的图像**已经包含** spec §1.2 的相机噪声（σ=2/255，由 seed 确定性生成）；数据集、demo、评测都消费同一张带噪图。`render_rgb()` 返回无噪声原图（仅用于 `smoke_render.py` 统计与可视化）。
- 辅助：`env.render_rgb()`（无噪声原图）、`env.bucket_pixel_counts()`（分割渲染统计各色桶像素）、`env.scene_info()`、`env.close()`；构造参数 `collide=True/False` 控制桶碰撞体（默认 True，False 为消融）。
- 布局控制：`VLAEnv(obstacle_mod=6, obstacle_phase=5)`（默认 = 旧规则 `seed % 6 == 5`，dataset_v1/test 依赖它）；对象加量集用 `obstacle_mod=3, obstacle_phase=0`，`obstacle_mod=1` 即 `--obstacle-only`（全部 obstacle 布局）。同一 seed 仍确定性复现同一场景。

## 场景与随机化（spec §1.2 更新版）

- 8 m × 8 m 棋盘地面（`builtin="checker"`，`texrepeat=3`，格子约 0.17 m），4 面 0.3 m 矮墙 + 天空渐变作视觉参照；矮墙在 4 m 外，5 s 内不可达，不影响判据。
- 相机挂 `chassis`：`pos="0.06 0 0.145"`（离地约 0.18 m）、`fovy=70`、俯 15°（`xyaxes="0 -1 0 0.259 0 0.966"`，MuJoCo 相机看自身 −z，已出图核对）。
- 3 个桶（r=0.07 m，h=0.22 m），极坐标以**起点**为原点：半径 0.65–1.4 m、方位角 **±30°**、相互间距 ≥0.45 m；颜色红/蓝/绿。
- 每回合随机：桶位/颜色排列（generic 场景）、起点 (x,y)±0.1 m + yaw±0.2 rad、光照 0.7–1.3、地面亮度 0.85–1.15、相机噪声 σ=2/255（由 seed 确定性生成）。
- 两种布局由 seed 决定（`seed % 6 == 5` → obstacle，其余 generic）：
  - generic：3 个随机位置 + 随机颜色排列（“同一位置可能是红也可能是蓝”）。
  - obstacle（给指令 11 用）：蓝桶随机在 ±30°；红桶放在机器人→蓝桶连线附近，距机器人 0.7–1.0 m、横向偏移 0.12–0.25 m；绿桶随机。**横向偏移下限取 0.12 而非 0**，否则红桶会完全挡住蓝桶，模型在开局看不到目标、标签不可学。
  - `gen_dataset.py` 把指令 0 与指令 11 配成一对，且该对落在 obstacle seed 上，因此“每个场景可承载任意指令、配对 100%”。
- **桶是真实碰撞体**（默认 `contype/conaffinity=1`、`condim=3`）：“靠近/绕行”时撞桶会顶住小车甚至翻倒，这是任务真实性的一部分。
  因此 `gen_dataset.py` **只把专家成功回合写进 shards**：撞桶/倒地/判据失败的尝试计入 `stats.json` 的 `attempted_episodes` / `discarded_episodes` 后丢弃（不进入 BC 集）。`--no-collide` 是消融开关（视觉桶，可穿过）。

## 专家（`expert.py`，只做标注）

| id | 文本 | 实现 |
|---|---|---|
| 0 | 保持站立 | (0, 0) |
| 1 | 向前走 | (0.25, 0)；另加“前方走廊避桶”转向（切向 clearance 0.28 m，避免撞上正前方桶） |
| 2 | 向后退 | (−0.20, 0) |
| 3 | 向左转 | (0.25, +0.60) |
| 4 | 向右转 | (0.25, −0.60) |
| 5 | 原地左旋 | (0, +1.0) |
| 6 | 原地右旋 | (0, −1.0) |
| 7 | 面向红桶 | v=0，ω=clip(1.5·θ_red, ±1) |
| 8/9/10 | 靠近红/蓝/绿桶 | v=0.32 (d>0.95) 否则 clip(1.0·(d−0.40),0,0.32)；ω 朝目标，同时避让另两个桶（切向 clearance 0.35 m） |
| 11 | 绕开红桶前往蓝桶 | 红桶按 `asin(0.55/d_red)` 切向绕行、绿桶按 0.35 m 避让，两者取最近挡路者，绕过即转回蓝桶 |

- 指令 3/4 用 **(v=0.25, |ω|=0.6)** 而不是 spec 表的 (0.15, ±0.8)：实测当前底层在 (0.15, ±0.8) 下左转前向位移仅 0.04 m（判据要求 >0.15 m），10 个 seed 只有约 10% 通过；换成 (0.25, ±0.6) 后左右转 10/10 通过（Δyaw +1.82 / −1.46 rad，前向 +0.30 / +0.65 m）。`Expert(tuned_turns=False)` 可恢复 spec 原常量。
- 恢复桶碰撞后，原始“只朝目标开”的专家会撞非目标桶（指令 8/9/10 实测 60–90%），因此 1、8、9、10、11 都加了**切向绕桶**项：在同一场景分布下，指令 8/9/10 回到 10/10，指令 11 8/10（两次失败都是红/绿桶把去蓝桶的路夹住、5 s 内没走到 d_blue<0.5）。
- 指令 11 的原始“边靠近边受斥力”写法实测不可用（红桶 min d 0.23–0.34 或绕太远到不了蓝桶）；改为切向绕行后 min d_red 0.42–0.52 m、终态 d_blue 0.28–0.44 m。

## 数据集清单

| 数据集 | 规模 | split | 用途 | 注意事项 |
|---|---|---|---|---|
| `data/dataset_v1` | **2500 回合 / 125,000 帧 / 50 shard / 3.9 GB** | trainval（scene_seed %10 → 0–7 训练、8–9 验证） | **主训练集**（12 条指令） | 只含专家成功回合；`--episodes` 是尝试数，2600 次尝试后保留 2500 条 |
| `data/dataset_test` | **657 回合 / 32,850 帧 / 14 shard / 1.1 GB** | test（seed 100002 起，与 trainval 不相交） | **只用于评测**，禁止参与训练/调参 | 满足 spec “≥600 回合或≥30k 帧”；每 50 回合/shard |
| `data/dataset_obj` | **1211 回合 / 60,550 帧 / 25 shard / 2.0 GB** | trainval（seed 200001–201470） | 对象类指令 **7–11 加量**（可与 dataset_v1 混采/过采样） | 补集只含 7–11；官方 `ds_validate` 会报 0–6 缺帧两条（预期）；instr11 全 obstacle 场景（占场景 34.7%） |
| `data/dataset_obj7` | **260 回合 / 13,000 帧 / 6 shard / 416 MB** | trainval（seed 300001–300778） | 单补 **指令 7“面向红桶”**（v3 备用） | 单一指令集、无“不同指令配对”、`paired_episode_rate=0`；场景 260/260 唯一、零丢弃 |
| `data/smoke` | **28 回合 / 1,400 帧 / 1 shard / 45 MB** | trainval | **冒烟/流水线联调**（只验证格式与链路） | 30 次尝试只保留 28 条成功回合；官方 `ds_validate` 仅 “每指令<200帧” 一条 FAIL 属预期，不能当正式训练集 |
| `data/dataset_v1_aux` | **2500 回合 / 125,000 帧 / 50 shard / 3.9 GB（+2.7 MB）** | trainval | aux 版主训练集（可与/替代 dataset_v1 使用） | 仅比 dataset_v1 多 `aux` [N,6] 键；index/场景 seed/图像/动作**逐条一致**（已核验） |
| `data/dataset_obj_aux` | **1211 回合 / 60,550 帧 / 25 shard / 2.0 GB（+1.3 MB）** | trainval | aux 版对象类加量集（指令 7–11） | 仅比 dataset_obj 多 `aux` 键；index/场景 seed/图像/动作逐条一致 |
| `data/aux_smoke` | **12 回合 / 600 帧 / 1 shard / 约 29 MB** | trainval | `--aux` 辅助标签的**格式/数值验证集**（不用于训练） | 每帧多一个 `aux` float32 [N,6]；详见下方 aux 说明 |

生成命令（均在本项目根目录、`.venv/bin/python` 下运行；省略的参数见 `sim/gen_dataset.py --help`）：

```bash
# dataset_v1（主训练集）
.venv/bin/python sim/gen_dataset.py --episodes 2600 --out data/dataset_v1 --workers 1 --min-success 2500

# dataset_test（测试集，seed 段与 trainval 不相交）
.venv/bin/python sim/gen_dataset.py --episodes 700 --out data/dataset_test --split test --workers 1 --min-success 600

# dataset_obj（指令 7–11 加量，权重 1:2:2:2:3）
.venv/bin/python sim/gen_dataset.py --episodes 1400 --min-success 1200 --split trainval \
    --instrs "7,8,9,10,11" --instr-weights "1,2,2,2,3" --out data/dataset_obj --workers 1

# dataset_obj7（只补指令 7，独立 seed 段）
.venv/bin/python sim/gen_dataset.py --episodes 260 --min-success 220 --instrs "7" \
    --seed-base 300000 --out data/dataset_obj7 --workers 1

# smoke（只做格式冒烟）
.venv/bin/python sim/gen_dataset.py --episodes 30 --out data/smoke --workers 1
```

新增参数示例（`--instrs / --instr-weights / --obstacle-mod / --obstacle-phase / --obstacle-only / --seed-base`）：

```bash
# 1) 对象类过采样：只生成 7–11，并指定权重
.venv/bin/python sim/gen_dataset.py --episodes 1400 --instrs "7,8,9,10,11" \
    --instr-weights "3,2,2,2,4" --out /tmp/ds_obj_v2 --workers 1

# 2) 单指令补集 + 独立 seed 段（避免与其它补集重复场景）
.venv/bin/python sim/gen_dataset.py --episodes 260 --instrs "7" --seed-base 300000 \
    --out /tmp/ds_obj7 --workers 1

# 3) 强制 obstacle 布局 / 自定义 obstacle 场景比例（seed % mod == phase 即 obstacle）
.venv/bin/python sim/gen_dataset.py --episodes 60 --obstacle-only --out /tmp/ds_obst_only --workers 1
.venv/bin/python sim/gen_dataset.py --episodes 60 --obstacle-mod 4 --obstacle-phase 0 \
    --out /tmp/ds_obst25 --workers 1
```

> `--obstacle-only` 等价于 `obstacle_mod=1`（所有场景都用“红桶在机器人→蓝桶连线旁”的布局）；
> 默认 legacy 规则是 `seed % 6 == 5`（dataset_v1/test 依赖它，不要改默认）；对象补集默认 `seed % 3 == 0`。

**`--aux` 辅助标签（备选方案，默认关闭）**

```bash
# 生成带 aux 的验证小集（只有 --aux 时才写 aux 键，默认完全向后兼容）
.venv/bin/python sim/gen_dataset.py --episodes 12 --out data/aux_smoke --aux --workers 1
```

- shard 里多一个 `aux`：**float32 [N,6]**，顺序 `[dx_red, dy_red, dx_blue, dy_blue, dx_green, dy_green]`，单位米；
  `dx` = 车头 +x 方向距离，`dy` = 车体左方 +y 方向距离，数值 clip 到 ±3.0。用拍摄该帧时 `info["pos"]/["yaw"]` 与三色桶世界坐标换算（`dx = cos(yaw)wx + sin(yaw)wy`，`dy = -sin(yaw)wx + cos(yaw)wy`）。
- `stats.json` 记 `aux: true`、`aux_order`、`aux_min`、`aux_max`；`--check-only` 若 shard 存在 `aux` 键会校验 dtype/shape/有限性/±3.0 范围并与 stats 的 min/max 对齐。
- 验证（`data/aux_smoke`，12 回合/600 帧）：用同 seed+动作回放取分割渲染的桶质心，与 aux 预测的像素横坐标对照，红桶可见的 368 帧 **中位误差 0.4 px、p90 1.1 px、左右侧一致性 368/368 = 100%**；边缘被裁的少数帧误差大属正常（桶在画面外）。
- 两组正式 aux 集（2026-11-22 20:30，与 512 训练并行）：`dataset_v1_aux`（2500 回合/125k 帧/3.9 GB/1153.6 s/108.4 fps）与 `dataset_obj_aux`（1211 回合/60.5k 帧/2.0 GB/578.6 s/104.7 fps），aux min/max 见各自 `stats.json`；与原集逐条对比：`episode_id/scene_seed/instr_id/T/success/pair_group/scene_mode` **0 处不一致**，shard 0 的 images/actions 字节级一致（只多 aux 键）。

## 数据集

`data/smoke` 为冒烟集（30 次尝试 → 28 条成功回合 / 1400 帧，2 次撞桶失败被丢弃），正式集为
`data/dataset_v1`（trainval，≥2500 成功回合）与 `data/dataset_test`（test，≥600 成功回合）。
目录结构：

```
data/<name>/
  shards/shard_XXXX.npz   每 50 个成功回合一个 shard：
                          images uint8 [N,128,128,3]  actions float32 [N,2]
                          instr_ids int16  episode_ids int32  step_idx int16  scene_seeds int32
  index.jsonl             每成功回合一行（episode_id 递增，含 pair_group）
  stats.json              attempts/kept/discarded、帧数、各指令帧数、动作 mean-std-min-max、shard 列表
  README.md               生成命令 + 每指令条数 + 丢弃率 + 样例图路径
  samples.png             12 条指令各一张开局样例
```

冒烟集 `data/smoke` 跑 `scripts/ds_validate.py` 会只在 “min frames per instr < 200” 一条失败
（每指令 1–3 回合 = 50–150 帧，这是 30 次尝试冒烟集的预期；正式集每指令数千帧可过）。

**正式集实测（2026-11-22，桶碰撞开启、只保留成功回合）**：

| 数据集 | split | 尝试 | 保留 | 丢弃 | 帧数 | shard | 体积 | 生成耗时 | 渲染 fps | 配对率 | ds_validate |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `data/dataset_v1` | trainval | 2689 | **2500** | 189（7.0%） | **125,000** | 50 | **3.9 GB** | 1677.8 s（机器同时跑 BC 训练，独立时约 16–18 min） | 74.5 | 93.4% | **PASSED** |
| `data/dataset_test` | test | 700 | **657** | 43（6.1%） | **32,850** | 14 | **1.1 GB** | 321.8 s | 102.1 | 94.1% | **PASSED** |
| `data/smoke` | trainval | 30 | 28 | 2（6.7%） | 1,400 | 1 | 45 MB | 13.4 s | 104.6 | 92.9% | 仅 “每指令<200帧” 一条 FAIL（冒烟集预期） |
| `data/dataset_obj` | trainval（仅 7–11） | 1400 | **1211** | 189（13.5%） | **60,550** | 25 | **2.0 GB** | 568.2 s | 106.6 | 76.1% | 仅 0–6 覆盖两条 FAIL（补集预期，见下） |
| `data/dataset_obj7` | trainval（仅 7） | 260 | **260** | 0（0%） | **13,000** | 6 | **416 MB** | 127.4 s | 102.1 | 0%（单指令无配对，补集） | 仅“缺其余 11 条指令”两条 FAIL（补集预期） |

- dataset_v1 每指令帧数：0:9500、1:10850、2:11200、3:10950、4:10000、5:11200、6:11200、7:11200、8:10800、9:10450、10:10800、11:6850（指令 11 因红/绿桶夹路，丢弃率约 39%，仍 137 回合/6850 帧）。
- dataset_test 每指令帧数：0:2550、1:2950、2:2950、3:2900、4:2550、5:2900、6:2900、7:2900、8:2850、9:2750、10:2850、11:1800。
- 动作统计（dataset_v1）：mean [0.104, 0.005]、std [0.161, 0.562]，v∈[−0.20,0.32]、ω∈[−1.0,1.0]；图像均值 126.2（min 2，max 255）。
- 训练/验证划分：`scene_seed % 10` → 0–7 训练、8–9 验证；test 的 seed 从 100002 起，与 trainval 不相交。
- 两个正式集都用项目自带 `scripts/ds_validate.py` 独立校验通过；`gen_dataset.py --check-only` 提供流式自检（逐 shard 读取，不再一次性载入 6 GB 图像：早期版本全量 `np.concatenate` 在 2500 回合时会把内存打满）。

**对象类加量集 `data/dataset_obj`（本轮新增，补 ep1 对象类闭环偏低）**

- 命令：`--episodes 1400 --min-success 1200 --split trainval --instrs "7,8,9,10,11" --instr-weights "1,2,2,2,3"`。
- 调度单元（2 个权重循环 = 20 次尝试）：**6 个 obstacle 场景各 1 条 instr11** + **7 个 generic 场景各 2 条不同的 7–10 指令**；即 instr11 占 30% 尝试、obstacle 场景占 46%、配对率约 70%。
- instr11 的 scene_seed 全部落在 obstacle 池（200001+3k，`seed%3==0`），每个 seed 唯一、不重复；其他指令在 generic 池（200001+3k+1）。
- 输出格式与 dataset_v1 完全一致（shards/index.jsonl/stats.json/README/samples.png），`stats.json` 额外记录 `instrs`/`instr_weights`/`obstacle_mod`/`obstacle_phase`。
- 实测（2026-11-22 15:45）：1400 次尝试 → **1211 成功回合 / 60,550 帧 / 25 shard / 2.0 GB**，丢弃 189（13.5%），耗时 568.2 s（0.41 s/尝试，106.6 fps），配对率 76.1%，obstacle 场景 260/750 = **34.7%**。
- 每指令：7: 140 回合/7,000 帧、8: 271/13,550、9: 273/13,650、10: 267/13,350、11: **260/13,000**；instr11 共 420 次尝试、丢弃 160（38%），成功 260 条、260 个唯一 obstacle seed（全部 `scene_mode=obstacle`）。
- 与 dataset_v1 对比，instr11 从 137 回合/6,850 帧翻倍到 260 回合/13,000 帧；instr8/9/10 约 1.2–1.25×；instr7 因权重最低只有 140 回合（比 v1 的 224 少）——ep1 中“面向红桶”30% 若仍不达标，建议下一轮单独给 7 提权重。
- 注意：该集只有 7–11，`scripts/ds_validate.py` 的 12 指令覆盖检查会报 `some instruction has zero frames` + `min frames per instr 0 < 200` 两条（预期；它是补集，需与 dataset_v1 合并训练）。其余格式/动作范围/索引/统计检查均通过；`gen_dataset.py --check-only` 流式自检通过（覆盖集 [7,8,9,10,11]）。

**单指令补集 `data/dataset_obj7`（专补指令 7“面向红桶”）**

- 命令：`--episodes 260 --min-success 220 --instrs "7" --seed-base 300000`（其余参数同 dataset_obj；`--seed-base` 与 dataset_obj 的 200001 段完全不相交）。
- 实测：**260 次尝试 → 260 成功回合（0 丢弃）/ 13,000 帧 / 6 shard / 416 MB**，127.4 s（0.49 s/回合，102.1 fps）。
- 每回合一个唯一 generic 场景（seed 300001–300778，`seed%10` 仍按 0–7 train / 8–9 val 划分），无重复场景；动作 v=0、ω∈[−0.783, 0.785]；图像均值 126.5。
- 单指令集没有“不同指令配对”，`paired_episode_rate=0`；`gen_dataset.py --check-only` 的配对断言只在 ≥2 条指令时生效。官方 ds_validate 只报“缺其余 11 条指令”两条（补集预期）。

- **只写成功回合**：`--episodes` 是尝试次数；失败尝试（撞桶/倒地/判据不过）丢弃，`--min-success` 可强制继续尝试直到成功回合数达标。`stats.json` 含 `attempted_episodes`/`discarded_episodes`/`discard_rate`/`attempt_success_rate` 与每指令 `attempts_per_instr`/`discards_per_instr`。
- 每 shard 50 个成功回合（`--shard-size`），`gen_dataset.py` 结束前会从磁盘重新加载全部 shard 并断言：dtype/shape、动作范围、12 条指令覆盖、index 行数 = 回合数、每回合 step_idx 连续且帧数 = T、配对率 ≥40%、stats 字段齐全；通过后打印 `SELF-CHECK PASSED`。`--save-gif PATH` 额外记录首个成功回合 GIF。
- 场景 seed 与指令配对：`scene_seed = base + 尝试序号//2`，pair 表 `[(1,2),(3,4),(5,6),(7,8),(9,10),(0,11)]`；每场 2 个不同指令，丢弃失败回合并重新编号后实测配对率 dataset_v1 93.4%、test 94.1%（spec 要求 ≥40%）。训练/验证按 `scene_seed % 10`（0–7 训练、8–9 验证）切；测试集 `--split test` 从 seed 100002 起。
- 注意 `actions` 存**物理量**（v, ω），训练侧自行归一化；`stats.json` 的 mean/std 仅供核对。

## 实测数据（本机 RTX 5060 Laptop 8 GB，单进程，2026-11-22）

**相机自检**（`media/camera_check.png`，6 个 seed 含 1 个 obstacle 场景，统计见 `media/camera_check.json`）：

| seed | 布局 | 图像均值 | p5/p95 | 暗(<20) | 亮(>200) | 三色桶像素（占比） |
|---|---|---|---|---|---|---|
| 0 | generic | 118.5 | 82/178 | 0 | 0 | red 206 / blue 686 / green 244（6.93%） |
| 1 | generic | 131.3 | 83/199 | 0 | 3% | red 244 / blue 312 / green 616（7.15%） |
| 2 | generic | 124.0 | 66/178 | 0 | 0 | red 732 / blue 325 / green 179（7.54%） |
| 3 | generic | 126.2 | 63/178 | 0 | 0 | red 867 / blue 239 / green 604（10.44%） |
| 4 | generic | 124.8 | 85/178 | 0 | 0 | red 223 / blue 255 / green 269（4.56%） |
| 5 | obstacle | 125.1 | 77/178 | 0 | 0 | red 576 / blue 267 / green 395（7.56%） |

- 图像均值 **118.5–131.3**（通过线 80–160）；每个回合三色桶都落在画面内、像素占比 **4.56%–10.44%**（747–1710 px/16384），肉眼可清楚分辨红/蓝/绿圆柱与地面棋盘。
- 早期版本踩坑：棋盘 `texrepeat=10`（格子 0.05 m）+ 阴影贴图阴影 acne 会让地面前景出现严重摩尔纹，均值也不稳；最终用 `texrepeat=3` + 关闭 castshadow 解决。

**数据生成性能**（单进程 EGL，本机 RTX 5060 Laptop，含压缩 npz 写盘）：

| 指标 | 数值 |
|---|---|
| 每成功回合耗时 | **0.38–0.47 s**（独立跑；与 BC 训练并行时 0.62–0.70 s） |
| 渲染吞吐 | **102–133 fps**（dataset_v1 并行负载下 74.5 fps，test 102.1 fps） |
| 尝试吞吐 | **1.6–2.7 尝试/s** |
| dataset_v1 规模 | 2689 次尝试 → **2500 成功回合 / 125,000 帧 / 3.9 GB**，1677.8 s（并行负载下） |
| dataset_test 规模 | 700 次尝试 → **657 成功回合 / 32,850 帧 / 1.1 GB**，321.8 s |
| 专家成功率（120 回合加测，恢复碰撞后：每指令 10 seed，指令 11 用 10 个 obstacle seed） | **116/120 = 96.7%**；失败分布：指令 0 1 次（漂移 0.104 m）、指令 4 1 次、指令 11 2 次（红/绿桶夹住去蓝桶路线） |
| 数据生成实测丢弃率 | dataset_v1 **7.0%**（指令 11 约 39%、指令 0 约 15%、指令 4 约 11%）；dataset_test 6.1%；smoke 6.7% |

**底层策略实测**（当前 `assets/lowlevel` = `fix_mixed-v0.3-w1.2` 那一对，Lead 2026-11-22 决定）：

| 指令常量 | 实测 5 s 结果 |
|---|---|
| (0,0) | 位移 0.01–0.10 m，Δyaw ≤0.15 rad（站立判据 95–100% 通过） |
| (0.25,0) | 前向 0.78–1.08 m |
| (−0.20,0) | 后向 0.57–0.74 m |
| (0,±1.0) | Δyaw ±3.1–3.4 rad，位移 0.11–0.23 m |
| (0.15,±0.8) | 左转 Δyaw +3.06 但前向仅 +0.04；右转 −2.03 / +0.20（左转不过判据） |
| (0.25,±0.6) | 左转 +1.82 / +0.30；右转 −1.46 / +0.65（采用） |

## 已知限制 / 需要 spec 注意的点

1. **专家常量偏离**：指令 3/4 用 (0.25, ±0.6)，不是 spec 表的 (0.15, ±0.8)；原因见上（当前底层边走边转的航向/速度耦合）。只改专家、不改判据。
2. **碰撞失败会被丢弃**：120 回合小样本专家成功率 96.7%，但生成器实测丢弃率 7.0%（大样本/更多场景），因此 `--min-success` 自动把尝试数从 2600 扩到 2689；代价是正式集各指令条数略有不均（指令 11 只有 137 回合）。撞桶本身不是判据，只影响数据可用性。
3. **obstacle/generic 两种布局**：指令 11 只在 obstacle seed 上出现（`seed % 6 == 5`），因此指令 11 的视觉分布与其余指令略有差异；这是“红桶必须挡在去蓝桶路上”和“同 seed 同场景 + 配对”两条约束的直接结果。generic 场景保留完整颜色随机排列；obstacle 场景红/蓝位置由角色决定（红在机器人→蓝连线旁）。
4. **obstacle 场景红桶横向偏移取 0.12–0.25 m**（spec 写 0–0.25）：偏移 <0.12 时红桶会把蓝桶完全遮住，目标不可观测、标签不可学。只报告，不改 specs/。
5. **图像噪声**写进数据（σ=2），压缩后约 33 KB/帧；2500 成功回合约 4.1 GB。若磁盘/内存紧张可调低 `noise_sigma` 或改存无噪图（当前按 spec 保留噪声）。
6. 场景默认种子上限 1.4 m；矮墙在 4 m 外只作视觉参照；相机 FOV 70° 下 ±30° 的桶在画面边缘可能被裁掉极小一列像素（中心可见，分割统计 >0）。
7. `env_vla.py` 构造时加载 PPO + VecNormalize 约 0.9 s；`smoke_render.py` 用 `load_policy=False` 不需要策略。`gen_dataset.py` 的自检已改成**逐 shard 流式**（`--check-only` 可单独复跑），避免 2500 回合全量载入约 6 GB 图像。
8. 当前只验证了单进程 EGL；`--workers` 参数仅为兼容 spec 命令保留，多进程渲染未实现（内存 14 GB 建议保持单进程）。
