## 六、VLA 指令执行（视觉 → 语言 → 动作）

给这台平衡车训了一个**自研小型 VLA**：车头相机看到红/蓝/绿三个桶，听懂 12 条中文指令，10 Hz 输出高层速度指令 (v, ω)，交给车上 100 Hz 平衡环执行。目录 [`vla/`](vla/)。

**为什么是分层**：STM32F103 跑不了视觉大模型，100 Hz 平衡也等不起网络推理。VLA 只在笔记本（RTX 5060）上做 10 Hz 的「看 + 听 → 速度」，平衡/限幅/看门狗仍在固件里——这是 VLA 在低算力小车上的标准落地形态。

**流程**：MuJoCo 场景（相机 + 三色桶 + 随机布局/光照）→ 特权状态专家自动标注 → 2500 回合 / 125,000 帧数据集 → SmolVLM-256M（LoRA r16 + 自研动作头）行为克隆 → 闭环评测 → 文本/语音（faster-whisper GPU 117 ms）→ 上位机 ASCII 协议 `#V250,0;` → 固件补丁（已编译/单测，未烧写）。

| 指标 | 结果 |
|---|---|
| 闭环成功率（12 指令 × 50 局，未见场景） | **71.8%（431/600）** |
| 常量类指令（站立/前后/转/旋） | 2/3/5/6 = 100%，已达底层+判据的物理上限 |
| 对象类指令（面向/靠近某色桶） | 26–62%，单帧视觉估距是瓶颈 |
| 视觉扰动（亮度 ×0.7/×1.3、10% 遮挡） | 无显著下降 |
| 灰图消融 | 对象类掉到 0%，证明视觉真的在用 |
| 推理延迟 | p50 49 ms / p95 51 ms（10 Hz 控制环要求 <100 ms） |
| 基线 | 随机 5.8%、恒零 11.7% |

**如实说清楚**：总体 71.8% 没到当初定的 80% 验收线，差距全部集中在「靠近某色桶」这类需要单帧估距的任务；分层架构、数据链路、模型、语音、协议、固件补丁都是可复现的真东西，失败原因（`not_reached` / `heading_too_large` / `fall`）逐回合记录在 `eval/results/*.json`。下一步：多帧历史输入、DAgger 修正分布漂移、或用深度/标记物补距离信息。

复现：

```bash
.venv/bin/python sim/gen_dataset.py --check-only --out data/dataset_v1   # 数据自检
.venv/bin/python sim/gen_dataset.py --episodes 2600 --min-success 2500 --out data/dataset_v1
.venv/bin/python vla/train_bc.py --data data/dataset_v1 --out runs/bc_v1 \
    --epochs 3 --bs 16 --image-size 256
.venv/bin/python eval/closed_loop.py --ckpt runs/bc_v1/best.pt --episodes 50 --instrs 0-11 \
    --scenarios base --seed-base 100002
```
