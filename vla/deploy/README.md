# deploy/：语音 + 上位机桥 + 固件协议

> 状态：**全部 mock 联调，不连真车、不烧写固件**。真实串口需要显式 `--serial --allow-real`
> 且 `DP2_ALLOW_REAL=1` 三重确认才会打开。

## 文件

| 文件 | 说明 |
|---|---|
| [`asr/asr.py`](asr/asr.py) | faster-whisper 本地中文识别；**自动选后端**（CUDA 空闲→small GPU float16，否则 base CPU int8）；16k float32 / wav / mp3 / 麦克风 → 文本 |
| [`asr/normalize.py`](asr/normalize.py) | 口语/同音错字/数字 → spec §1.3 的 12 条标准指令（返回 id + 置信度） |
| [`asr/command_map.json`](asr/command_map.json) | 12 条指令、同义词、颜色/桶/动词混淆表、模糊阈值 |
| [`asr/eval_asr.py`](asr/eval_asr.py) | 评测：18 条 1.0× + 6 条 0.75× TTS 语音 + 26 条纯文本（`--backend` 可切三个后端） |
| [`asr/REPORT.md`](asr/REPORT.md) | **验收报告**：模型体积、CPU 延迟 p50、识别/归一化正确率 |
| [`asr/testaudio/`](asr/testaudio/) | 18 条 16 kHz 1.0× 测试 wav + manifest（1.2 MB） |
| [`asr/testaudio_slow/`](asr/testaudio_slow/) | 6 条 0.75× 语速测试 wav + manifest |
| [`bridge.py`](bridge.py) | 相机帧 + 指令 → 假/真 VLA → `#V...;` / 单字符 → mock 车 / 串口；10 Hz 日志 |
| [`PROTOCOL.md`](PROTOCOL.md) | 上下行协议：单字符兼容、连续 (v, ω) 帧、`$...$` 遥测、安全互锁 |
| [`firmware-notes.md`](firmware-notes.md) | 固件只读分析 + 支持连续 (v, ω) 的最小改动方案（伪代码/风险/回退） |
| [`firmware-patch/`](firmware-patch/) | **可直接应用的 unified diff**（HAL_work 编译 SUCCESS / RTOS 备用）+ PC 单测 `test_parse_vw.c`（33 checks） |
| [`logs/`](logs/) | mock 联调日志、相机样张 |

## 快速开始

```bash
cd /home/czy/倒立摆/wheeltec_vla

# 1) 语音识别 + 归一化（麦克风 3 秒）
.venv/bin/python deploy/asr/asr.py --mic 3 --normalize

# 2) 上位机桥 mock：向前走 3 秒，打印每帧与遥测
.venv/bin/python deploy/bridge.py --mock --instruction "往前走一点" --duration 3

# 3) 闭环类指令（假策略按 mock 场景几何闭环）
.venv/bin/python deploy/bridge.py --mock --instr-id 8 --duration 3

# 4) 向下兼容单字符遥控
.venv/bin/python deploy/bridge.py --mock --instr-id 3 --proto legacy --duration 2

# 5) 语音直接驱动桥
.venv/bin/python deploy/bridge.py --mock --wav deploy/asr/testaudio/15_raokaihongqu_lan.wav --duration 2

# 6) 复跑 ASR 验收（24 条语音含 0.75× + 26 条文本；自动选后端）
.venv/bin/python deploy/asr/eval_asr.py --slow

# 7) 固件补丁单测 + 编译（不改仓库：在仓库副本上应用）
gcc -std=c99 -Wall -Wextra -I deploy/firmware-patch deploy/firmware-patch/test_parse_vw.c -lm -o /tmp/test_parse_vw && /tmp/test_parse_vw
```

`--proto` 三选一：`vw`（默认，`#V<mm_s>,<mrad_s>;`）、`param`（`#vc=;#wc=;`）、`legacy`（A/E/G/C/Z）。
真实链路：`--serial /dev/rfcomm0 --baud 9600 --allow-real`（另需环境变量）。

## 验收数字（详见 asr/REPORT.md 与 logs/bridge_mock.log）

- ASR：small 模型 464 MB / base 142 MB；**GPU small float16 p50 ≈ 0.12 s**、CPU small 2.19 s、CPU base 0.78 s；
- 24/24 TTS 中文语音（1.0× + 0.75× 语速，三种后端均 100%）→ 12 条指令；26/26 纯文本口语/错字/拒识（100%）；
- bridge mock：10.0–10.5 Hz（目标 10 Hz），3 s 前进 0.75 m，遥测 `{A...}$` 正常回读；
- 固件：现有单字符只能给固定档位（前进 ≈0.1675 m/s、转向 ≈0.705 rad/s）；`firmware-patch/` 提供连续 (v, ω) 补丁，
  在 `DP2_HAL_work` 上编译 SUCCESS（未打补丁产物 md5 与 `releases/dp2_n10_live.bin` 一致）。

## 安全

- 默认 mock，永不自动打开串口；`--serial` 未带 `--allow-real` + `DP2_ALLOW_REAL=1` 会直接拒绝。
- 指令未识别 → 保持站立；策略单帧超 300 ms → 发零速；退出/Ctrl-C → 发零速。
- 实机前先架空轮子，按 `firmware-notes.md` §8 / `firmware-patch/README.md` §6 的低速步骤验证；**本次没有烧写**。
