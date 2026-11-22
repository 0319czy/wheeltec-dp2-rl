# 本地语音识别（faster-whisper）验收报告

> 负责人：deploy-asr；日期：2026-11-22；范围：`deploy/asr/` + `assets/hf/whisper/`
> 结论：语音链路可用，**默认自动选 GPU small float16（p50 ≈ 0.12 s）**，无 GPU 时回退 **base CPU int8（p50 ≈ 0.78 s）**；
> 24 条 TTS 中文语音（18 条 1.0× + 6 条 0.75× 语速）→ ASR → 12 条指令，三种后端均 **24/24 = 100%**；
> 纯文本（口语/错字/数字/拒识）**26/26 = 100%**。

## 1. 模型

| 项 | small（默认 GPU 后端） | base（CPU 回退） |
|---|---|---|
| 来源 | ModelScope `Systran/faster-whisper-small` | ModelScope `Systran/faster-whisper-base` |
| 本地目录 | `assets/hf/whisper/faster-whisper-small/` | `assets/hf/whisper/faster-whisper-base/` |
| 总体积 | **464 MB**（`model.bin` 483,546,902 B） | **142 MB**（`model.bin` 145,217,532 B） |
| 运行时 | faster-whisper 1.2.1、ctranslate2 4.9.0、soundfile 0.14.0、numpy 2.5.5、Python 3.12 | 同左 |
| 设备/精度 | `cuda` / `float16`（回退时 `cpu` / `int8`） | `cpu` / `int8` |
| 解码 | `beam_size=5`、`language="zh"`、领域提示词开启 | `beam_size=5`、`language="zh"`、**无提示词 + `no_repeat_ngram_size=4`**（抑制短句重复） |
| 加载耗时 | 1.32 s（GPU，`local_files_only=True`） | 0.55 s |

下载（ModelScope 单文件接口，逐个文件下全）：

```bash
export https_proxy=http://127.0.0.1:7890 http_proxy=http://127.0.0.1:7890
for model in small base; do
  d=assets/hf/whisper/faster-whisper-$model; mkdir -p "$d"; cd "$d"
  for f in config.json tokenizer.json vocabulary.txt model.bin README.md configuration.json; do
    curl -sL --retry 3 "https://modelscope.cn/api/v1/models/Systran/faster-whisper-$model/repo?FilePath=$f&Revision=master" -o "$f"
  done
  cd -
done
```

## 2. 延迟与后端选择

测法：18 条 1.0×（1.87–2.42 s）+ 6 条 0.75× 语速（2.0–3.6 s）TTS 中文语音，逐条 `transcribe`，
统计端到端耗时（含解码、30 s padding 编码、decode）。测量时 load average ≈ 9（同期 sim/RL/eval 在跑，16 逻辑核）。

| 后端 | 设备/精度 | 1.0× p50 | 0.75× p50 | 正确率（24 条） | 显存占用 |
|---|---|---|---|---|---|
| **small GPU float16（默认）** | cuda / float16 | **117 ms**（90–153） | **128 ms**（104–164） | **24/24 = 100%** | 3.76 → 3.20 GiB（约 0.56 GiB） |
| small CPU int8（准确率优先） | cpu / int8 | 2 188 ms（1 989–2 615） | 2 567 ms（2 329–3 009） | 24/24 = 100% | — |
| base CPU int8（无 GPU 回退） | cpu / int8 | 779 ms（663–984） | 787 ms（707–817） | 24/24 = 100% | — |

自动选择逻辑（`asr.py: select_backend()`）：

1. `DP2_ASR_BACKEND` 环境变量或 `--backend` 强制指定；默认 `auto`。
2. `auto`：`torch.cuda.is_available()` 且**空闲显存 ≥ 1.5 GiB** → small `cuda/float16`；否则 → base `cpu/int8`；base 缺失则 small `cpu/int8`。
3. 每次运行只选一次并缓存；GPU 模型测完即释放（本次测试 `del model` + `empty_cache`，未与训练抢爆显存）。
4. 可用 `DP2_ASR_BACKEND=small_cpu|base_cpu|small_gpu` 覆盖，便于对比/降级。

> 注：若训练把 GPU 占到 6.5/8 GiB（本次评测后段实测），`auto` 会按阈值回退 `base_cpu`；
> 已实测回退路径 **24/24、p50 771 ms**。GPU small 的 0.56 GiB 占用是在空闲 3.76 GiB 时测得，不会 OOM。

说明：

- Whisper 编码器对任意长度输入都按 **30 s 窗口**计算，CPU 短句延迟基本固定 ≈2 s；GPU 上降到 ≈0.12 s。
- 之前 CPU-only 版本 18 条 p50 2 384 ms（当时 load 5.5–8.2），与本次 small CPU 2 188 ms 同量级。
- base 的原始文本常输出繁体（如「繞開紅桶前往藍桶」），`normalize.py` 已内置繁转简表，**归一化后仍 100%**；
  因此 base 的「ASR 原文一致率」只有 50%，但指令正确率是 100%。

## 3. 语音端到端评测（TTS 合成音频 → ASR → normalize）

音频：`deploy/asr/testaudio/`（18 条 1.0× wav，1.2 MB）+ `deploy/asr/testaudio_slow/`（6 条 0.75× 语速 wav，-25% 语速），
16 kHz mono，`manifest.json` 记录文本/音色/期望 id；edge-tts 四种 zh-CN 音色合成。

1.0× 的 18 条逐条结果（与上一版一致，三种后端都全对）：

| 语音 | ASR 文本（small GPU） | 归一化 | 期望 | 结果 |
|---|---|---|---|---|
| 向前走 | 向前走， | 1 向前走 | 1 | ✅ |
| 往前走一点 | 往前走一点， | 1 向前走 | 1 | ✅ |
| 向后退 | 向后退， | 2 向后退 | 2 | ✅ |
| 退一下 | 退一下， | 2 向后退 | 2 | ✅ |
| 向左转 | 向左转。 | 3 向左转 | 3 | ✅ |
| 向右转 | 向右转， | 4 向右转 | 4 | ✅ |
| 原地左转 | 原地左转， | 5 原地左旋 | 5 | ✅ |
| 原地右转 | 原地右转， | 6 原地右旋 | 6 | ✅ |
| 原地转圈 | 原地转圈， | 5 原地左旋（默认方向） | 5 | ✅ |
| 保持站立 | 保持站立。 | 0 保持站立 | 0 | ✅ |
| 面向红桶 | 面向红桶， | 7 面向红桶 | 7 | ✅ |
| 去找红桶 | 去找红桶。 | 8 靠近红桶 | 8 | ✅ |
| 靠近蓝桶 | 靠近蓝桶。 | 9 靠近蓝桶 | 9 | ✅ |
| 去找绿桶 | 去找绿桶。 | 10 靠近绿桶 | 10 | ✅ |
| 绕开红桶前往蓝桶 | 绕开红桶前往蓝桶。 | 11 绕开红桶前往蓝桶 | 11 | ✅ |
| 请帮我靠近红桶 | 请帮我靠近红桶。 | 8 靠近红桶 | 8 | ✅ |
| 向左转一圈 | 向左转一圈， | 5 原地左旋 | 5 | ✅ |
| 向前走（+25% 语速） | 向前走， | 1 向前走 | 1 | ✅ |

0.75× 语速的 6 条（向前走 / 向左转 / 原地转圈 / 靠近蓝桶 / 绕开红桶前往蓝桶 / 保持站立）：
三种后端归一化全部正确（6/6）；base 原文为繁体，经繁转简后仍命中。

- 语音端到端（1.0×+0.75×，auto/GPU）：**24/24 = 100%**，p50 112 ms（本次 eval 实测，min 88 / max 461）；
- 语音端到端（base CPU 回退）：**24/24 = 100%**，p50 696 ms。

## 4. 纯文本归一化评测（26 条）

覆盖：口语（往前走一点/退一下）、错字（红筒/蓝铜/录筒/右砖/往前揍/腿一下/保持战力/圆地转圈）、
繁体（繞開紅桶前往藍桶）、数字指令（第12条/指令3）、礼貌前缀（请帮我）、拒识（今天天气不错/打开音乐）。

- 正确率：**26/26 = 100.0%**
- 一句话里出现两个动作（如 ASR 幻觉追加「退下，向右转」）时，取**最先出现**的动作，避免被后段带偏。

## 5. 麦克风

| 项 | 值 |
|---|---|
| 设备 | MCHOSE V9 PRO USB Audio：`card 3` / `device 0` / `plughw:CARD=PRO,DEV=0` |
| 权限 | 直接访问 `/dev/snd` 成功（PulseAudio/PipeWire 默认用户可录） |
| 自检 | `arecord -D plughw:CARD=PRO,DEV=0 -f S16_LE -r 16000 -c 1 -d 1` → 16 000 点、1.0 s、RMS 0.0020、peak 0.014（环境底噪，正常） |
| 代码路径 | `asr.record_mic(duration)` 用 arecord 子进程 + soundfile 读取，返回 16 kHz float32；`--mic N` 直接可用 |

> 正确率评测用 **本地 TTS 合成音频**；麦克风链路单独做了 1 s 自检（设备/权限/采样率/电平正常），没有外放打扰环境。

## 6. 复现命令

```bash
cd /home/czy/倒立摆/wheeltec_vla

# 纯文本归一化（不加载模型）
.venv/bin/python deploy/asr/eval_asr.py --text-only

# 全量评测：24 条语音（含 0.75×）+ 26 条文本，自动选后端
.venv/bin/python deploy/asr/eval_asr.py --slow

# 指定后端对比（GPU small / CPU small / CPU base）
.venv/bin/python deploy/asr/eval_asr.py --slow --backend small_gpu
.venv/bin/python deploy/asr/eval_asr.py --slow --backend small_cpu
.venv/bin/python deploy/asr/eval_asr.py --slow --backend base_cpu --out deploy/asr/report_asr_base_cpu.json

# 单条 + 延迟
.venv/bin/python deploy/asr/asr.py --wav deploy/asr/testaudio/15_raokaihongqu_lan.wav --normalize --bench 3
.venv/bin/python deploy/asr/asr.py --backend base_cpu --wav deploy/asr/testaudio/15_raokaihongqu_lan.wav --bench 3

# 麦克风
.venv/bin/python deploy/asr/asr.py --mic 3 --normalize
.venv/bin/python deploy/asr/asr.py --list-mics
```

产物：`deploy/asr/report_asr.json`（auto）、`deploy/asr/report_asr_base_cpu.json`（base）。

## 7. 已知问题 / 注意事项

1. **faster-whisper 1.2.1 与 av 19.0.2 不兼容**：直接 `model.transcribe("x.mp3")` 会因
   `av.open(..., metadata_errors=...)` 报 TypeError。本实现改用 soundfile 解码（libsndfile 1.2.2，支持
   wav/mp3/flac/ogg）后传 float32 数组，已绕开；不要改用 av 解码路径。
2. GPU 后端需要约 0.6 GiB 显存；当前训练占用下空闲显存 <1.5 GiB 时 `auto` 会回退 base，不会 OOM。
3. base 后端默认关闭提示词并开 `no_repeat_ngram_size=4`，否则短句可能重复幻觉；small 后端保留领域提示词。
4. `原地转圈` 未给方向时默认映射到 **原地左旋(id=5)**，并在 `note` 里提示；如需右旋要说“原地右转”。
5. 归一化是规则驱动，12 条指令域内很稳；域外句子返回 `id=None`（bridge 会按“保持站立”处理）。
