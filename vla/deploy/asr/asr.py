"""本地中文语音识别（faster-whisper），自动选择后端。

契约（specs/VLA-TASK-SPEC.md §5）：
    transcribe(audio_16k_mono_float32) -> str

后端自动选择（可用环境变量 DP2_ASR_BACKEND 强制）：
    auto（默认）: CUDA 可用且空闲显存 >= 1.5 GiB -> small float16；否则 base int8
    small_gpu   : assets/hf/whisper/faster-whisper-small，cuda float16
    small_cpu   : small，cpu int8（准确率优先、慢）
    base_cpu    : assets/hf/whisper/faster-whisper-base，cpu int8 + ngram 抑制重复

实测（2026-11-22，24 条中文短句）：small GPU p50 117 ms / small CPU 2188 ms /
base CPU 779 ms，三者归一化正确率均 100%；详见 deploy/asr/REPORT.md。

同时支持：
    - wav / mp3 / flac 文件路径（用 soundfile 解码，绕开 faster-whisper 1.2.1
      与 av 19 的 metadata_errors 参数不兼容问题）；
    - (samples, sample_rate) 元组，自动重采样到 16 kHz；
    - 麦克风录音（arecord 子进程，默认 MCHOSE V9 PRO = plughw:CARD=PRO,DEV=0）。

命令行：
    python deploy/asr/asr.py --wav media/voice.wav --normalize
    python deploy/asr/asr.py --mic 3 --normalize
    python deploy/asr/asr.py --list-mics
    python deploy/asr/asr.py --backend base_cpu --wav media/voice.wav --bench 3
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
WHISPER_ROOT = ROOT / "assets" / "hf" / "whisper"
DEFAULT_MODEL_DIR = WHISPER_ROOT / "faster-whisper-small"
BASE_MODEL_DIR = WHISPER_ROOT / "faster-whisper-base"
SAMPLE_RATE = 16000
DEFAULT_MIC = os.environ.get("DP2_MIC", "plughw:CARD=PRO,DEV=0")
DOMAIN_PROMPT = "向前走，向后退，向左转，向右转，原地左旋，原地右旋，靠近红桶，靠近蓝桶，靠近绿桶，绕开红桶前往蓝桶，保持站立。"

BACKENDS = {
    "small_gpu": {
        "name": "small_gpu",
        "model_dir": DEFAULT_MODEL_DIR,
        "device": "cuda",
        "compute_type": "float16",
        "initial_prompt": DOMAIN_PROMPT,
        "no_repeat_ngram_size": 0,
    },
    "small_cpu": {
        "name": "small_cpu",
        "model_dir": DEFAULT_MODEL_DIR,
        "device": "cpu",
        "compute_type": "int8",
        "initial_prompt": DOMAIN_PROMPT,
        "no_repeat_ngram_size": 0,
    },
    "base_cpu": {
        "name": "base_cpu",
        "model_dir": BASE_MODEL_DIR,
        "device": "cpu",
        "compute_type": "int8",
        "initial_prompt": None,
        "no_repeat_ngram_size": 4,
    },
}

_AUTO = object()
_MODEL = None
_MODEL_KEY = None
_MODEL_BACKEND: dict | None = None
_BACKEND: dict | None = None


def _cuda_free_gib() -> float:
    try:
        import torch

        if not torch.cuda.is_available():
            return 0.0
        free, _total = torch.cuda.mem_get_info()
        return float(free) / float(2 ** 30)
    except Exception:
        return 0.0


def select_backend(force: str | None = None) -> dict:
    """选择并缓存后端；force 或环境变量 DP2_ASR_BACKEND 可覆盖 auto。"""
    global _BACKEND
    choice = (force or os.environ.get("DP2_ASR_BACKEND") or "auto").strip().lower()
    if choice not in ("auto", "small_gpu", "small_cpu", "base_cpu"):
        raise ValueError("未知 ASR 后端 %r（可选 auto/small_gpu/small_cpu/base_cpu）" % choice)
    free = _cuda_free_gib()
    reason = ""
    if choice == "auto":
        if free >= 1.5 and DEFAULT_MODEL_DIR.exists():
            choice = "small_gpu"
            reason = "CUDA 空闲 %.2f GiB -> small float16" % free
        elif BASE_MODEL_DIR.exists():
            choice = "base_cpu"
            reason = "CUDA 空闲 %.2f GiB（不足 1.5 或无 GPU）-> base int8" % free
        else:
            choice = "small_cpu"
            reason = "无 GPU 且 base 模型缺失 -> small int8"
    spec = dict(BACKENDS[choice])
    spec["cuda_free_gib"] = round(free, 2)
    spec["reason"] = reason or ("强制 %s" % choice)
    _BACKEND = spec
    return spec


def get_backend(force: str | None = None) -> dict:
    if force and force != "auto":
        return select_backend(force)
    if _BACKEND is None:
        return select_backend()
    return _BACKEND


def get_backend_info() -> dict:
    spec = dict(get_backend())
    spec["model_dir"] = str(spec["model_dir"])
    return spec


def reset_model() -> None:
    global _MODEL, _MODEL_KEY, _MODEL_BACKEND
    _MODEL = None
    _MODEL_KEY = None
    _MODEL_BACKEND = None


def load_model(model_dir: str | Path | None = None, device: str | None = None,
               compute_type: str | None = None, cpu_threads: int = 4,
               backend: dict | None = None):
    """加载并缓存 CTranslate2 Whisper 模型（local_files_only）。"""
    global _MODEL, _MODEL_KEY, _MODEL_BACKEND
    if backend is None and model_dir is None:
        backend = get_backend()
    if backend is not None:
        model_dir = backend["model_dir"]
        device = device or backend["device"]
        compute_type = compute_type or backend["compute_type"]
    model_path = Path(model_dir) if model_dir else DEFAULT_MODEL_DIR
    device = device or "cpu"
    compute_type = compute_type or "int8"
    if not model_path.exists():
        raise FileNotFoundError(
            "找不到 Whisper 模型目录 %s；请先按 deploy/asr/REPORT.md 的说明下载" % model_path
        )
    key = (str(model_path), device, compute_type, int(cpu_threads))
    if _MODEL is not None and _MODEL_KEY == key:
        _MODEL_BACKEND = backend
        return _MODEL
    from faster_whisper import WhisperModel

    _MODEL = WhisperModel(
        str(model_path),
        device=device,
        compute_type=compute_type,
        cpu_threads=cpu_threads,
        local_files_only=True,
    )
    _MODEL_KEY = key
    _MODEL_BACKEND = backend
    return _MODEL


def _resample(samples: np.ndarray, rate_in: int, rate_out: int = SAMPLE_RATE) -> np.ndarray:
    if rate_in == rate_out:
        return samples.astype(np.float32, copy=False)
    if samples.size == 0:
        return samples.astype(np.float32, copy=False)
    duration = samples.shape[0] / float(rate_in)
    count_out = max(1, int(round(duration * rate_out)))
    x_old = np.linspace(0.0, 1.0, samples.shape[0], dtype=np.float64)
    x_new = np.linspace(0.0, 1.0, count_out, dtype=np.float64)
    return np.interp(x_new, x_old, samples).astype(np.float32)


def load_audio(audio, sample_rate: int | None = None) -> np.ndarray:
    """把各种输入统一成 16 kHz、单声道、float32 [-1, 1]。"""
    if isinstance(audio, (str, bytes, Path)):
        path = Path(audio)
        if not path.exists():
            raise FileNotFoundError("音频文件不存在：%s" % path)
        import soundfile as sf

        samples, rate = sf.read(str(path), dtype="float32", always_2d=False)
        samples = np.asarray(samples, dtype=np.float32)
        if samples.ndim > 1:
            samples = samples.mean(axis=1)
        return _resample(samples, int(rate))
    if isinstance(audio, tuple) and len(audio) == 2:
        samples, rate = audio
        samples = np.asarray(samples, dtype=np.float32)
        if samples.ndim > 1:
            samples = samples.mean(axis=1)
        return _resample(samples, int(rate))
    samples = np.asarray(audio, dtype=np.float32)
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    if sample_rate is not None and int(sample_rate) != SAMPLE_RATE:
        return _resample(samples, int(sample_rate))
    return samples


def list_mic_devices() -> list[dict]:
    """解析 arecord -l，返回可用录音设备。"""
    devices = []
    try:
        output = subprocess.run(["arecord", "-l"], capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.SubprocessError):
        return devices
    for line in output.splitlines():
        if not line.startswith("card "):
            continue
        parts = line.split()
        try:
            card = int(parts[1].rstrip(":"))
            device = int(parts[parts.index("device") + 1].rstrip(":"))
        except (ValueError, IndexError):
            continue
        name = line.split("[", 1)[1].split("]", 1)[0] if "[" in line else ""
        devices.append({"card": card, "device": device, "name": name,
                        "alsa": "plughw:CARD=%s,DEV=%d" % (name.replace(" ", "_"), device)})
    return devices


def record_mic(duration_s: float = 3.0, device: str | None = None,
               sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """用 arecord 录一段 16 kHz 单声道，返回 float32（小数秒会录音后裁掉尾部）。"""
    duration_s = max(0.2, float(duration_s))
    seconds = max(1, int(math.ceil(duration_s)))
    target = device or DEFAULT_MIC
    handle = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    temp_path = handle.name
    handle.close()
    command = [
        "arecord", "-q", "-D", target, "-f", "S16_LE",
        "-r", str(int(sample_rate)), "-c", "1",
        "-d", str(seconds), temp_path,
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=seconds + 10)
        if result.returncode != 0:
            raise RuntimeError("录音失败（arecord 返回 %d）：%s" % (result.returncode, result.stderr.strip()))
        import soundfile as sf

        samples, rate = sf.read(temp_path, dtype="float32", always_2d=False)
        samples = np.asarray(samples, dtype=np.float32)
        if samples.ndim > 1:
            samples = samples.mean(axis=1)
        samples = _resample(samples, int(rate))
        wanted = int(round(duration_s * SAMPLE_RATE))
        if 0 < wanted < samples.shape[0]:
            samples = samples[:wanted]
        return samples.astype(np.float32, copy=False)
    finally:
        try:
            os.unlink(temp_path)
        except OSError:
            pass


def transcribe_with_meta(audio, model=None, language: str = "zh", beam_size: int = 5,
                         vad_filter: bool = False,
                         initial_prompt: str | None | object = _AUTO,
                         no_repeat_ngram_size: int | None = None,
                         repetition_penalty: float = 1.0,
                         model_dir: str | Path | None = None,
                         backend: str | dict | None = None) -> dict:
    """转写并返回文本、耗时与所用后端等元数据。"""
    backend_spec = None
    if isinstance(backend, dict):
        backend_spec = backend
    elif isinstance(backend, str):
        backend_spec = get_backend(backend)
    elif model is None:
        backend_spec = get_backend()
    if backend_spec is None:
        backend_spec = _MODEL_BACKEND
    if model is None:
        model = load_model(backend=backend_spec)
    if initial_prompt is _AUTO:
        prompt = backend_spec["initial_prompt"] if backend_spec else DOMAIN_PROMPT
    else:
        prompt = initial_prompt
    if no_repeat_ngram_size is None:
        ngram = backend_spec["no_repeat_ngram_size"] if backend_spec else 0
    else:
        ngram = int(no_repeat_ngram_size)
    samples = load_audio(audio)
    duration = samples.shape[0] / float(SAMPLE_RATE)
    started = time.perf_counter()
    segments, info = whisper_transcribe(model, samples, language, beam_size, vad_filter,
                                        prompt, ngram, repetition_penalty)
    text = "".join(segment.text for segment in segments).strip()
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    result = {
        "text": text,
        "language": getattr(info, "language", language),
        "language_probability": float(getattr(info, "language_probability", 0.0) or 0.0),
        "audio_duration_s": round(float(duration), 3),
        "latency_ms": round(elapsed_ms, 1),
        "backend": (backend_spec or {}).get("name", "custom"),
    }
    return result


def whisper_transcribe(model, samples, language, beam_size, vad_filter,
                       initial_prompt, no_repeat_ngram_size, repetition_penalty):
    return model.transcribe(
        samples,
        language=language,
        beam_size=beam_size,
        vad_filter=vad_filter,
        condition_on_previous_text=False,
        initial_prompt=initial_prompt,
        no_repeat_ngram_size=no_repeat_ngram_size,
        repetition_penalty=repetition_penalty,
    )


def transcribe(audio, model=None, language: str = "zh", beam_size: int = 5,
               vad_filter: bool = False,
               initial_prompt: str | None | object = _AUTO,
               no_repeat_ngram_size: int | None = None,
               repetition_penalty: float = 1.0,
               model_dir: str | Path | None = None,
               backend: str | dict | None = None) -> str:
    """spec 接口：audio_16k_mono_float32 / wav 路径 / 麦克风数组 → 文本。"""
    return transcribe_with_meta(
        audio, model=model, language=language, beam_size=beam_size,
        vad_filter=vad_filter, initial_prompt=initial_prompt,
        no_repeat_ngram_size=no_repeat_ngram_size, repetition_penalty=repetition_penalty,
        model_dir=model_dir, backend=backend,
    )["text"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="本地 faster-whisper 中文 ASR（自动选择后端）")
    parser.add_argument("--wav", default=None, help="音频文件路径（wav/mp3/flac）")
    parser.add_argument("--mic", type=float, default=None, help="录音秒数，例如 3")
    parser.add_argument("--device", default=None, help="arecord 设备，默认 %s" % DEFAULT_MIC)
    parser.add_argument("--model", default=None, help="模型目录，默认按后端自动选择")
    parser.add_argument("--backend", default="auto",
                        choices=["auto", "small_gpu", "small_cpu", "base_cpu"],
                        help="ASR 后端（auto: GPU 可用用 small float16，否则 base int8）")
    parser.add_argument("--beam", type=int, default=5)
    parser.add_argument("--no-prompt", action="store_true", help="不使用领域提示词")
    parser.add_argument("--vad", action="store_true", help="启用 VAD 过滤")
    parser.add_argument("--ngram", type=int, default=None, help="no_repeat_ngram_size 覆盖")
    parser.add_argument("--normalize", action="store_true", help="同时做指令归一化")
    parser.add_argument("--bench", type=int, default=0, help="重复 N 次测 p50 延迟")
    parser.add_argument("--list-mics", action="store_true")
    args = parser.parse_args(argv)

    if args.list_mics:
        for item in list_mic_devices():
            print(json.dumps(item, ensure_ascii=False))
        return 0
    if args.wav is None and args.mic is None:
        parser.print_help()
        return 1

    backend = get_backend(args.backend)
    model = load_model(args.model, backend=backend if args.model is None else None)
    print("[asr] backend=%s device=%s compute=%s（%s）" % (
        backend["name"], backend["device"], backend["compute_type"], backend["reason"]),
        file=sys.stderr)
    if args.mic is not None:
        audio = record_mic(args.mic, args.device)
        source = "mic"
    else:
        audio = args.wav
        source = args.wav

    prompt = None if args.no_prompt else _AUTO
    repeats = max(1, args.bench)
    results = []
    for _ in range(repeats):
        meta = transcribe_with_meta(audio, model=model, beam_size=args.beam,
                                    vad_filter=args.vad, initial_prompt=prompt,
                                    no_repeat_ngram_size=args.ngram)
        results.append(meta)
    latencies = sorted(item["latency_ms"] for item in results)
    middle = int(len(latencies) / 2)
    payload = dict(results[-1])
    payload.update(source=source, repeats=repeats, p50_ms=latencies[middle],
                   min_ms=latencies[0], max_ms=latencies[-1],
                   backend_info=dict(backend, model_dir=str(backend["model_dir"])))
    if args.normalize:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from normalize import normalize

        payload["normalized"] = normalize(payload["text"])
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
