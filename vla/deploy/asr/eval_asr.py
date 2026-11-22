"""ASR + 指令归一化评测脚本。

用法：
    python deploy/asr/eval_asr.py                 # 跑录音集（需先有 testaudio/manifest.json）
    python deploy/asr/eval_asr.py --text-only     # 只跑纯文本错字/口语用例
    python deploy/asr/eval_asr.py --limit 5       # 只跑前 5 条语音
    python deploy/asr/eval_asr.py --out deploy/asr/report_asr.json

输出：
    - 每条用例的识别文本、归一化 id、期望 id、是否正确、ASR 延迟；
    - 汇总：语音端到端指令正确率、ASR 原文一致率、归一化 p50 延迟；
    - 纯文本（口语/错字/数字/拒识）正确率；
    - JSON 报告（默认 deploy/asr/report_asr.json）。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from normalize import normalize, clean_text  # noqa: E402

AUDIO_DIR = HERE / "testaudio"
MANIFEST = AUDIO_DIR / "manifest.json"
SLOW_DIR = HERE / "testaudio_slow"
SLOW_MANIFEST = SLOW_DIR / "manifest.json"

# 纯文本用例：口语、错字、数字、拒识（不依赖 Whisper）
TEXT_CASES = [
    ("往前走", 1),
    ("往前走一点", 1),
    ("向后退", 2),
    ("退一下", 2),
    ("左转", 3),
    ("右转", 4),
    ("原地转圈", 5),
    ("原地右转", 6),
    ("去找红桶", 8),
    ("靠近蓝桶", 9),
    ("去找绿桶", 10),
    ("绕开红桶去蓝桶", 11),
    ("红筒", 8),
    ("靠近蓝铜", 9),
    ("去找录筒", 10),
    ("向右砖", 4),
    ("往前揍一点", 1),
    ("腿一下", 2),
    ("保持战力", 0),
    ("圆地转圈", 5),
    ("对着红桶", 7),
    ("第12条", 11),
    ("指令3", 3),
    ("请帮我走到绿桶那里", 10),
    ("今天天气不错", None),
    ("打开音乐", None),
]


def run_text_cases() -> dict:
    rows = []
    correct = 0
    for text, expected in TEXT_CASES:
        result = normalize(text)
        good = result["id"] == expected
        correct += int(good)
        rows.append({"input": text, "expected_id": expected, "got_id": result["id"],
                     "got_text": result["text"], "method": result["method"],
                     "score": result["score"], "ok": good})
    return {"total": len(rows), "correct": correct,
            "accuracy": round(correct / float(len(rows)), 4), "rows": rows}


def run_audio(limit: int | None = None, backend: str = "auto",
              include_slow: bool = False) -> dict:
    import asr

    if not MANIFEST.exists():
        return {"error": "缺少 %s（用 gen_testaudio.py 生成）" % MANIFEST}
    entries = [("1.0x", AUDIO_DIR, item)
               for item in json.loads(MANIFEST.read_text(encoding="utf-8"))]
    if include_slow and SLOW_MANIFEST.exists():
        entries += [("0.75x", SLOW_DIR, item)
                    for item in json.loads(SLOW_MANIFEST.read_text(encoding="utf-8"))]
    if limit:
        entries = entries[:limit]
    spec = asr.get_backend(backend)
    model = asr.load_model(backend=spec)
    rows = []
    correct = 0
    text_exact = 0
    latencies = []
    for speed, folder, item in entries:
        meta = asr.transcribe_with_meta(str(folder / item["file"]), model=model)
        result = normalize(meta["text"])
        good = result["id"] == item["expected_id"]
        same_text = clean_text(meta["text"]) == clean_text(item["spoken"])
        correct += int(good)
        text_exact += int(same_text)
        latencies.append(meta["latency_ms"])
        rows.append({
            "wav": item["file"],
            "speed": speed,
            "spoken": item["spoken"],
            "asr_text": meta["text"],
            "asr_exact": same_text,
            "normalized_id": result["id"],
            "normalized_text": result["text"],
            "expected_id": item["expected_id"],
            "score": result["score"],
            "latency_ms": meta["latency_ms"],
            "ok": good,
        })
    sorted_lat = sorted(latencies)
    middle = int(len(sorted_lat) / 2)
    return {
        "backend": dict(spec, model_dir=str(spec["model_dir"])),
        "total": len(rows),
        "correct": correct,
        "accuracy": round(correct / float(len(rows)), 4),
        "asr_exact": round(text_exact / float(len(rows)), 4),
        "latency_p50_ms": sorted_lat[middle],
        "latency_min_ms": sorted_lat[0],
        "latency_max_ms": sorted_lat[-1],
        "rows": rows,
    }


def print_table(rows: list[dict]) -> None:
    print("| 语音/文本 | ASR 文本 | 归一化 id | 期望 id | 正确 |")
    print("|---|---|---|---|---|")
    for row in rows:
        if "asr_text" in row:
            speed = row.get("speed")
            left = row["spoken"] if not speed else "%s（%s）" % (row["spoken"], speed)
            mid = row["asr_text"]
            got = row["normalized_id"]
            expected = row["expected_id"]
        else:
            left = row["input"]
            mid = "(纯文本)"
            got = row["got_id"]
            expected = row["expected_id"]
        mark = "✅" if row.get("ok") else "❌"
        print("| %s | %s | %s | %s | %s |" % (left, mid, got, expected, mark))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ASR + 归一化评测")
    parser.add_argument("--text-only", action="store_true", help="只跑纯文本用例，不加载模型")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--slow", action="store_true", help="同时跑 0.75x 语速音频")
    parser.add_argument("--backend", default="auto",
                        choices=["auto", "small_gpu", "small_cpu", "base_cpu"],
                        help="ASR 后端（auto: GPU 可用用 small float16，否则 base int8）")
    parser.add_argument("--out", default=str(HERE / "report_asr.json"))
    args = parser.parse_args(argv)

    report = {"generated_at": time.strftime("%Y-%m-%d %H:%M:%S"), "text": run_text_cases()}
    print("## 纯文本用例（口语 / 错字 / 数字 / 拒识）")
    print_table(report["text"]["rows"])
    print("\n纯文本：%d/%d = %.1f%%" % (
        report["text"]["correct"], report["text"]["total"], 100.0 * report["text"]["accuracy"]))

    if not args.text_only:
        audio = run_audio(args.limit or None, backend=args.backend, include_slow=args.slow)
        report["audio"] = audio
        print("\n## 语音端到端（TTS 合成 → Whisper → 归一化）")
        if "error" in audio:
            print(audio["error"])
        else:
            spec = audio["backend"]
            print("后端：%s / %s / %s（%s）" % (
                spec["name"], spec["device"], spec["compute_type"], spec["reason"]))
            print_table(audio["rows"])
            print("\n语音端到端：%d/%d = %.1f%%；ASR 原文一致率 %.1f%%；延迟 p50 %.0f ms（min %.0f / max %.0f）" % (
                audio["correct"], audio["total"], 100.0 * audio["accuracy"],
                100.0 * audio["asr_exact"], audio["latency_p50_ms"],
                audio["latency_min_ms"], audio["latency_max_ms"]))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n报告已写入 %s" % out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
