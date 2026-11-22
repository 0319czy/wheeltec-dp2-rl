"""Camera / scene self-check: render several random rounds and write media/camera_check.png.

Usage
-----
    .venv/bin/python sim/smoke_render.py --out media/camera_check.png
    .venv/bin/python sim/smoke_render.py --out media/camera_check.png --rounds 12

For every round the script reports (and prints):
  * RGB image mean / per-channel mean / p5 / p95 / fraction of very dark and very bright pixels
  * pixels covered by each bucket colour, measured with MuJoCo segmentation rendering
  * whether all three buckets are inside the camera view, bucket slot positions and layout mode

The montage is 2x upscaled (each 128x128 frame is drawn at 256x256) with a text label so the
buckets can be recognised by eye.  A JSON with all statistics is written next to the png.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from env_vla import COLORS, VLAEnv  # noqa: E402


def collect(env, seed):
    """One fresh round; returns RGB image and segmentation statistics."""
    image = env.reset(instr_id=0, seed=seed)
    image_clean = env.render_rgb()
    counts = env.bucket_pixel_counts()
    total_px = image_clean.shape[0] * image_clean.shape[1]
    mean = float(image_clean.mean())
    channel = [float(image_clean[:, :, c].mean()) for c in range(3)]
    p5 = float(np.percentile(image_clean, 5))
    p95 = float(np.percentile(image_clean, 95))
    dark = float((image_clean.mean(axis=2) < 20).mean())
    bright = float((image_clean.mean(axis=2) > 200).mean())
    buckets = {color: int(counts.get(color, 0)) for color in COLORS}
    return {
        "seed": int(seed),
        "mode": env.scene["mode"],
        "colors": [COLORS[c] for c in env.scene["colors"]],
        "positions": [[round(float(v), 3) for v in p] for p in env.scene["positions"]],
        "image_mean": round(mean, 2),
        "channel_mean": [round(v, 2) for v in channel],
        "p5": round(p5, 1),
        "p95": round(p95, 1),
        "dark_frac": round(dark, 4),
        "bright_frac": round(bright, 4),
        "bucket_pixels": buckets,
        "bucket_total_px": int(sum(buckets.values())),
        "bucket_frac": round(sum(buckets.values()) / total_px, 5),
        "all_visible": bool(all(v > 0 for v in buckets.values())),
        "light_scale": round(float(env.scene["light_scale"]), 3),
        "floor_scale": round(float(env.scene["floor_scale"]), 3),
        "image": image,
    }


def montage(rounds, out_path, upscale=2, cols=3):
    from PIL import Image, ImageDraw

    n = len(rounds)
    rows = int(np.ceil(n / cols))
    cell = 128 * upscale
    pad = 26
    canvas = Image.new("RGB", (cols * cell, rows * (cell + pad)), (18, 18, 18))
    draw = ImageDraw.Draw(canvas)
    for k, rnd in enumerate(rounds):
        row = int(k / cols)
        col = k % cols
        tile = Image.fromarray(rnd["image"]).resize((cell, cell), Image.NEAREST)
        canvas.paste(tile, (col * cell, row * (cell + pad) + pad))
        label = "seed %d %s  mean %.0f  bucket %.1f%%" % (
            rnd["seed"], rnd["mode"], rnd["image_mean"], 100.0 * rnd["bucket_frac"])
        draw.text((col * cell + 4, row * (cell + pad) + 7), label, fill=(235, 235, 235))
        draw.text((col * cell + 4, row * (cell + pad) + pad + cell - 14),
                  " | ".join(rnd["colors"]), fill=(255, 230, 120))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    return canvas.size


def main():
    ap = argparse.ArgumentParser(description="render camera check montage")
    ap.add_argument("--out", type=str, default="media/camera_check.png")
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--seed0", type=int, default=0, help="first scene seed")
    ap.add_argument("--json", type=str, default=None, help="stats json path (default next to png)")
    args = ap.parse_args()

    out = Path(args.out)
    seeds = [args.seed0 + i for i in range(args.rounds)]
    env = VLAEnv(load_policy=False)
    rounds = [collect(env, seed) for seed in seeds]
    env.close()

    size = montage(rounds, out)
    json_path = Path(args.json) if args.json else out.with_suffix(".json")
    stats = {
        "rounds": args.rounds,
        "seeds": seeds,
        "montage": str(out),
        "montage_size": list(size),
        "image_mean_range": [min(r["image_mean"] for r in rounds), max(r["image_mean"] for r in rounds)],
        "bucket_frac_range": [min(r["bucket_frac"] for r in rounds), max(r["bucket_frac"] for r in rounds)],
        "bucket_pixels_range": [min(r["bucket_total_px"] for r in rounds), max(r["bucket_total_px"] for r in rounds)],
        "all_visible_all_rounds": all(r["all_visible"] for r in rounds),
        "rounds_detail": [{k: v for k, v in r.items() if k != "image"} for r in rounds],
    }
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(stats, handle, ensure_ascii=False, indent=2)

    print("wrote %s (%dx%d) and %s" % (out, size[0], size[1], json_path))
    print("%-5s %-8s %-8s %-8s %-6s %-6s %-6s %s" % (
        "seed", "mode", "mean", "p5/p95", "dark", "bright", "allvis", "bucket px by colour (share of frame)"))
    for rnd in rounds:
        px = " ".join("%s=%d" % (color, rnd["bucket_pixels"][color]) for color in COLORS)
        print("%-5d %-8s %-8.1f %3.0f/%-4.0f %-6.2f %-6.2f %-6s %s (%.2f%%)" % (
            rnd["seed"], rnd["mode"], rnd["image_mean"], rnd["p5"], rnd["p95"],
            rnd["dark_frac"], rnd["bright_frac"], rnd["all_visible"], px, 100.0 * rnd["bucket_frac"]))
    lows = [r["image_mean"] for r in rounds]
    print("\nimage mean: %.1f .. %.1f (target 80-160) | bucket pixels: %d .. %d (%.2f%% .. %.2f%% of frame)"
          % (min(lows), max(lows), min(r["bucket_total_px"] for r in rounds),
             max(r["bucket_total_px"] for r in rounds),
             100.0 * min(r["bucket_frac"] for r in rounds), 100.0 * max(r["bucket_frac"] for r in rounds)))
    print("all 3 buckets visible in every round:", all(r["all_visible"] for r in rounds))
    assert 80.0 <= min(lows) and max(lows) <= 160.0, "image mean outside 80-160"
    assert all(r["all_visible"] for r in rounds), "some bucket not visible"
    print("CAMERA CHECK PASSED")


if __name__ == "__main__":
    main()
