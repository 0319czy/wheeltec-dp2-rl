"""Dataset validator for VLA spec section 3. Read-only.

Usage: .venv/bin/python scripts/ds_validate.py --data data/smoke [--samples-out media/samples.png]
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

INSTR = ["stand", "forward", "backward", "left turn", "right turn", "spin left",
         "spin right", "face red", "approach red", "approach blue", "approach green",
         "avoid red go blue"]
KEYS = ["images", "actions", "instr_ids", "episode_ids", "step_idx", "scene_seeds"]
FAILS = []


def bad(msg):
    FAILS.append(msg)
    print("[FAIL] " + msg)


def good(msg):
    print("[ OK ] " + msg)


def scan(shard, st):
    d = np.load(shard)
    miss = [k for k in KEYS if k not in d.files]
    if miss:
        bad(shard.name + " missing keys " + str(miss))
        return
    img = d["images"]
    act = d["actions"]
    ii = d["instr_ids"]
    si = d["step_idx"]
    n = int(act.shape[0])
    if img.shape != (n, 128, 128, 3):
        bad(shard.name + " images shape " + str(img.shape))
    if img.dtype != np.uint8:
        bad(shard.name + " images dtype " + str(img.dtype))
    if act.shape != (n, 2) or act.dtype != np.float32:
        bad(shard.name + " actions " + str(act.shape) + " " + str(act.dtype))
    for k in KEYS:
        if len(d[k]) != n:
            bad(shard.name + " key len mismatch " + k)
    if n == 0:
        return
    if ii.min() < 0 or ii.max() > 11:
        bad(shard.name + " instr_ids out of range")
    if si.min() < 0 or si.max() > 49:
        bad(shard.name + " step_idx out of range")
    if np.abs(act[:, 0]).max() > 0.3501:
        bad(shard.name + " v out of range")
    if np.abs(act[:, 1]).max() > 1.0001:
        bad(shard.name + " w out of range")
    st["frames"] += n
    st["eps"].update(int(x) for x in np.unique(d["episode_ids"]))
    st["amin"] = np.minimum(st["amin"], act.min(0))
    st["amax"] = np.maximum(st["amax"], act.max(0))
    for v in ii:
        st["counts"][int(v)] += 1
    st["means"].append(float(img.reshape(n, -1).mean()))
    for ci in np.unique(ii):
        ci = int(ci)
        if ci not in st["first"]:
            pos = int(np.where(ii == ci)[0][0])
            st["first"][ci] = img[pos]


def save_grid(first, out_path):
    from PIL import Image, ImageDraw
    items = sorted(first.items())
    cols = 4
    n_rows = int(math.ceil(len(items) / float(cols)))
    canvas = Image.new("RGB", (cols * 132, n_rows * 150), (30, 30, 30))
    pen = ImageDraw.Draw(canvas)
    for idx in range(len(items)):
        iid, im = items[idx]
        x = (idx % cols) * 132 + 2
        y = int(idx / cols) * 150 + 18
        canvas.paste(Image.fromarray(im), (x, y))
        pen.text((x, y - 14), "instr " + str(iid), fill=(255, 255, 255))
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out)
    print("[ OK ] samples -> " + str(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--samples-out", default=None)
    args = ap.parse_args()
    root = Path(args.data)
    sdir = root / "shards"
    if not sdir.is_dir():
        print("no shards dir " + str(sdir))
        return 1
    shards = sorted(sdir.glob("*.npz"))
    if not shards:
        print("no npz in " + str(sdir))
        return 1
    good("shards: " + str(len(shards)))
    st = {"frames": 0, "eps": set(), "amin": np.full(2, np.inf),
          "amax": np.full(2, -np.inf), "counts": np.zeros(12, dtype=np.int64),
          "means": [], "first": {}}
    for sh in shards:
        try:
            scan(sh, st)
        except Exception as exc:
            bad(sh.name + " parse error " + repr(exc))
    good("frames " + str(st["frames"]) + " episodes " + str(len(st["eps"])))
    good("action range v=[%.3f,%.3f] w=[%.3f,%.3f]" % (st["amin"][0], st["amax"][0],
                                                       st["amin"][1], st["amax"][1]))
    if st["means"]:
        avg = float(np.mean(st["means"]))
        good("mean brightness %.1f" % avg)
        if avg < 40.0 or avg > 200.0:
            bad("brightness out of 40-200")
    total = max(1, st["frames"])
    for i in range(12):
        pct = 100.0 * st["counts"][i] / total
        print("        %2d %-16s %7d  %.1f%%" % (i, INSTR[i], st["counts"][i], pct))
    if st["counts"].min() == 0:
        bad("some instruction has zero frames")
    if st["counts"].min() < 200:
        bad("min frames per instr %d < 200" % st["counts"].min())
    ip = root / "index.jsonl"
    if not ip.exists():
        bad("no index.jsonl")
    else:
        lines = [json.loads(x) for x in ip.read_text().splitlines() if x.strip()]
        good("index lines " + str(len(lines)))
        ids = [rec.get("episode_id") for rec in lines]
        if len(set(ids)) != len(ids):
            bad("duplicate episode_id in index")
        missing = [e for e in st["eps"] if e not in set(ids)]
        if missing:
            bad(str(len(missing)) + " episodes missing from index")
        need = ["episode_id", "scene_seed", "instr_id", "instruction", "success", "T"]
        head = lines[0] if lines else {}
        for k in need:
            if k not in head:
                bad("index missing field " + k)
        seeds = set(rec.get("scene_seed") for rec in lines)
        good("paired episodes extra " + str(len(lines) - len(seeds)))
        succ = [rec.get("success") for rec in lines if rec.get("success") is not None]
        if succ:
            print("        expert success %.1f%% (%d/%d)" % (100.0 * float(np.mean(succ)),
                                                            int(np.sum(succ)), len(succ)))
    sp = root / "stats.json"
    if not sp.exists():
        bad("no stats.json")
    else:
        stj = json.loads(sp.read_text())
        good("stats keys " + str(sorted(stj.keys())[:8]))
        if "num_frames" in stj and int(stj["num_frames"]) != st["frames"]:
            bad("stats num_frames mismatch")
    if args.samples_out and st["first"]:
        try:
            save_grid(st["first"], args.samples_out)
        except Exception as exc:
            print("[WARN] grid failed " + repr(exc))
    print("=" * 50)
    if FAILS:
        print("FAILED: " + str(len(FAILS)))
        return 1
    print("PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
