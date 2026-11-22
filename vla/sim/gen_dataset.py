"""Generate the VLA behaviour-cloning dataset (spec: specs/VLA-TASK-SPEC.md 3).

Usage
-----
    .venv/bin/python sim/gen_dataset.py --episodes 30 --out data/smoke
    .venv/bin/python sim/gen_dataset.py --episodes 2600 --out data/dataset_v1 --min-success 2500
    .venv/bin/python sim/gen_dataset.py --episodes 600  --out data/dataset_test --split test
    .venv/bin/python sim/gen_dataset.py --episodes 60 --out /tmp/smoke_nocollide --no-collide

The generator runs the privileged expert from sim/expert.py closed loop in sim/env_vla.py,
records (image, physical action, instruction id, step index, scene seed) and writes:

    <out>/shards/shard_XXXX.npz   50 episodes per shard (spec 3)
    <out>/index.jsonl             one line per episode, ordered by episode_id
    <out>/stats.json              totals, action statistics, shard list
    <out>/README.md               generation command, scale, per-instruction counts, samples
    <out>/samples.png             one sample frame per instruction

Only **expert-successful** episodes are written (behaviour cloning): with the real bucket
collision bodies enabled, a small fraction of attempts can collide with a bucket or fall
(and therefore fail the spec judge).  Those attempts are counted in stats.json
(`attempted_episodes`, `discarded_episodes`) but do not enter shards/index.
`--no-collide` is the ablation switch (visual-only buckets, the robot drives through them).

Pairing (spec 3): each scene_seed is used by two episodes with different instructions
(>= 40 % required; the realised rate is reported in stats/index and asserted at the end).
Scene layout mode is derived from the seed (seed % 6 == 5 -> instruction-11 "obstacle"
scene), and the instruction pairs are arranged so instruction 11 always lands on an
obstacle scene.  `--save-gif` additionally records the first successful episode as a GIF.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from env_vla import INSTRUCTIONS, VLAEnv  # noqa: E402
from expert import Expert  # noqa: E402

# Each scene is shared by the two instructions of one pair.  Pair index 5 must be
# (0, 11) so that seed % 6 == 5 (obstacle layout) is used by instruction 11.
PAIRS = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (0, 11)]
BASE_TRAINVAL = 0        # divisible by 6, scene_seeds 0..N/2
BASE_TEST = 100002       # divisible by 6, different seed range for the test split
SHARD_SIZE = 50


def instruction_for_episode(episode_index, split):
    """(scene_seed, instr_id) for one attempt; deterministic and pair-consistent."""
    base = BASE_TEST if split == "test" else BASE_TRAINVAL
    scene_index = int(episode_index / 2)
    scene_seed = base + scene_index
    pair = PAIRS[scene_index % len(PAIRS)]
    instr_id = pair[episode_index % 2]
    return scene_seed, instr_id


BASE_OBJ = 200001  # divisible by 3; object-only dataset seed range (disjoint from v1/test)


def parse_instr_args(instrs, weights):
    """Parse --instrs / --instr-weights into (ids, weights) or None for legacy mode."""
    if not instrs:
        return None
    ids = [int(x) for x in str(instrs).replace(" ", "").split(",") if x != ""]
    if not ids:
        raise SystemExit("--instrs is empty")
    if len(set(ids)) != len(ids):
        raise SystemExit("--instrs contains duplicates")
    for iid in ids:
        if not 0 <= iid <= 11:
            raise SystemExit("--instrs id out of range: %r" % (iid,))
    if weights:
        ws = [int(x) for x in str(weights).replace(" ", "").split(",") if x != ""]
        if len(ws) != len(ids):
            raise SystemExit("--instr-weights length != --instrs length")
        if any(w <= 0 for w in ws):
            raise SystemExit("--instr-weights must be positive integers")
    else:
        ws = [1] * len(ids)
    return ids, ws, dict(zip(ids, ws))


def object_scene_plan(ids, weights):
    """Yield (mode, [instr_id, ...]) scene groups for the object-only dataset.

    Unit (2 weight cycles) for 7:8:9:10:11 = 1:2:2:2:3:
      6 obstacle scenes with a single instruction-11 episode (unique scene each)
      7 generic scenes with two *different* non-11 instructions
    -> 20 attempts / 13 scenes, instr 11 = 30 % of attempts, 46 % obstacle scenes.
    """
    if len(ids) == 1:
        # single-instruction supplement: one unique scene per episode, no duplicate data
        only = int(ids[0])
        mode = "obstacle" if only == 11 else "generic"
        while True:
            yield mode, [only]
    cycle = []
    for iid, weight in zip(ids, weights):
        cycle.extend([iid] * int(weight))
    while True:
        two = list(cycle) * 2
        elevens = [x for x in two if x == 11]
        others = sorted(x for x in two if x != 11)
        half = int(len(others) / 2)
        scenes = []
        if half > 0:
            pairs = []
            for k in range(half):
                a = others[k]
                b = others[k + half]
                if a == b:
                    for j in range(k + half + 1, len(others)):
                        if others[j] != a:
                            others[k + half], others[j] = others[j], others[k + half]
                            b = others[k + half]
                            break
                pairs.append((a, b))
            scenes.extend(("generic", [a, b]) for a, b in pairs)
        scenes.extend(("obstacle", [11]) for _ in elevens)
        # interleave so instruction 11 is not clustered
        generic = [sc for sc in scenes if sc[0] == "generic"]
        obstacle = [sc for sc in scenes if sc[0] == "obstacle"]
        for k in range(max(len(generic), len(obstacle))):
            if k < len(obstacle):
                yield obstacle[k]
            if k < len(generic):
                yield generic[k]


AUX_LIMIT = 3.0
AUX_ORDER = ("red", "blue", "green")


def aux_from_info(info, limit=AUX_LIMIT):
    """Bucket centres in the robot body frame, as [dx_red,dy_red,dx_blue,dy_blue,dx_green,dy_green].

    dx = forward distance along the robot heading (chassis +x), dy = left distance
    (chassis +y), both in metres and clipped to +-limit.  The world -> body rotation is
    R(-yaw), so dx = cos(yaw)*wx + sin(yaw)*wy, dy = -sin(yaw)*wx + cos(yaw)*wy.
    """
    robot_x, robot_y = info["pos"]
    yaw = float(info["yaw"])
    cos_y, sin_y = math.cos(yaw), math.sin(yaw)
    colors = list(info["bucket_colors"])
    positions = info["bucket_positions"]
    by_color = {}
    for color, pos in zip(colors, positions):
        wx = float(pos[0]) - float(robot_x)
        wy = float(pos[1]) - float(robot_y)
        dx = cos_y * wx + sin_y * wy
        dy = -sin_y * wx + cos_y * wy
        by_color[color] = (dx, dy)
    values = []
    for color in AUX_ORDER:
        dx, dy = by_color[color]
        values.extend((float(np.clip(dx, -limit, limit)), float(np.clip(dy, -limit, limit))))
    return np.asarray(values, dtype=np.float32)


def rollout(env, expert, instr_id, seed, want_aux=False):
    """One expert episode; returns images, actions, aux (or None) and metadata."""
    image = env.reset(instr_id=instr_id, seed=seed)
    info = env.history[0]
    images = []
    actions = []
    auxes = []
    done = False
    while not done:
        action = np.asarray(expert.act(instr_id, info), dtype=np.float32)
        action = np.array([float(np.clip(action[0], -0.35, 0.35)),
                           float(np.clip(action[1], -1.0, 1.0))], dtype=np.float32)
        images.append(image)
        actions.append(action)
        if want_aux:
            auxes.append(aux_from_info(info))
        image, done, info = env.step(action)
    images = np.stack(images, axis=0).astype(np.uint8)
    actions = np.stack(actions, axis=0).astype(np.float32)
    aux = np.stack(auxes, axis=0).astype(np.float32) if want_aux else None
    meta = {
        "success": bool(info["success"]),
        "fall": bool(info["fall"]),
        "T": int(len(actions)),
        "cum_dyaw": float(info["cum_dyaw"]),
        "dist": {str(k): float(v) for k, v in info["dist"].items()},
        "pos": [float(info["pos"][0]), float(info["pos"][1])],
        "yaw": float(info["yaw"]),
    }
    return images, actions, aux, meta


def make_sample_grid(samples, path, upscale=2):
    """samples: list of (label, uint8 image) -> write a labelled grid png."""
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return False
    cols = 4
    rows = int(np.ceil(len(samples) / cols))
    cell = 128 * upscale
    pad = 18
    canvas = Image.new("RGB", (cols * cell, rows * (cell + pad)), (20, 20, 20))
    draw = ImageDraw.Draw(canvas)
    for k, (label, image) in enumerate(samples):
        row = int(k / cols)
        col = k % cols
        tile = Image.fromarray(image).resize((cell, cell), Image.NEAREST)
        canvas.paste(tile, (col * cell, row * (cell + pad) + pad))
        draw.text((col * cell + 4, row * (cell + pad) + 3), label, fill=(235, 235, 235))
    canvas.save(path)
    return True


def save_gif(images, path, duration_ms=100):
    try:
        from PIL import Image
    except ImportError:
        return False
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = [Image.fromarray(frame).resize((256, 256), Image.NEAREST) for frame in images]
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=duration_ms, loop=0)
    return True


def write_dataset(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    shards_dir = out / "shards"
    if shards_dir.exists():
        for old in shards_dir.glob("*.npz"):
            old.unlink()
    for old_name in ("index.jsonl", "stats.json", "README.md", "samples.png"):
        old_path = out / old_name
        if old_path.is_file():
            old_path.unlink()
    (out / "shards").mkdir(parents=True, exist_ok=True)

    collide = not args.no_collide
    instr_arg = parse_instr_args(args.instrs, args.instr_weights)
    obstacle_mod = args.obstacle_mod
    obstacle_phase = args.obstacle_phase
    if obstacle_mod is None:
        obstacle_mod = 3 if instr_arg else 6
    if obstacle_phase is None:
        obstacle_phase = 0 if instr_arg else 5
    if args.obstacle_only:
        obstacle_mod, obstacle_phase = 1, 0
    want_aux = bool(args.aux)
    env = VLAEnv(max_steps=50, collide=collide, obstacle_mod=obstacle_mod, obstacle_phase=obstacle_phase)
    expert = Expert()

    if instr_arg is not None:
        seed_base = int(args.seed_base) if args.seed_base is not None else BASE_OBJ
        object_plan = object_scene_plan(instr_arg[0], instr_arg[1])
        pending_instrs = []
        current_scene_seed = None
        scene_counters = {"obstacle": 0, "generic": 0}

        def next_attempt():
            nonlocal pending_instrs, current_scene_seed
            if not pending_instrs:
                mode, scene_instrs = next(object_plan)
                if mode == "obstacle":
                    current_scene_seed = seed_base + obstacle_mod * scene_counters["obstacle"]
                    scene_counters["obstacle"] += 1
                else:
                    offset = 1 % obstacle_mod
                    current_scene_seed = seed_base + obstacle_mod * scene_counters["generic"] + offset
                    scene_counters["generic"] += 1
                pending_instrs.extend(scene_instrs)
            return current_scene_seed, pending_instrs.pop(0)
    else:
        next_attempt = None

    episodes_per_instr = [0] * len(INSTRUCTIONS)
    attempts_per_instr = [0] * len(INSTRUCTIONS)
    discard_per_instr = [0] * len(INSTRUCTIONS)
    frames_per_instr = [0] * len(INSTRUCTIONS)
    index_rows = []
    shards = []
    samples = []
    sample_instrs = set()
    gif_saved = False

    shard_images = []
    shard_actions = []
    shard_instr = []
    shard_episode = []
    shard_step = []
    shard_seed = []
    shard_aux = []
    shard_episode_count = 0

    action_sum = np.zeros(2, dtype=np.float64)
    action_sq = np.zeros(2, dtype=np.float64)
    action_min = np.full(2, np.inf)
    action_max = np.full(2, -np.inf)
    aux_min = np.full(6, np.inf)
    aux_max = np.full(6, -np.inf)
    total_frames = 0
    total_actions = 0
    attempted = 0
    discarded = 0
    start = time.time()

    def flush_shard():
        nonlocal shard_images, shard_actions, shard_instr, shard_episode, shard_step, shard_seed
        nonlocal shard_aux, shard_episode_count
        if not shard_images:
            return
        name = "shard_%04d.npz" % len(shards)
        path = out / "shards" / name
        payload = dict(
            images=np.concatenate(shard_images, axis=0).astype(np.uint8),
            actions=np.concatenate(shard_actions, axis=0).astype(np.float32),
            instr_ids=np.concatenate(shard_instr, axis=0).astype(np.int16),
            episode_ids=np.concatenate(shard_episode, axis=0).astype(np.int32),
            step_idx=np.concatenate(shard_step, axis=0).astype(np.int16),
            scene_seeds=np.concatenate(shard_seed, axis=0).astype(np.int32),
        )
        if want_aux:
            payload["aux"] = np.concatenate(shard_aux, axis=0).astype(np.float32)
        np.savez_compressed(path, **payload)
        shards.append(name)
        shard_images = []
        shard_actions = []
        shard_instr = []
        shard_episode = []
        shard_step = []
        shard_seed = []
        shard_aux = []
        shard_episode_count = 0

    min_success = int(args.min_success or 0)
    max_attempts = int(args.episodes) if min_success <= 0 else max(int(args.episodes), min_success + 1200)
    while attempted < int(args.episodes) or (min_success > 0 and len(index_rows) < min_success and attempted < max_attempts):
        attempt_index = attempted
        if next_attempt is not None:
            scene_seed, instr_id = next_attempt()
        else:
            scene_seed, instr_id = instruction_for_episode(attempt_index, args.split)
        images, actions, aux, meta = rollout(env, expert, instr_id, scene_seed, want_aux=want_aux)
        attempted += 1
        attempts_per_instr[instr_id] += 1
        if not meta["success"]:
            discarded += 1
            discard_per_instr[instr_id] += 1
            if attempted % 10 == 0:
                print("attempt %4d  instr %2d seed %6d  DISCARD (fall=%s)  kept %d / %d"
                      % (attempted, instr_id, scene_seed, meta["fall"], len(index_rows), args.episodes), flush=True)
            continue

        episode_id = len(index_rows)
        step_idx = np.arange(len(actions), dtype=np.int16)
        shard_images.append(images)
        shard_actions.append(actions)
        shard_instr.append(np.full(len(actions), instr_id, dtype=np.int16))
        shard_episode.append(np.full(len(actions), episode_id, dtype=np.int32))
        shard_step.append(step_idx)
        shard_seed.append(np.full(len(actions), scene_seed, dtype=np.int32))
        if want_aux:
            shard_aux.append(aux)

        episodes_per_instr[instr_id] += 1
        frames_per_instr[instr_id] += int(meta["T"])
        total_frames += int(meta["T"])
        total_actions += int(meta["T"])
        action_sum += actions.sum(axis=0)
        action_sq += (actions.astype(np.float64) ** 2).sum(axis=0)
        action_min = np.minimum(action_min, actions.min(axis=0))
        action_max = np.maximum(action_max, actions.max(axis=0))
        if want_aux:
            aux_min = np.minimum(aux_min, aux.min(axis=0))
            aux_max = np.maximum(aux_max, aux.max(axis=0))

        index_rows.append(
            {
                "episode_id": int(episode_id),
                "scene_seed": int(scene_seed),
                "instr_id": int(instr_id),
                "instruction": INSTRUCTIONS[instr_id],
                "success": True,
                "fall": False,
                "T": int(meta["T"]),
                "num_step": int(meta["T"]),
                "cum_dyaw": float(meta["cum_dyaw"]),
                "final_dist": meta["dist"],
                "bucket_colors": {"pos": [("red", "blue", "green")[int(c)] for c in env.scene["colors"]]},
                "pair_group": int(scene_seed),
                "scene_mode": env.scene["mode"],
            }
        )
        if instr_id not in sample_instrs:
            samples.append(("%d %s" % (instr_id, INSTRUCTIONS[instr_id]), images[0]))
            sample_instrs.add(instr_id)
        if args.save_gif and not gif_saved:
            try:
                save_gif(images, args.save_gif)
                gif_saved = True
                print("saved GIF -> %s (%d frames)" % (args.save_gif, len(images)), flush=True)
            except Exception as exc:
                print("GIF save failed: %r" % (exc,), flush=True)

        shard_episode_count += 1
        if shard_episode_count >= args.shard_size:
            flush_shard()

        if attempted % 10 == 0 or (attempted >= int(args.episodes) and len(index_rows) >= min_success):
            elapsed = time.time() - start
            print("attempt %4d  kept %4d  discard %3d  instr %2d  seed %6d  T=%2d  %.2f s/ep  fps %.1f"
                  % (attempted, len(index_rows), discarded, instr_id, scene_seed, meta["T"],
                     elapsed / max(attempted, 1), total_frames / max(elapsed, 1e-6)), flush=True)
    flush_shard()
    wall = time.time() - start
    env.close()

    if not index_rows:
        raise SystemExit("no successful episode was generated; nothing written")

    with open(out / "index.jsonl", "w", encoding="utf-8") as handle:
        for row in index_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    # ---- stats.json ------------------------------------------------------------------
    kept = len(index_rows)
    mean = action_sum / max(total_actions, 1)
    var = action_sq / max(total_actions, 1) - mean * mean
    per_scene = {}
    for row in index_rows:
        per_scene.setdefault(int(row["scene_seed"]), set()).add(int(row["instr_id"]))
    paired = sum(1 for row in index_rows if len(per_scene[int(row["scene_seed"])]) >= 2)
    pair_rate = paired / max(kept, 1)
    stats = {
        "dataset": out.name,
        "split": args.split,
        "instrs": instr_arg[0] if instr_arg is not None else list(range(12)),
        "instr_weights": instr_arg[2] if instr_arg is not None else {str(i): 1 for i in range(12)},
        "obstacle_mod": int(obstacle_mod),
        "obstacle_phase": int(obstacle_phase),
        "seed_base": int(args.seed_base) if (instr_arg is not None and args.seed_base is not None) else (BASE_OBJ if instr_arg is not None else (BASE_TEST if args.split == "test" else BASE_TRAINVAL)),
        "episodes": int(kept),
        "attempted_episodes": int(attempted),
        "discarded_episodes": int(discarded),
        "discard_rate": float(discarded / max(attempted, 1)),
        "attempt_success_rate": float(kept / max(attempted, 1)),
        "success_rate": 1.0,  # only successful expert episodes are written (BC dataset)
        "collide": bool(collide),
        "aux": bool(want_aux),
        "frames": int(total_frames),
        "steps_per_episode": 50,
        "image_shape": [128, 128, 3],
        "shard_size": int(args.shard_size),
        "shards": shards,
        "episodes_per_instr": {str(i): int(episodes_per_instr[i]) for i in range(12)},
        "attempts_per_instr": {str(i): int(attempts_per_instr[i]) for i in range(12)},
        "discards_per_instr": {str(i): int(discard_per_instr[i]) for i in range(12)},
        "frames_per_instr": {str(i): int(frames_per_instr[i]) for i in range(12)},
        "success_per_instr": {str(i): int(episodes_per_instr[i]) for i in range(12)},
        "actions": {
            "mean": [float(x) for x in mean],
            "std": [float(x) for x in np.sqrt(np.maximum(var, 0.0))],
            "min": [float(x) for x in action_min],
            "max": [float(x) for x in action_max],
            "v_range": [-0.35, 0.35],
            "w_range": [-1.0, 1.0],
        },
        "scene_count": int(len(per_scene)),
        "paired_episode_rate": float(pair_rate),
        "config": {
            "angle_max_deg": 30.0,
            "radius_min": 0.65,
            "radius_max": 1.4,
            "min_bucket_sep": 0.45,
            "bucket_radius": 0.07,
            "bucket_height": 0.22,
            "image_size": 128,
            "vla_hz": 10,
            "lowlevel_hz": 100,
            "physics_timestep": 0.005,
            "episode_len": 50,
            "camera_noise_sigma": 2.0,
        },
        "wall_time_s": round(wall, 2),
        "render_fps": round(total_frames / max(wall, 1e-6), 2),
        "episodes_per_s": round(kept / max(wall, 1e-6), 3),
        "attempts_per_s": round(attempted / max(wall, 1e-6), 3),
    }
    if want_aux:
        stats["aux_order"] = ["dx_red", "dy_red", "dx_blue", "dy_blue", "dx_green", "dy_green"]
        stats["aux_min"] = [float(x) for x in aux_min]
        stats["aux_max"] = [float(x) for x in aux_max]
    with open(out / "stats.json", "w", encoding="utf-8") as handle:
        json.dump(stats, handle, ensure_ascii=False, indent=2)

    sample_path = out / "samples.png"
    have_samples = make_sample_grid(samples, sample_path)
    write_readme(out, stats, args, have_samples)

    print("\nWROTE", out)
    print("  attempts %d  kept %d (%.1f%%)  discarded %d  frames %d  %.2f s/ep  render %.1f fps"
          % (attempted, kept, 100.0 * kept / max(attempted, 1), discarded, total_frames,
             wall / max(attempted, 1), stats["render_fps"]))
    print("  per-instr kept: %s" % (stats["episodes_per_instr"],))
    check_dataset(out)
    return stats


def write_readme(out, stats, args, have_samples):
    lines = []
    lines.append("# Dataset %s" % out.name)
    lines.append("")
    lines.append("DP2 balance-car VLA behaviour-cloning data, contract: `specs/VLA-TASK-SPEC.md` 3.")
    lines.append("Only expert-successful episodes are stored; failed attempts are discarded")
    lines.append("(bucket collision bodies are **%s**)." % ("enabled" if stats["collide"] else "disabled (ablation)"))
    lines.append("")
    lines.append("## Generation")
    lines.append("")
    lines.append("```bash")
    lines.append("cd /home/czy/倒立摆/wheeltec_vla")
    cmd = ".venv/bin/python sim/gen_dataset.py --episodes %d --out %s" % (stats["attempted_episodes"], out)
    if args.split == "test":
        cmd += " --split test"
    if args.instrs:
        cmd += ' --instrs "%s"' % args.instrs
        if args.instr_weights:
            cmd += ' --instr-weights "%s"' % args.instr_weights
    if args.min_success:
        cmd += " --min-success %d" % int(args.min_success)
    if args.no_collide:
        cmd += " --no-collide"
    cmd += " --workers 1"
    lines.append(cmd)
    lines.append("```")
    lines.append("")
    if len(stats.get("instrs", list(range(12)))) < 12:
        lines.append("- **subset dataset**: only instructions %s are generated (weights %s);"
                     % (stats["instrs"], stats["instr_weights"]))
        lines.append("  `scripts/ds_validate.py` will therefore flag instructions 0-6 as missing/zero frames")
        lines.append("  (expected for this supplement; it is meant to be merged with dataset_v1).")
    lines.append("- attempts: **%d**, kept (successful) episodes: **%d** (%.1f%%), discarded: **%d**"
                 % (stats["attempted_episodes"], stats["episodes"], 100.0 * stats["attempt_success_rate"],
                    stats["discarded_episodes"]))
    lines.append("- frames: **%d**, frames/episode: %d" % (stats["frames"], stats["steps_per_episode"]))
    lines.append("- shards: %d x %d episodes (`shards/`)" % (len(stats["shards"]), stats["shard_size"]))
    lines.append("- %.2f s/attempt, %.1f rendered fps, wall %.1f s, paired %.1f%%"
                 % (stats["wall_time_s"] / max(stats["attempted_episodes"], 1), stats["render_fps"],
                    stats["wall_time_s"], 100.0 * stats["paired_episode_rate"]))
    lines.append("- action mean %s, std %s, v in [%.2f, %.2f], w in [%.2f, %.2f]"
                 % (["%.3f" % x for x in stats["actions"]["mean"]],
                    ["%.3f" % x for x in stats["actions"]["std"]],
                    stats["actions"]["min"][0], stats["actions"]["max"][0],
                    stats["actions"]["min"][1], stats["actions"]["max"][1]))
    lines.append("")
    lines.append("## Per-instruction episodes (all successful)")
    lines.append("")
    lines.append("| id | instruction | kept episodes | frames | discard rate |")
    lines.append("|---|---|---|---|---|")
    for i in range(12):
        ep = stats["episodes_per_instr"][str(i)]
        fr = stats["frames_per_instr"][str(i)]
        att = stats["attempts_per_instr"][str(i)]
        dr = 100.0 * stats["discards_per_instr"][str(i)] / max(att, 1)
        lines.append("| %d | %s | %d | %d | %.0f%% |" % (i, INSTRUCTIONS[i], ep, fr, dr))
    lines.append("")
    if have_samples:
        lines.append("Sample frames (first frame of one episode per instruction): `samples.png`")
        lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- Images returned by `VLAEnv.reset()`/`step()` already include the spec 1.2 camera")
    lines.append("  noise (sigma=2, deterministic from the scene seed); `render_rgb()` returns the clean frame.")
    lines.append("- `scene_seed` pairing: each scene is shared by two instructions; `pair_group` in")
    lines.append("  `index.jsonl` is the scene seed. Split by `scene_seed % 10` (0-7 train, 8-9 val).")
    lines.append("")
    lines.append("## Self-check")
    lines.append("")
    lines.append("`gen_dataset.py` reloads all shards and verifies dtypes/shapes/action ranges/instruction")
    lines.append("coverage/index line count/pairing before returning (`SELF-CHECK PASSED`).")
    lines.append("")
    with open(out / "README.md", "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def check_dataset(out):
    """Streaming reload + assertions (spec 3).  Shards are checked one at a time so the
    2500-episode dataset (about 6 GB of images) never has to be fully resident in RAM."""
    out = Path(out)
    with open(out / "stats.json", encoding="utf-8") as handle:
        stats = json.load(handle)
    index = [json.loads(line) for line in (out / "index.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    shard_names = list(stats["shards"])
    total_frames = 0
    instr_seen = set()
    episode_ids = []
    frames_by_episode = {}
    shard_episode_counts = []
    action_sum = np.zeros(2, dtype=np.float64)
    action_sq = np.zeros(2, dtype=np.float64)
    action_min = np.full(2, np.inf)
    action_max = np.full(2, -np.inf)
    aux_min = np.full(6, np.inf)
    aux_max = np.full(6, -np.inf)
    pixel_sum = 0.0
    pixel_min = 255
    pixel_max = 0
    aux_present = None
    aux_min = np.full(6, np.inf)
    aux_max = np.full(6, -np.inf)
    for name in shard_names:
        with np.load(out / "shards" / name) as shard:
            images = shard["images"]
            actions = shard["actions"]
            instr = shard["instr_ids"]
            ep = shard["episode_ids"]
            step = shard["step_idx"]
            seeds = shard["scene_seeds"]
            n = len(images)
            assert images.dtype == np.uint8 and images.shape == (n, 128, 128, 3), (name, images.shape)
            assert actions.dtype == np.float32 and actions.shape == (n, 2), (name, actions.shape)
            assert instr.dtype == np.int16 and instr.shape == (n,), name
            assert ep.dtype == np.int32 and ep.shape == (n,), name
            assert step.dtype == np.int16 and step.shape == (n,), name
            assert seeds.dtype == np.int32 and seeds.shape == (n,), name
            has_aux = "aux" in shard.files
            if aux_present is None:
                aux_present = has_aux
            assert has_aux == aux_present, (name, "aux key inconsistent across shards")
            if has_aux:
                aux = shard["aux"]
                assert aux.dtype == np.float32 and aux.shape == (n, 6), (name, aux.shape, aux.dtype)
                assert np.all(np.isfinite(aux)), name
                assert float(np.abs(aux).max()) <= 3.0 + 1e-6, (name, float(np.abs(aux).max()))
                aux_min = np.minimum(aux_min, aux.min(axis=0))
                aux_max = np.maximum(aux_max, aux.max(axis=0))
            assert np.all((actions[:, 0] >= -0.35 - 1e-6) & (actions[:, 0] <= 0.35 + 1e-6)), name
            assert np.all((actions[:, 1] >= -1.0 - 1e-6) & (actions[:, 1] <= 1.0 + 1e-6)), name
            assert int(instr.min()) >= 0 and int(instr.max()) <= 11, name
            assert int(step.min()) >= 0 and int(step.max()) <= 49, name
            local_ids = np.unique(ep)
            assert len(local_ids) <= int(stats["shard_size"]), (name, len(local_ids))
            for eid in local_ids:
                mask = ep == eid
                steps = step[mask]
                assert int(steps[0]) == 0 and int(steps.max()) + 1 == len(steps), (name, int(eid))
                assert len(np.unique(instr[mask])) == 1, (name, int(eid))
                assert len(np.unique(seeds[mask])) == 1, (name, int(eid))
                episode_ids.append(int(eid))
                frames_by_episode[int(eid)] = len(steps)
            shard_episode_counts.append(len(local_ids))
            total_frames += n
            instr_seen.update(int(x) for x in np.unique(instr).tolist())
            action_sum += actions.sum(axis=0, dtype=np.float64)
            action_sq += (actions.astype(np.float64) ** 2).sum(axis=0)
            action_min = np.minimum(action_min, actions.min(axis=0))
            action_max = np.maximum(action_max, actions.max(axis=0))
            pixel_sum += float(images.mean()) * n
            pixel_min = min(pixel_min, int(images.min()))
            pixel_max = max(pixel_max, int(images.max()))
    assert len(set(episode_ids)) == len(episode_ids), "duplicate episode ids across shards"
    assert len(index) == int(stats["episodes"]), (len(index), stats["episodes"])
    assert sorted(episode_ids) == list(range(len(index))), "episode ids are not 0..N-1"
    assert total_frames == int(stats["frames"]) == sum(row["T"] for row in index)
    for row in index:
        assert frames_by_episode[int(row["episode_id"])] == int(row["T"]), row["episode_id"]
        assert bool(row["success"]), row["episode_id"]
    expected_instrs = set(int(x) for x in stats.get("instrs", list(range(12))))
    assert instr_seen == expected_instrs, (sorted(instr_seen), sorted(expected_instrs))
    assert int(stats["discarded_episodes"]) == int(stats["attempted_episodes"]) - int(stats["episodes"])
    assert stats["collide"] in (True, False)
    assert bool(stats.get("aux", False)) == bool(aux_present), "stats.aux does not match shard keys"
    if aux_present:
        assert np.allclose(aux_min, np.asarray(stats["aux_min"], dtype=np.float32), atol=1e-5)
        assert np.allclose(aux_max, np.asarray(stats["aux_max"], dtype=np.float32), atol=1e-5)
    mean = action_sum / max(total_frames, 1)
    std = np.sqrt(np.maximum(action_sq / max(total_frames, 1) - mean * mean, 0.0))
    assert np.allclose(mean, np.asarray(stats["actions"]["mean"], dtype=np.float64), atol=1e-5)
    assert np.allclose(std, np.asarray(stats["actions"]["std"], dtype=np.float64), atol=1e-5)
    per_scene = {}
    for row in index:
        per_scene.setdefault(row["scene_seed"], set()).add(row["instr_id"])
    paired = sum(1 for row in index if len(per_scene[row["scene_seed"]]) >= 2)
    pair_rate = paired / max(len(index), 1)
    if len(expected_instrs) >= 2:
        assert pair_rate >= 0.40, pair_rate
    for key in ("actions", "shards", "frames_per_instr", "episodes_per_instr",
                "attempted_episodes", "discarded_episodes", "attempt_success_rate"):
        assert key in stats, key
    for key in ("mean", "std", "min", "max"):
        assert len(stats["actions"][key]) == 2
    print("\nSELF-CHECK PASSED (streaming, one shard at a time)")
    print("  shards %d | images (%d,128,128,3) uint8 | actions (N,2) float32 | instr coverage %s"
          % (len(shard_names), total_frames, sorted(expected_instrs)))
    print("  episodes %d (all success) | index lines %d | frames %d | paired %.1f%%"
          % (len(index), len(index), total_frames, 100.0 * pair_rate))
    print("  action mean %s std %s | range v [%.3f, %.3f] w [%.3f, %.3f] | image mean %.1f (min %d, max %d)"
          % (["%.3f" % x for x in mean], ["%.3f" % x for x in std],
             action_min[0], action_max[0], action_min[1], action_max[1],
             pixel_sum / max(total_frames, 1), pixel_min, pixel_max))
    print("  attempts %d = kept %d + discarded %d | collide %s"
          % (stats["attempted_episodes"], stats["episodes"], stats["discarded_episodes"], stats["collide"]))
    if aux_present:
        print("  aux: present (N,6) float32 | min %s | max %s"
              % (["%.2f" % x for x in aux_min], ["%.2f" % x for x in aux_max]))


def main():
    ap = argparse.ArgumentParser(description="DP2 VLA dataset generator")
    ap.add_argument("--episodes", type=int, default=30, help="number of attempts (failed attempts are discarded)")
    ap.add_argument("--out", type=str, default="data/smoke")
    ap.add_argument("--shard-size", type=int, default=SHARD_SIZE)
    ap.add_argument("--split", choices=("trainval", "test"), default="trainval")
    ap.add_argument("--min-success", type=int, default=0,
                    help="keep attempting until this many successful episodes exist (0 = only --episodes)")
    ap.add_argument("--instrs", type=str, default=None,
                    help="comma separated instruction ids to generate (default: all 12, legacy pair table)")
    ap.add_argument("--instr-weights", type=str, default=None,
                    help="comma separated integer weights for --instrs (e.g. 1,2,2,2,3)")
    ap.add_argument("--obstacle-mod", type=int, default=None,
                    help="scene layout: seed %% mod == phase is an obstacle scene (default 6/5 legacy, 3/0 with --instrs)")
    ap.add_argument("--obstacle-phase", type=int, default=None, help="obstacle layout phase (default 5 legacy, 0 with --instrs)")
    ap.add_argument("--seed-base", type=int, default=None,
                    help="scene seed base for --instrs mode (default 200001; use a disjoint base for supplements)")
    ap.add_argument("--obstacle-only", action="store_true",
                    help="force every scene to use the instruction-11 obstacle layout (obstacle_mod=1)")
    ap.add_argument("--aux", action="store_true",
                    help="store an extra per-frame float32 [N,6] aux array with bucket positions in the body frame")
    ap.add_argument("--no-collide", action="store_true",
                    help="ablation: disable the bucket collision bodies (robot drives through buckets)")
    ap.add_argument("--save-gif", type=str, default=None,
                    help="write the first successful episode as a GIF (for demo material)")
    ap.add_argument("--workers", type=int, default=1, help="accepted for spec command compatibility; single process")
    ap.add_argument("--overwrite", action="store_true", help="kept for compatibility; known outputs are always replaced")
    ap.add_argument("--check-only", action="store_true", help="only re-run the streaming self-check on --out")
    args = ap.parse_args()
    if args.check_only:
        check_dataset(args.out)
        return
    if args.episodes < 12:
        print("warning: fewer than 12 attempts cannot cover all instructions", flush=True)
    write_dataset(args)


if __name__ == "__main__":
    main()
