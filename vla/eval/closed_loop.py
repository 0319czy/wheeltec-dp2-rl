"""闭环评测:VLA 在 sim/env_vla.py 里跑 12 条指令,统计成功率 / 视觉扰动 / 延迟 / 消融基线。

契约:specs/VLA-TASK-SPEC.md 第 1.4 节(VLAEnv + success_func)与第 7 节(验收标准)。
本脚本只依赖第 1.4 节的接口,不 import sim/ 里的任何内部实现,也不修改 sim/。

用法:
    .venv/bin/python eval/closed_loop.py --ckpt runs/bc_v1/best.pt --episodes 50 --instrs 0-11

产物:eval/results/<name>.json,内含
  - 每个场景(基准 / 亮度x0.7 / 亮度x1.3 / 10%遮挡 / 固定灰图消融)下
    12 条指令各自的成功率与总体成功率
  - 单步推理延迟 p50/p95(bs=1,GPU 预热 20 次后测 100 次)与批量吞吐
  - 基线:random(均匀随机动作)、constant(恒零动作)
  - 与基准相比的视觉扰动掉点、灰图消融掉点

若 sim/env_vla.py 还没出现,可用内置合成假环境先自测链路:
    .venv/bin/python eval/closed_loop.py --ckpt ... --env-module mock --episodes 2
(mock 只用于自测,正式验收必须用默认的 sim.env_vla)
"""
from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import math
import os
import sys
import time

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

INSTRUCTIONS_DEFAULT = [
    "保持站立", "向前走", "向后退", "向左转", "向右转", "原地左旋", "原地右旋",
    "面向红桶", "靠近红桶", "靠近蓝桶", "靠近绿桶", "绕开红桶前往蓝桶",
]


# ---------------------------------------------------------------------------
# 环境适配层(spec 1.4)
# ---------------------------------------------------------------------------

def load_env_module(spec: str):
    """按 '包.模块' 或 '.py 路径' 或 'mock' 加载环境模块,返回 (VLAEnv, success_func)。"""
    if spec == "mock":
        mod = importlib.import_module("eval.synth_scene")
    elif spec.endswith(".py"):
        path = spec if os.path.isabs(spec) else os.path.join(PROJECT_ROOT, spec)
        name = "vla_eval_env_" + os.path.splitext(os.path.basename(path))[0]
        ispec = importlib.util.spec_from_file_location(name, path)
        if ispec is None or ispec.loader is None:
            raise ImportError(f"无法加载环境模块: {path}")
        mod = importlib.util.module_from_spec(ispec)
        sys.modules[name] = mod
        ispec.loader.exec_module(mod)
    else:
        mod = importlib.import_module(spec)
    env_cls = getattr(mod, "VLAEnv", None) or getattr(mod, "MockVLAEnv", None)
    if env_cls is None:
        raise AttributeError(f"{spec} 里没有 VLAEnv(sim-data 还没写好?)")
    success_func = getattr(mod, "success_func", None)
    return env_cls, success_func


def make_env(env_cls, max_steps):
    """兼容不同构造签名:只认 spec 要求的行为,构造参数做多种尝试。"""
    errors = []
    for kwargs in ({"max_steps": max_steps}, {"max_steps": max_steps, "seed": 0},
                   {}, {"seed": 0}):
        try:
            return env_cls(**kwargs)
        except TypeError as e:
            errors.append(f"{kwargs}: {e}")
            continue
    raise RuntimeError("VLAEnv 构造失败,接口与 spec 1.4 不一致: " + " | ".join(errors))


def reset_env(env, instr_id, seed):
    """兼容 reset(instr_id=, seed=) / reset(instr_id) / reset() 三种写法。"""
    for call in (
        lambda: env.reset(instr_id=instr_id, seed=seed),
        lambda: env.reset(instr_id, seed),
        lambda: env.reset(seed=seed),
        lambda: env.reset(),
    ):
        try:
            return call()
        except TypeError:
            continue
    raise RuntimeError("env.reset 调用失败,接口与 spec 1.4 不一致")


def judge_success(success_func, instr_id, info, history):
    """返回 (success, source)。优先用 env 自己给的 info['success'],否则调 sim 的 success_func。"""
    env_success = None
    if isinstance(info, dict) and info.get("success") is not None:
        env_success = bool(info["success"])
    func_success = None
    if success_func is not None:
        for args in ((instr_id, info, history), (instr_id, info)):
            try:
                func_success = bool(success_func(*args))
                break
            except TypeError:
                continue
    if env_success is not None:
        return env_success, "env_info"
    if func_success is not None:
        return func_success, "success_func"
    return False, "none"


# ---------------------------------------------------------------------------
# 图像扰动(spec 第 7 节)
# ---------------------------------------------------------------------------

def as_uint8_image(img):
    arr = np.asarray(img)
    if arr.dtype != np.uint8:
        if arr.dtype in (np.float32, np.float64) and arr.max() <= 1.5:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def apply_perturbation(img, kind, rng, size=128):
    kind = SCENARIO_ALIASES.get(kind, kind)
    if kind == "base":
        return img
    if kind == "bright_low":
        return np.clip(img.astype(np.float32) * 0.7, 0, 255).astype(np.uint8)
    if kind == "bright_high":
        return np.clip(img.astype(np.float32) * 1.3, 0, 255).astype(np.uint8)
    if kind == "occlusion":
        out = img.copy()
        side = int(math.sqrt(0.10) * size)
        y = int(rng.integers(0, max(1, size - side)))
        x = int(rng.integers(0, max(1, size - side)))
        out[y:y + side, x:x + side, :] = 128
        return out
    if kind == "gray":
        return np.full_like(img, 128)
    raise ValueError(f"未知扰动: {kind}")


SCENARIOS = [
    ("base", "基准"),
    ("bright_low", "亮度x0.7"),
    ("bright_high", "亮度x1.3"),
    ("occlusion", "10%遮挡"),
    ("gray", "固定灰图(视觉消融)"),
]

# 场景别名 -> 规范名(命令行可用 bright0.7 / bright1.3 / occlusion10 等写法)
SCENARIO_ALIASES = {
    "base": "base", "normal": "base",
    "bright0.7": "bright_low", "bright_low": "bright_low", "bright_07": "bright_low",
    "bright1.3": "bright_high", "bright_high": "bright_high", "bright_13": "bright_high",
    "occlusion10": "occlusion", "occlusion": "occlusion", "occ10": "occlusion",
    "gray": "gray", "grey": "gray", "gray_image": "gray",
}


# ---------------------------------------------------------------------------
# 策略
# ---------------------------------------------------------------------------

class ModelPolicy:
    def __init__(self, policy):
        self.policy = policy

    def act(self, images, instructions):
        return self.policy.act_batch(images, instructions)


class RandomPolicy:
    def __init__(self, v_max=0.35, w_max=1.0, seed=0):
        self.v_max, self.w_max = v_max, w_max
        self.rng = np.random.default_rng(seed)

    def act(self, images, instructions):
        n = len(images)
        out = np.zeros((n, 2), dtype=np.float32)
        out[:, 0] = self.rng.uniform(-self.v_max, self.v_max, size=n)
        out[:, 1] = self.rng.uniform(-self.w_max, self.w_max, size=n)
        return out


class ConstantPolicy:
    def __init__(self, v=0.0, w=0.0):
        self.v, self.w = v, w

    def act(self, images, instructions):
        n = len(images)
        out = np.zeros((n, 2), dtype=np.float32)
        out[:, 0] = self.v
        out[:, 1] = self.w
        return out


# ---------------------------------------------------------------------------
# 评测主循环
# ---------------------------------------------------------------------------

def evaluate_scenario(env_cls, success_func, policy, instr_ids, episodes, max_steps,
                      scenario, seed_base, batch_episodes, lat_records, env_pool=None,
                      action_ema=0.0):
    """跑一个场景:每条指令 episodes 局,回合之间复用 env_pool(避免每次重载 PPO/EGL)。

    env_pool 为 None 时内部创建一批环境并在结束时关闭;传入则复用外部环境池。
    """
    own_pool = env_pool is None
    if own_pool:
        env_pool = [make_env(env_cls, max_steps) for _ in range(batch_episodes)]
    per_instr = {}
    total_ok, total_n = 0, 0
    for instr_id in instr_ids:
        instruction = INSTRUCTIONS_DEFAULT[instr_id]
        seeds = [seed_base + instr_id * 100003 + i * 17 for i in range(episodes)]
        ok = 0
        outcomes = []
        for start in range(0, len(seeds), batch_episodes):
            chunk = seeds[start:start + batch_episodes]
            group = _run_group_with_success(env_cls, success_func, policy, instr_id,
                                            instruction, chunk, max_steps, scenario,
                                            rng_seed=instr_id * 1000 + start,
                                            scenario_name=scenario, lat_records=lat_records,
                                            env_pool=env_pool, action_ema=action_ema)
            ok += sum(1 for g in group if g["success"])
            outcomes.extend(group)
        reasons = {}
        for g in outcomes:
            if g.get("fail_reason"):
                reasons[g["fail_reason"]] = reasons.get(g["fail_reason"], 0) + 1
        per_instr[instr_id] = {
            "instruction": instruction,
            "episodes": len(seeds),
            "success": ok,
            "rate": ok / max(1, len(seeds)),
            "fail_reasons": reasons,
            "episodes_detail": outcomes,
        }
        total_ok += ok
        total_n += len(seeds)
        print(f"    instr {instr_id:2d} {instruction:<9s} {ok}/{len(seeds)} "
              f"({ok / max(1, len(seeds)) * 100:.0f}%)", flush=True)
    if own_pool:
        for env in env_pool:
            _close_env(env)
    return {"scenario": scenario,
            "per_instr": per_instr,
            "overall": {"success": total_ok, "episodes": total_n,
                        "rate": total_ok / max(1, total_n)}}


def _close_env(env):
    close = getattr(env, "close", None)
    if callable(close):
        try:
            close()
        except Exception:
            pass


def _run_group_with_success(env_cls, success_func, policy, instr_id, instruction, seeds,
                            max_steps, scenario, rng_seed, scenario_name, lat_records,
                            env_pool=None, action_ema=0.0):
    envs = []
    for i, s in enumerate(seeds):
        env = env_pool[i] if env_pool is not None else make_env(env_cls, max_steps)
        img = as_uint8_image(reset_env(env, instr_id, int(s)))
        envs.append({"env": env, "img": img, "done": False, "history": [], "info": {},
                     "ema": np.zeros(2, dtype=np.float32), "seed": int(s)})
    rng = np.random.default_rng(rng_seed)
    steps = 0
    while steps < max_steps + 5:
        active = [e for e in envs if not e["done"]]
        if not active:
            break
        imgs = [apply_perturbation(e["img"], scenario_name, rng) for e in active]
        t0 = time.perf_counter()
        actions = policy.act(imgs, [instruction] * len(active))
        dt = (time.perf_counter() - t0) * 1000.0
        lat_records.append((dt, len(active)))
        actions = np.asarray(actions, dtype=np.float32).reshape(len(active), 2)
        if action_ema and action_ema > 0.0:
            # 控制级一阶低通:a_t = ema * a_{t-1} + (1 - ema) * a_new
            for e, act in zip(active, actions):
                e["ema"] = action_ema * e["ema"] + (1.0 - action_ema) * act
                act[:] = e["ema"]
        for e, act in zip(active, actions):
            img, done, info = e["env"].step(act)
            e["img"] = as_uint8_image(img)
            e["info"] = info if isinstance(info, dict) else {}
            e["history"].append(e["info"])
            e["done"] = bool(done)
        steps += 1
        if all(e["done"] for e in envs):
            break
    out = []
    for e in envs:
        success, source = judge_success(success_func, instr_id, e["info"], e["history"])
        reason = None if success else failure_reason(instr_id, e["info"], e["history"])
        out.append({"seed": e.get("seed"), "success": bool(success), "steps": steps,
                    "judge_source": source, "fail_reason": reason})
        if env_pool is None:
            _close_env(e["env"])
    return out


def _criterion_quantities(info, history=None):
    """从 info(或历史里的 history 子字典)提取判据量 (disp, fwd, dyaw)。"""
    if isinstance(info, dict) and info.get("start_pos") is not None and info.get("pos") is not None:
        start = np.asarray(info["start_pos"], dtype=np.float64)
        pos = np.asarray(info["pos"], dtype=np.float64)
        delta = pos - start
        y0 = float(info.get("start_yaw", 0.0))
        return float(np.hypot(delta[0], delta[1])), \
            float(delta[0] * math.cos(y0) + delta[1] * math.sin(y0)), \
            float(info.get("cum_dyaw", 0.0))
    hist = info.get("history") if isinstance(info, dict) else None
    if not isinstance(hist, dict) and history:
        for h in reversed(history):
            if isinstance(h, dict) and isinstance(h.get("history"), dict):
                hist = h["history"]
                break
    if isinstance(hist, dict):
        return (float(hist.get("displacement", 0.0)),
                float(hist.get("forward_disp", 0.0)),
                float(hist.get("dyaw", 0.0)))
    return None


def failure_reason(instr_id, info, history=None):
    """失败回合的原因标签(与 spec 1.3 判据一一对应);成功返回 None。

    标签:fall / displaced / yaw_drift / forward_too_short / forward_too_long /
          backward_too_short / backward_too_long / yaw_not_in_range / drifted /
          heading_too_large / red_not_reached / blue_not_reached / green_not_reached /
          red_too_close / other。
    """
    if not isinstance(info, dict):
        return "other"
    if info.get("fall", False):
        return "fall"
    q = _criterion_quantities(info, history)
    if q is None:
        return "other"
    disp, fwd, dyaw = q
    dist = info.get("dist") or {}
    heading = info.get("heading_err_by_color") or {}

    def _d(color):
        try:
            return float(dist[color])
        except Exception:
            return 9.0

    def _h(color):
        try:
            return float(heading[color])
        except Exception:
            return 9.0

    if instr_id == 0:
        if disp < 0.10 and abs(dyaw) < 0.20:
            return None
        return "displaced" if disp >= 0.10 else "yaw_drift"
    if instr_id == 1:
        if 0.50 <= fwd <= 1.90:
            return None
        return "forward_too_short" if fwd < 0.50 else "forward_too_long"
    if instr_id == 2:
        if 0.35 <= (-fwd) <= 1.50:
            return None
        return "backward_too_short" if (-fwd) < 0.35 else "backward_too_long"
    if instr_id == 3:
        if 1.0 <= dyaw <= 3.5 and fwd > 0.15:
            return None
        return "yaw_not_in_range" if not (1.0 <= dyaw <= 3.5) else "forward_too_short"
    if instr_id == 4:
        if -3.5 <= dyaw <= -1.0 and fwd > 0.15:
            return None
        return "yaw_not_in_range" if not (-3.5 <= dyaw <= -1.0) else "forward_too_short"
    if instr_id == 5:
        if 1.5 <= dyaw <= 5.0 and disp < 0.40:
            return None
        return "yaw_not_in_range" if not (1.5 <= dyaw <= 5.0) else "drifted"
    if instr_id == 6:
        if -5.0 <= dyaw <= -1.5 and disp < 0.40:
            return None
        return "yaw_not_in_range" if not (-5.0 <= dyaw <= -1.5) else "drifted"
    if instr_id == 7:
        return None if abs(_h(0)) < 0.30 else "heading_too_large"
    if instr_id == 8:
        return None if _d(0) < 0.45 else "red_not_reached"
    if instr_id == 9:
        return None if _d(1) < 0.45 else "blue_not_reached"
    if instr_id == 10:
        return None if _d(2) < 0.45 else "green_not_reached"
    if instr_id == 11:
        if _d(1) >= 0.50:
            return "blue_not_reached"
        min_red = 9.0
        if history:
            for h in history:
                if isinstance(h, dict) and isinstance(h.get("dist"), dict):
                    try:
                        min_red = min(min_red, float(h["dist"][0]))
                    except (KeyError, TypeError, ValueError):
                        continue
        if min_red == 9.0:
            min_red = _d(0)
        return None if min_red > 0.40 else "red_too_close"
    return "other"


def latency_summary(lat_records):
    if not lat_records:
        return {}
    per_sample, per_call = [], []
    for dt, n in lat_records:
        per_call.append(dt)
        per_sample.append(dt / max(1, n))
    return {
        "calls": len(lat_records),
        "samples": int(sum(n for _, n in lat_records)),
        "per_sample_ms_p50": float(np.percentile(per_sample, 50)),
        "per_sample_ms_p95": float(np.percentile(per_sample, 95)),
        "per_call_ms_p50": float(np.percentile(per_call, 50)),
        "per_call_ms_p95": float(np.percentile(per_call, 95)),
    }


def parse_instrs(spec: str):
    ids = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            ids.extend(range(int(a), int(b) + 1))
        elif part:
            ids.append(int(part))
    return sorted(set(ids))


def main():
    ap = argparse.ArgumentParser(description="VLA 闭环评测(spec 第 7 节)")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--episodes", type=int, default=50, help="每条指令的回合数(基准场景)")
    ap.add_argument("--perturb-episodes", type=int, default=None,
                    help="视觉扰动场景每指令回合数(默认=--episodes)")
    ap.add_argument("--instrs", default="0-11")
    ap.add_argument("--env-module", default="sim.env_vla",
                    help="环境模块:sim.env_vla(默认)/ .py 路径 / mock(自测)")
    ap.add_argument("--scenarios", default="base,bright_low,bright_high,occlusion,gray",
                    help="逗号分隔;可用 base/bright_low/bright_high/occlusion/gray"
                         "(也接受 bright0.7/bright1.3/occlusion10 写法)")
    ap.add_argument("--batch-episodes", type=int, default=8,
                    help="并行回合数(批量推理,提速;显存不够就调小)")
    ap.add_argument("--max-steps", type=int, default=50)
    ap.add_argument("--seed-base", type=int, default=2026)
    ap.add_argument("--image-size", type=int, default=None, help="覆盖推理图像分辨率")
    ap.add_argument("--out-dir", default="eval/results")
    ap.add_argument("--name", default=None)
    ap.add_argument("--bench-baselines", dest="bench_baselines", action="store_true", default=False,
                    help="基线也按 --episodes 跑(默认基线每指令 10 局,省时间)")
    ap.add_argument("--baseline-episodes", type=int, default=10)
    ap.add_argument("--skip-baselines", action="store_true")
    ap.add_argument("--skip-latency", action="store_true")
    ap.add_argument("--skip-env-pool", action="store_true",
                    help="不复用环境池(调试用;每回合重载 env,很慢)")
    ap.add_argument("--action-ema", type=float, default=0.0,
                    help="动作一阶低通系数(0=关闭;0.5=对 (v,w) 做 a_t=ema*a_{t-1}+(1-ema)*a_new)")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"],
                    help="推理设备:auto=有 CUDA 用 cuda 否则 cpu(无 GPU 机器也能评测)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    name = args.name or os.path.splitext(os.path.basename(args.ckpt))[0]
    out_path = os.path.join(args.out_dir, name + ".json")

    env_cls, success_func = load_env_module(args.env_module)
    print(f"[env] {args.env_module}: VLAEnv={env_cls.__name__} "
          f"success_func={'有' if success_func else '无(用 info[success])'}")

    from vla.infer import VLAPolicy, resolve_device
    t0 = time.time()
    device = resolve_device(args.device)
    if device.type == "cpu" and args.batch_episodes > 1:
        print(f"[env] CPU 推理:--batch-episodes {args.batch_episodes} -> 1(CPU 批量推理太慢)")
        args.batch_episodes = 1
    policy = VLAPolicy.load(args.ckpt, device=device, image_size=args.image_size, verbose=True)
    load_s = time.time() - t0
    print(f"[policy] 加载 {args.ckpt} 用时 {load_s:.1f}s; 图像 {policy.cfg.image_size}px "
          f"seq={policy.cfg.image_seq_len} device={policy.device}")
    model_policy = ModelPolicy(policy)

    scenarios = [SCENARIO_ALIASES.get(s.strip(), s.strip())
                 for s in args.scenarios.split(",") if s.strip()]
    instr_ids = parse_instrs(args.instrs)
    result = {
        "ckpt": os.path.abspath(args.ckpt),
        "env_module": args.env_module,
        "episodes_per_instr": args.episodes,
        "instrs": instr_ids,
        "seed_base": args.seed_base,
        "batch_episodes": args.batch_episodes,
        "action_ema": float(args.action_ema),
        "image": {"size": policy.cfg.image_size, "seq_len": policy.cfg.image_seq_len},
        "device": str(policy.device),
        "batch_episodes_effective": args.batch_episodes,
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "scenarios": {},
        "baselines": {},
        "latency": {},
    }

    # 单步(bs=1)延迟:预热 20 次后测 100 次(与 vla/infer.py 一致)
    if not args.skip_latency:
        img = np.full((128, 128, 3), 128, dtype=np.uint8)
        lat = policy.benchmark(img, INSTRUCTIONS_DEFAULT[1], warmup=20, n=100)
        lat["bs"] = 1
        result["latency"]["bs1"] = lat
        print(f"[latency bs=1] p50={lat['p50_ms']:.1f}ms p95={lat['p95_ms']:.1f}ms (n={lat['n']})")

    # 一次性环境池:12 指令 x N 局复用同一批 env,避免每个回合重载 PPO/EGL(每 env 约 1.4 s)
    if not args.skip_env_pool:
        print(f"[env] 预创建 {args.batch_episodes} 个 VLAEnv 复用(每个约 1.4s)...", flush=True)
    t_pool = time.time()
    env_pool = None if args.skip_env_pool else [
        make_env(env_cls, args.max_steps) for _ in range(args.batch_episodes)]
    if env_pool is not None:
        print(f"[env] 环境池就绪 {len(env_pool)} 个,用时 {time.time() - t_pool:.1f}s", flush=True)

    remaining = [s for s in scenarios]
    for scenario in remaining:
        label = dict(SCENARIOS).get(scenario, scenario)
        n_eps = args.episodes if scenario == "base" else (
            args.perturb_episodes if args.perturb_episodes else args.episodes)
        print(f"[eval] 场景 {scenario} ({label}),每指令 {n_eps} 局 ...", flush=True)
        t1 = time.time()
        lat_records = []
        n_eps = args.episodes if scenario == "base" else (
            args.perturb_episodes if args.perturb_episodes else args.episodes)
        res = evaluate_scenario(env_cls, success_func, model_policy, instr_ids,
                                n_eps, args.max_steps, scenario,
                                args.seed_base, args.batch_episodes, lat_records,
                                env_pool=env_pool, action_ema=args.action_ema)
        res["elapsed_s"] = round(time.time() - t1, 1)
        res["latency"] = latency_summary(lat_records)
        result["scenarios"][scenario] = res
        print(f"[eval] {scenario}: 总体 {res['overall']['success']}/{res['overall']['episodes']} "
              f"= {res['overall']['rate'] * 100:.1f}%  ({res['elapsed_s']}s)", flush=True)

    # 基线(基准场景,同一批 seed)
    if not args.skip_baselines:
        baselines = {
            "random": RandomPolicy(policy.cfg.v_max, policy.cfg.w_max, seed=args.seed_base),
            "constant": ConstantPolicy(0.0, 0.0),
        }
        for bname, pol in baselines.items():
            print(f"[baseline] {bname} ...", flush=True)
            lat_records = []
            n_base = args.episodes if args.bench_baselines else args.baseline_episodes
            res = evaluate_scenario(env_cls, success_func, pol, instr_ids,
                                    n_base, args.max_steps, "base",
                                    args.seed_base, args.batch_episodes, lat_records,
                                    env_pool=env_pool, action_ema=args.action_ema)
            res["elapsed_s"] = None
            result["baselines"][bname] = res
            print(f"[baseline] {bname}: 总体 {res['overall']['rate'] * 100:.1f}%", flush=True)

    # 汇总:相对基准的掉点
    base = result["scenarios"].get("base")
    if base is not None:
        n_base = base["overall"]["episodes"]
        summary = {}
        unpaired = []
        for k, v in result["scenarios"].items():
            n_eps = v["overall"]["episodes"]
            paired = (n_eps == n_base)
            summary[k] = {
                "overall_rate": v["overall"]["rate"],
                "episodes": n_eps,
                "delta_pp_vs_base": ((v["overall"]["rate"] - base["overall"]["rate"]) * 100
                                     if paired else None),
            }
            if not paired:
                unpaired.append(f"{k}({n_eps} 局) vs base({n_base} 局)")
        result["summary"] = summary
        if unpaired:
            summary["note"] = ("以下场景与 base 的每指令局数不同,delta 置空、需另跑相同局数的 "
                               "base 做配对比较: " + "; ".join(unpaired))
        if "gray" in summary and summary["gray"]["delta_pp_vs_base"] is not None:
            result["vision_ablation_delta_pp"] = summary["gray"]["delta_pp_vs_base"]
        for bname, b in result["baselines"].items():
            summary["baseline_" + bname] = {"overall_rate": b["overall"]["rate"],
                                            "episodes": b["overall"]["episodes"]}

    if env_pool is not None:
        for env in env_pool:
            _close_env(env)
        env_pool = None

    result["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"[out] {out_path}")

    # 控制台汇总表
    print("\n=== 汇总 ===")
    for scenario, res in result["scenarios"].items():
        print(f"{scenario:14s} overall={res['overall']['rate'] * 100:5.1f}%  "
              + " ".join(f"{i}:{res['per_instr'][i]['rate'] * 100:4.0f}%"
                         for i in instr_ids))
    for bname, res in result["baselines"].items():
        print(f"{bname:14s} overall={res['overall']['rate'] * 100:5.1f}%")
    if "latency" in result and result["latency"].get("bs1"):
        lat = result["latency"]["bs1"]
        print(f"latency bs=1 p50={lat['p50_ms']:.1f}ms p95={lat['p95_ms']:.1f}ms")


if __name__ == "__main__":
    main()
