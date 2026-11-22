"""VLA 推理封装 + 延迟测试。

对外接口(spec 第 4 节):

    from vla.infer import VLAPolicy
    policy = VLAPolicy.load("runs/bc_v1/best.pt")
    v, w = policy.act(image_uint8_128x128x3, "靠近红桶")

延迟测试(10 Hz 闭环要求单步 < 100 ms):

    .venv/bin/python vla/infer.py --ckpt runs/bc_v1/best.pt --bench --n 100

本脚本不联网:基座模型从 assets/hf/SmolVLM-256M-Instruct 本地加载(local_files_only)。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vla.model import (  # noqa: E402
    INSTRUCTIONS,
    VLAConfig,
    build_processor,
    denormalize,
    load_model_from_checkpoint,
    prepare_inputs,
)


def resolve_device(device=None):
    """'auto'/None -> 有 CUDA 用 cuda,否则 cpu;也接受 'cuda'/'cpu'/torch.device。"""
    if device is None or (isinstance(device, str) and device.lower() == "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _percentile(values, q):
    if not values:
        return float("nan")
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


class VLAPolicy:
    """加载好的 VLA 策略:图像 + 指令 -> (v, w) 物理速度指令。"""

    def __init__(self, model, processor, cfg: VLAConfig, extra: dict | None = None,
                 device: str | torch.device | None = "auto"):
        self.model = model
        self.processor = processor
        self.cfg = cfg
        self.extra = extra or {}
        self.device = resolve_device(device)
        self.model.eval()

    # -- 构造 ---------------------------------------------------------------
    @classmethod
    def load(cls, ckpt_path: str, device: str | torch.device | None = "auto",
             image_size: int | None = None, verbose: bool = False) -> "VLAPolicy":
        device = resolve_device(device)
        model, extra = load_model_from_checkpoint(ckpt_path, device=device, verbose=verbose)
        cfg = model.cfg
        if image_size is not None and int(image_size) != cfg.image_size:
            # 允许推理时临时切换图像分辨率(必须同步 image_seq_len)
            cfg = VLAConfig.from_dict(cfg.to_dict())
            cfg.image_size = int(image_size)
            cfg.image_seq_len = VLAConfig.for_image_size(int(image_size)).image_seq_len
            if verbose:
                print(f"[VLAPolicy] 覆盖 image_size={cfg.image_size} seq={cfg.image_seq_len}")
        proc = build_processor(cfg.base_model, cfg.image_size, cfg.image_seq_len)
        return cls(model, proc, cfg, extra, device=device)

    # -- 推理 ---------------------------------------------------------------
    @torch.inference_mode()
    def act_batch(self, images, instructions) -> np.ndarray:
        """批量推理:返回物理动作 np.float32 [N,2] (v m/s, w rad/s)。"""
        if isinstance(instructions, str):
            instructions = [instructions] * len(images)
        enc = prepare_inputs(self.processor, list(images), list(instructions),
                             device=self.device)
        pred_norm, _ = self.model(enc)
        phys = denormalize(pred_norm.float().cpu(), self.cfg.v_max, self.cfg.w_max)
        return phys.numpy().astype(np.float32)

    def act(self, image, instruction: str):
        """单张图 + 单条指令 -> (v, w)。"""
        out = self.act_batch([image], [instruction])[0]
        return float(out[0]), float(out[1])

    def act_norm_batch(self, images, instructions) -> np.ndarray:
        if isinstance(instructions, str):
            instructions = [instructions] * len(images)
        enc = prepare_inputs(self.processor, list(images), list(instructions),
                             device=self.device)
        with torch.inference_mode():
            pred_norm, _ = self.model(enc)
        return pred_norm.float().cpu().numpy()

    # -- 延迟测试 ------------------------------------------------------------
    @torch.inference_mode()
    def benchmark(self, image, instruction: str, warmup: int = 20, n: int = 100) -> dict:
        """单步(bs=1)延迟:预热 warmup 次后测 n 次,返回 p50/p95/mean。"""
        for _ in range(warmup):
            self.act(image, instruction)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        times = []
        for _ in range(n):
            t = time.perf_counter()
            self.act(image, instruction)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t) * 1000.0)  # ms
        return {
            "n": n,
            "p50_ms": _percentile(times, 50),
            "p95_ms": _percentile(times, 95),
            "mean_ms": float(np.mean(times)),
            "min_ms": float(np.min(times)),
            "max_ms": float(np.max(times)),
        }

    @torch.inference_mode()
    def benchmark_batch(self, images, instructions, warmup: int = 10, n: int = 30) -> dict:
        for _ in range(warmup):
            self.act_batch(images, instructions)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        times = []
        for _ in range(n):
            t = time.perf_counter()
            self.act_batch(images, instructions)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t) * 1000.0)
        return {
            "batch": len(images),
            "n": n,
            "p50_ms": _percentile(times, 50),
            "p95_ms": _percentile(times, 95),
            "mean_ms": float(np.mean(times)),
        }


def main():
    ap = argparse.ArgumentParser(description="VLA 推理延迟测试 / 单帧试算")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--bench", action="store_true", help="跑 bs=1 延迟测试(预热 20 次,测 100 次)")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--batch", type=int, default=0, help=">0 时额外测批量推理吞吐")
    ap.add_argument("--image-size", type=int, default=None)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--out", default=None, help="延迟结果 JSON 输出路径")
    ap.add_argument("--random-image", action="store_true", help="用随机噪声图测")
    args = ap.parse_args()

    t0 = time.time()
    policy = VLAPolicy.load(args.ckpt, device=args.device, image_size=args.image_size,
                            verbose=True)
    load_s = time.time() - t0
    cfg = policy.cfg
    rng = np.random.default_rng(0)
    img = (rng.random((128, 128, 3)) * 255).astype(np.uint8)
    if args.random_image:
        pass
    print(f"[load] {args.ckpt}  ckpt 前向配置: image={cfg.image_size}px "
          f"seq={cfg.image_seq_len} lora_r={cfg.lora_r}  device={policy.device}")
    print(f"[load] 加载耗时 {load_s:.1f}s, GPU 已分配 {torch.cuda.memory_allocated() / 2**20:.0f}MiB")

    # 12 条指令各跑一次,确认指令 -> 动作通路有输出差异
    outs = policy.act_batch([img] * len(INSTRUCTIONS), INSTRUCTIONS)
    print("[sanity] 同一张随机图、12 条指令的动作输出(v, w):")
    for i, ins in enumerate(INSTRUCTIONS):
        print(f"   {i:2d} {ins:<10s} v={outs[i, 0]:+.3f} w={outs[i, 1]:+.3f}")

    result = {"ckpt": os.path.abspath(args.ckpt), "load_s": load_s,
              "image_size": cfg.image_size, "image_seq_len": cfg.image_seq_len,
              "sanity_actions": {INSTRUCTIONS[i]: [float(outs[i, 0]), float(outs[i, 1])]
                                 for i in range(len(INSTRUCTIONS))}}
    if args.bench:
        lat = policy.benchmark(img, "靠近红桶", warmup=args.warmup, n=args.n)
        result["latency_bs1"] = lat
        print(f"[latency bs=1] p50={lat['p50_ms']:.1f}ms p95={lat['p95_ms']:.1f}ms "
              f"mean={lat['mean_ms']:.1f}ms")
    if args.batch > 0:
        imgs = np.stack([img] * args.batch)
        ins = [INSTRUCTIONS[i % len(INSTRUCTIONS)] for i in range(args.batch)]
        latb = policy.benchmark_batch(imgs, ins)
        result["latency_batch"] = latb
        print(f"[latency bs={args.batch}] p50={latb['p50_ms']:.1f}ms "
              f"p95={latb['p95_ms']:.1f}ms ({latb['mean_ms'] / args.batch:.1f}ms/样本)")

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"[out] {args.out}")


if __name__ == "__main__":
    main()
