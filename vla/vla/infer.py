"""VLA 推理封装 + 延迟测试。

对外接口(spec 第 4 节):

    from vla.infer import VLAPolicy
    policy = VLAPolicy.load("runs/bc_v1/best.pt")
    v, w = policy.act(image_uint8_128x128x3, "靠近红桶")

延迟测试(10 Hz 闭环要求单步 < 100 ms):

    .venv/bin/python vla/infer.py --ckpt runs/bc_v1/best.pt --bench --n 100

推理加速(三个开关,默认全开,详见 vla/fast_infer.py 与 vla/README.md「推理优化」):
- merge_lora=True:加载后 `model.lm.merge_and_unload()` 把 LoRA 并进基座,
  只适用于推理/评测;**训练/续训必须显式传 merge_lora=False**。
- fast_proc=True:缓存 12 条指令的 input_ids + 快速图像预处理(与 processor 逐位一致)。
- cuda_graph=True:固定形状 CUDA graph(只在 fast_proc 且 CUDA 下有效)。

开关也可用环境变量覆盖(便于不改 eval 脚本做 A/B):
    VLA_MERGE_LORA=0 VLA_FAST_PROC=0 VLA_CUDA_GRAPH=0

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


def _env_flag(name: str, default: bool) -> bool:
    """环境变量开关:0/false/no/off 视为关闭,其余非空值为开启;未设置用 default。"""
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return bool(default)
    return raw.strip().lower() not in ("0", "false", "no", "off")


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
                 device: str | torch.device | None = "auto",
                 merge_lora: bool = False, fast_proc: bool = False,
                 cuda_graph: bool = False, verbose: bool = False):
        self.model = model
        self.processor = processor
        self.cfg = cfg
        self.extra = extra or {}
        self.device = resolve_device(device)
        self.model.eval()
        self.merge_lora = bool(merge_lora)
        self.verbose = bool(verbose)
        # -- 快速路径(失败自动回退 HF processor + 原始 forward)------------------
        self.fast = None
        self.runner = None
        self.fast_error = None
        self.graph_check: dict | None = None
        self._fast_pix_dtype = None
        if fast_proc:
            try:
                from vla.fast_infer import make_fast_backend

                self._fast_pix_dtype = next(model.parameters()).dtype
                self.fast, self.runner = make_fast_backend(
                    model, processor, cfg, device=self.device,
                    enable_graph=bool(cuda_graph), verbose=verbose)
                if cuda_graph and self.device.type == "cuda":
                    self.graph_check = self.runner.self_check(self.fast)
                    if verbose:
                        chk = self.graph_check
                        print(f"[VLAPolicy] CUDA graph 自检: ok={chk['ok']} "
                              f"max|diff|={chk.get('max_diff')} "
                              f"graph={chk.get('graph')} {chk.get('reason') or ''}")
            except Exception as exc:  # 快速路径整体不可用:静默回退
                self.fast = None
                self.runner = None
                self.fast_error = f"{type(exc).__name__}: {exc}"
                if verbose:
                    print(f"[VLAPolicy] 快速预处理不可用,回退 HF processor: {self.fast_error}")

    # -- 构造 ---------------------------------------------------------------
    @classmethod
    def load(cls, ckpt_path: str, device: str | torch.device | None = "auto",
             image_size: int | None = None, verbose: bool = False,
             merge_lora: bool | None = None, fast_proc: bool | None = None,
             cuda_graph: bool | None = None) -> "VLAPolicy":
        """加载 checkpoint。

        merge_lora : None=读环境变量 VLA_MERGE_LORA(默认 True);True=合并 LoRA(仅推理);
                     False=保留 LoRA(需要继续训练/续训时用)。
        fast_proc  : None=读 VLA_FAST_PROC(默认 True);快速预处理 + 快速前向。
        cuda_graph : None=读 VLA_CUDA_GRAPH(默认 True);固定形状 CUDA graph(需 fast_proc)。
        """
        merge_lora = _env_flag("VLA_MERGE_LORA", True) if merge_lora is None else bool(merge_lora)
        fast_proc = _env_flag("VLA_FAST_PROC", True) if fast_proc is None else bool(fast_proc)
        cuda_graph = _env_flag("VLA_CUDA_GRAPH", True) if cuda_graph is None else bool(cuda_graph)
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
        if merge_lora:
            t0 = time.time()
            model.lm.merge_and_unload()
            if verbose:
                print(f"[VLAPolicy] LoRA 已合并进基座(merge_and_unload,{time.time() - t0:.2f}s);"
                      f"该模型只能推理,不能继续训练")
        proc = build_processor(cfg.base_model, cfg.image_size, cfg.image_seq_len)
        return cls(model, proc, cfg, extra, device=device, merge_lora=merge_lora,
                   fast_proc=fast_proc, cuda_graph=cuda_graph, verbose=verbose)

    # -- 编码 / 前向(自动选择快速路径)----------------------------------------
    def _encode(self, images, instructions):
        if self.fast is not None:
            try:
                # bs=1 且走 CUDA graph 时,pixel_values 留 CPU,由 graph 输入 copy_ 直接 H2D
                graph_bs1 = (self.runner is not None and self.runner.enable_graph
                             and len(images) == 1)
                return self.fast.prepare(images, instructions,
                                         pixel_dtype=self._fast_pix_dtype,
                                         to_device=not graph_bs1)
            except Exception as exc:  # 图像尺寸/指令异常等:回退 processor 慢路径
                self.fast_error = f"{type(exc).__name__}: {exc}"
                if self.verbose:
                    print(f"[VLAPolicy] 快速预处理失败,回退 processor: {self.fast_error}")
                self.fast = None
                self.runner = None
        return prepare_inputs(self.processor, images, instructions, device=self.device)

    def _forward(self, enc):
        if self.runner is not None:
            return self.runner.run(enc), None
        return self.model(enc)

    # -- 推理 ---------------------------------------------------------------
    @torch.inference_mode()
    def act_batch(self, images, instructions) -> np.ndarray:
        """批量推理:返回物理动作 np.float32 [N,2] (v m/s, w rad/s)。"""
        if isinstance(instructions, str):
            instructions = [instructions] * len(images)
        enc = self._encode(list(images), list(instructions))
        pred_norm, _ = self._forward(enc)
        phys = denormalize(pred_norm.float().cpu(), self.cfg.v_max, self.cfg.w_max)
        return phys.numpy().astype(np.float32)

    def act(self, image, instruction: str):
        """单张图 + 单条指令 -> (v, w)。"""
        out = self.act_batch([image], [instruction])[0]
        return float(out[0]), float(out[1])

    def act_norm_batch(self, images, instructions) -> np.ndarray:
        if isinstance(instructions, str):
            instructions = [instructions] * len(images)
        enc = self._encode(list(images), list(instructions))
        with torch.inference_mode():
            pred_norm, _ = self._forward(enc)
        return pred_norm.float().cpu().numpy()

    # -- 元信息 --------------------------------------------------------------
    def info(self) -> dict:
        """当前生效的推理路径(写报告/JSON 用)。"""
        graph_ready = bool(self.runner is not None and self.runner.graph_ready)
        graph_err = self.runner.graph_error if self.runner is not None else self.fast_error
        return {
            "merge_lora": self.merge_lora,
            "fast_proc": self.fast is not None,
            "cuda_graph": graph_ready,
            "graph_error": graph_err,
            "fast_error": self.fast_error,
            "image_size": int(self.cfg.image_size),
            "image_seq_len": int(self.cfg.image_seq_len),
            "instr_max_len": int(self.fast.max_len) if self.fast is not None else None,
            "graph_count": int(self.runner.graph_count) if self.runner is not None else 0,
            "graph_max_diff": (float(self.runner.graph_max_diff)
                               if self.runner is not None and self.runner.graph_max_diff is not None
                               else None),
            "pad_len": int(self.fast.pad_len) if self.fast is not None else None,
            "device": str(self.device),
            "dtype": str(next(self.model.parameters()).dtype),
        }

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
        result = {
            "n": n,
            "p50_ms": _percentile(times, 50),
            "p95_ms": _percentile(times, 95),
            "mean_ms": float(np.mean(times)),
            "min_ms": float(np.min(times)),
            "max_ms": float(np.max(times)),
        }
        result.update({k: v for k, v in self.info().items()
                       if k in ("merge_lora", "fast_proc", "cuda_graph", "image_size")})
        return result

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
    ap.add_argument("--merge-lora", action=argparse.BooleanOptionalAction, default=None,
                    help="加载后合并 LoRA(默认 True;训练/续训用 --no-merge-lora)")
    ap.add_argument("--fast-proc", action=argparse.BooleanOptionalAction, default=None,
                    help="使用快速预处理/快速前向(默认 True)")
    ap.add_argument("--cuda-graph", action=argparse.BooleanOptionalAction, default=None,
                    help="使用固定形状 CUDA graph(默认 True,需 --fast-proc)")
    args = ap.parse_args()

    t0 = time.time()
    policy = VLAPolicy.load(args.ckpt, device=args.device, image_size=args.image_size,
                            verbose=True, merge_lora=args.merge_lora,
                            fast_proc=args.fast_proc, cuda_graph=args.cuda_graph)
    load_s = time.time() - t0
    cfg = policy.cfg
    rng = np.random.default_rng(0)
    img = (rng.random((128, 128, 3)) * 255).astype(np.uint8)
    if args.random_image:
        pass
    print(f"[load] {args.ckpt}  ckpt 前向配置: image={cfg.image_size}px "
          f"seq={cfg.image_seq_len} lora_r={cfg.lora_r}  device={policy.device}")
    print(f"[load] 加载耗时 {load_s:.1f}s, GPU 已分配 {torch.cuda.memory_allocated() / 2**20:.0f}MiB")
    print(f"[load] 推理路径: {policy.info()}")

    # 12 条指令各跑一次,确认指令 -> 动作通路有输出差异
    outs = policy.act_batch([img] * len(INSTRUCTIONS), INSTRUCTIONS)
    print("[sanity] 同一张随机图、12 条指令的动作输出(v, w):")
    for i, ins in enumerate(INSTRUCTIONS):
        print(f"   {i:2d} {ins:<10s} v={outs[i, 0]:+.3f} w={outs[i, 1]:+.3f}")

    result = {"ckpt": os.path.abspath(args.ckpt), "load_s": load_s,
              "image_size": cfg.image_size, "image_seq_len": cfg.image_seq_len,
              "infer_path": policy.info(),
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
