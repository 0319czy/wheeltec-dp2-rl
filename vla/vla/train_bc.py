"""行为克隆训练脚本:SmolVLM-256M + LoRA + 动作头(BC, MSE)。

用法(见 specs/VLA-TASK-SPEC.md 第 9 节):

    .venv/bin/python vla/train_bc.py --data data/dataset_v1 --out runs/bc_v1 --epochs 6 --bs 16

数据格式(spec 第 3 节):data/<name>/shards/*.npz(images/actions/instr_ids/
episode_ids/step_idx/scene_seeds)+ 可选 index.jsonl。划分:scene_seed % 10 的 0-7 训练、8-9 验证。

显存/耗时:每 --log-every 步打印 loss、单步耗时、样本/秒、GPU 峰值显存;
训练结束写 <out>/train_log.json,并按验证集 action MSE 保存 best.pt(只含 LoRA + 动作头)。
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
import time

import numpy as np
import torch
from collections import OrderedDict

from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vla.model import (  # noqa: E402
    ACTION_V_MAX,
    ACTION_W_MAX,
    INSTRUCTIONS,
    VLAConfig,
    VLAModel,
    build_processor,
    load_action_head_partial,
    normalize_actions,
    prepare_inputs,
    save_checkpoint,
)

# 距离 prompt(可选):把"目标距离"拼到指令后面,用于验证"给了测距后 VLA 能到什么水平"
DIST_PROMPT = False
TARGET_COLOR = {7: 0, 8: 0, 9: 1, 10: 2, 11: 1}   # 指令 -> 目标颜色(0 红 / 1 蓝 / 2 绿)


def instruction_with_distance(text: str, instr_id: int, aux_row) -> str:
    """aux_row = [dx_red,dy_red,dx_blue,dy_blue,dx_green,dy_green](m)。"""
    k = TARGET_COLOR.get(int(instr_id))
    if k is None or aux_row is None:
        return text
    dx, dy = float(aux_row[2 * k]), float(aux_row[2 * k + 1])
    d = (dx * dx + dy * dy) ** 0.5
    return "%s｜目标距离%.2fm" % (text, d)


# DataLoader worker 里懒加载的 processor(父进程设置配置,fork 后子进程继承)
_WORKER_CTX: dict = {}


def _get_worker_processor():
    proc = _WORKER_CTX.get("processor")
    if proc is None:
        proc = build_processor(_WORKER_CTX["model_path"], _WORKER_CTX["image_size"],
                               _WORKER_CTX["image_seq_len"])
        _WORKER_CTX["processor"] = proc
    return proc


def _augment_image(img, rng):
    """训练期轻量增强:亮度 ±20% + 10% 遮挡块(对应 spec 第 7 节的视觉扰动)。

    只做亮度和遮挡,绝不做水平翻转(会破坏"向左转/向右转"的语义)。
    """
    arr = np.asarray(img).astype(np.float32)
    if rng.random() < 0.5:
        arr = arr * rng.uniform(0.80, 1.20)
    if rng.random() < 0.15:
        h, w = arr.shape[0], arr.shape[1]
        side = max(1, int(math.sqrt(0.10) * min(h, w)))
        y = int(rng.integers(0, max(1, h - side)))
        x = int(rng.integers(0, max(1, w - side)))
        arr[y:y + side, x:x + side, :] = rng.uniform(0.0, 255.0)
    return np.clip(arr, 0, 255).astype(np.uint8)


def _collate_impl(items, augment):
    """在 worker 进程里做图像/文本预处理(CPU 并行),主进程只做 GPU 前向。"""
    proc = _get_worker_processor()
    if augment:
        rng = _WORKER_CTX.get("aug_rng")
        if rng is None:
            rng = np.random.default_rng(int(time.time() * 1000) % 100000)
            _WORKER_CTX["aug_rng"] = rng
        aug_items = []
        flip_enabled = bool(_WORKER_CTX.get("augment_flip"))
        for it in items:
            # 同一回合的多帧必须"同一次翻转/同一种颜色扰动决策"保持一致;亮度/遮挡逐帧独立
            flip = bool(flip_enabled and rng.random() < 0.5)
            frames = []
            for fr in it.get("images", [it["image"]]):
                fr2 = _augment_image(fr, rng)
                if flip:
                    fr2 = np.ascontiguousarray(fr2[:, ::-1, :])
                frames.append(fr2)
            new_it = dict(it, images=frames, image=frames[-1])
            if flip and int(it["instr_id"]) in (1, 7, 8, 9, 10, 11):
                a = np.asarray(it["action"]).copy()
                a[1] = -a[1]
                new_it["action"] = a
            aug_items.append(new_it)
        items = aug_items
    images = [it.get("images", [it["image"]]) for it in items]
    instr_texts = [INSTRUCTIONS[int(it["instr_id"])] for it in items]
    if DIST_PROMPT:
        instr_texts = [instruction_with_distance(t, it["instr_id"], it.get("aux"))
                       for t, it in zip(instr_texts, items)]
    enc = prepare_inputs(proc, images, instr_texts, device=None)
    actions = torch.from_numpy(np.stack([it["action"] for it in items]).astype(np.float32))
    meta = {
        "instr_id": np.array([it["instr_id"] for it in items], dtype=np.int16),
        "episode_id": np.array([it["episode_id"] for it in items], dtype=np.int32),
        "scene_seed": np.array([it["scene_seed"] for it in items], dtype=np.int32),
        "step_idx": np.array([it["step_idx"] for it in items], dtype=np.int16),
    }
    out = {"enc": enc, "actions": actions, "meta": meta}
    aux_list = [it.get("aux") for it in items]
    if any(a is not None for a in aux_list):
        dim = len(next(a for a in aux_list if a is not None))
        aux = np.zeros((len(items), dim), dtype=np.float32)
        mask = np.zeros((len(items),), dtype=np.float32)
        for i, a in enumerate(aux_list):
            if a is not None:
                aux[i] = a
                mask[i] = 1.0
        out["aux"] = torch.from_numpy(aux)
        out["aux_mask"] = torch.from_numpy(mask)
    return out


def collate_batch(items):
    """训练用 collate:带亮度/遮挡增强。"""
    return _collate_impl(items, bool(_WORKER_CTX.get("augment")))


def collate_eval(items):
    """验证/评测用 collate:不做任何增强。"""
    return _collate_impl(items, False)


class ShardDataset(Dataset):
    """按 (shard, 行号) 索引的 npz 数据集。

    未压缩 npz 用 mmap 惰性读取(不占常驻内存);压缩 npz 回退为整 shard 载入内存
    (每个 shard 约 120 MB,只缓存最近 1-3 个)。
    """

    def __init__(self, shard_paths, indices, cache_size=3, frames=1, frame_stride=1):
        self.shard_paths = list(shard_paths)
        self.indices = indices            # [N,2] int64:(shard 下标, 行号)
        self.cache_size = int(cache_size)
        self.frames = max(1, int(frames))
        self.frame_stride = max(1, int(frame_stride))
        self._cache = OrderedDict()

    def __len__(self):
        return int(self.indices.shape[0])

    def _npz(self, shard_id):
        """载入一个 shard 的全部数组到内存并缓存。

        注意:np.load(npz, mmap_mode="r") 返回的是 NpzFile,**不做内存映射**,
        每次 d["images"] 都会重读整个 images 数组(122 MB/shard,约 25 ms/样本)。
        因此这里一次性物化成 dict(每 shard 约 123 MB),并做 LRU 缓存;
        配合"按 shard 分组打乱"的 epoch 顺序,每个 shard 每 epoch 只读一次。
        """
        key = int(shard_id)
        d = self._cache.get(key)
        if d is not None:
            self._cache.move_to_end(key)
            return d
        path = self.shard_paths[key]
        raw = np.load(path)
        d = {k: np.asarray(raw[k]) for k in raw.files}
        raw.close()
        self._cache[key] = d
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return d

    def __getitem__(self, i):
        shard_id, row = int(self.indices[i, 0]), int(self.indices[i, 1])
        d = self._npz(shard_id)
        cur_img = d["images"][row]
        frames = [cur_img]
        if self.frames > 1:
            # 稀疏历史窗:取 [t-(frames-1)*stride, ..., t-2*stride, t-stride, t]
            # 只在同一 episode 内取帧(shard 内 episode 的帧连续存放);越界则重复本 episode 首帧
            step = int(d["step_idx"][row])
            ep = int(d["episode_ids"][row])
            ep_first = row - step
            if ep_first < 0 or int(d["episode_ids"][ep_first]) != ep:
                ep_first = row       # 极端兜底:第一个可用帧就是当前帧
            frames = []
            for k in range(self.frames - 1, -1, -1):
                off = k * self.frame_stride
                idx = row - off
                if off <= 0:
                    idx = row
                elif idx < 0 or int(d["episode_ids"][idx]) != ep or off > step:
                    idx = ep_first
                frames.append(d["images"][idx])
        item = {
            "images": frames,
            "image": cur_img,
            "action": np.asarray(d["actions"][row], dtype=np.float32),
            "instr_id": int(d["instr_ids"][row]),
            "episode_id": int(d["episode_ids"][row]),
            "step_idx": int(d["step_idx"][row]),
            "scene_seed": int(d["scene_seeds"][row]),
        }
        if "aux" in d:
            item["aux"] = np.asarray(d["aux"][row], dtype=np.float32)
        return item


def _shard_num_frames(path):
    """只读 npz 里 images.npy 的头部拿帧数,不解压 122 MB 的图像数组。

    np.load(npz)["images"].shape 会解压整个成员;52 个 shard 就是 6.4 GB 无用解压。
    """
    import zipfile
    with zipfile.ZipFile(path) as z:
        with z.open("images.npy") as f:
            version = np.lib.format.read_magic(f)
            if version == (1, 0):
                shape, _fortran, _dtype = np.lib.format.read_array_header_1_0(f)
            else:
                shape, _fortran, _dtype = np.lib.format.read_array_header_2_0(f)
    return int(shape[0])


def build_index(data_dir, limit_frames=None, index_jsonl=None):
    """扫描一个或多个数据目录的 shards/*.npz,返回 (shard_paths, indices[N,2], 元数据数组)。

    data_dir 可以是逗号分隔的多个目录(例如 "data/dataset_v1,data/dataset_obj")。
    """
    dirs = [os.path.abspath(d.strip()) for d in str(data_dir).split(",") if d.strip()]
    shard_paths = []
    shard_dirs = []
    for d in dirs:
        found = sorted(glob.glob(os.path.join(d, "shards", "*.npz")))
        if not found:
            found = sorted(glob.glob(os.path.join(d, "*.npz")))
        if not found:
            raise FileNotFoundError(f"在 {d} 下找不到 shards/*.npz(数据还没生成?)")
        shard_paths.extend(found)
        shard_dirs.extend([d] * len(found))

    rows = []
    offsets = []
    seeds, instrs, steps, eps = [], [], [], []
    sources = []
    for si, path in enumerate(shard_paths):
        n = _shard_num_frames(path)
        d = np.load(path)
        offsets.append(len(rows))
        seeds.append(np.asarray(d["scene_seeds"]))
        instrs.append(np.asarray(d["instr_ids"]))
        steps.append(np.asarray(d["step_idx"]))
        eps.append(np.asarray(d["episode_ids"]))
        rows.extend([(si, r) for r in range(n)])
        sources.append(np.full(n, dirs.index(shard_dirs[si]), dtype=np.int16))
        d.close()
    indices = np.asarray(rows, dtype=np.int64)
    scene_seed = np.concatenate(seeds).astype(np.int64)
    instr_id = np.concatenate(instrs).astype(np.int64)
    step_idx = np.concatenate(steps).astype(np.int64)
    episode_id = np.concatenate(eps).astype(np.int64)
    source = np.concatenate(sources).astype(np.int64)

    if limit_frames is not None and limit_frames < len(indices):
        # 均匀抽样,保持场景/指令分布(仅用于快速冒烟)
        pick = np.linspace(0, len(indices) - 1, int(limit_frames)).astype(np.int64)
        indices = indices[pick]
        scene_seed, instr_id = scene_seed[pick], instr_id[pick]
        step_idx, episode_id = step_idx[pick], episode_id[pick]
        source = source[pick]
    return shard_paths, indices, scene_seed, instr_id, step_idx, episode_id, source


def make_train_val(data_dir, limit_frames=None, val_ratio_seeds=True):
    (shard_paths, indices, scene_seed, instr_id,
     step_idx, episode_id, source) = build_index(data_dir, limit_frames)
    if val_ratio_seeds:
        # spec 第 3 节:scene_seed % 10 -> 0-7 训练、8-9 验证
        mod = scene_seed % 10
        train_mask = mod <= 7
        val_mask = mod >= 8
        if val_mask.sum() == 0:  # 数据太少(如 smoke 只有几个 seed)时按 8:2 随机分
            rng = np.random.default_rng(0)
            perm = rng.permutation(len(indices))
            n_val = max(1, int(len(indices) * 0.2))
            val_mask = np.zeros(len(indices), dtype=bool)
            val_mask[perm[:n_val]] = True
            train_mask = ~val_mask
    else:
        perm = np.random.default_rng(0).permutation(len(indices))
        n_val = max(1, int(len(indices) * 0.2))
        val_mask = np.zeros(len(indices), dtype=bool)
        val_mask[perm[:n_val]] = True
        train_mask = ~val_mask
    stats = {
        "data_dir": [os.path.abspath(d) for d in str(data_dir).split(",") if d.strip()],
        "num_shards": len(shard_paths),
        "frames_total": int(len(indices)),
        "frames_train": int(train_mask.sum()),
        "frames_val": int(val_mask.sum()),
        "instr_frames": {int(k): int((instr_id == k).sum()) for k in np.unique(instr_id)},
    }
    return (shard_paths, indices, scene_seed, instr_id, step_idx, episode_id, source,
            train_mask, val_mask, stats)


def parse_instr_weights(spec):
    """'7:2,8:2,11:4' -> {7: 2.0, 8: 2.0, 11: 4.0}。"""
    weights = {}
    if not spec:
        return weights
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            k, v = part.split(":")
        else:
            k, v = part, "1"
        weights[int(k)] = float(v)
    return weights


def weighted_epoch_indices(indices, instr_ids, weights, rng, balance=False, total=None,
                           row_weights=None):
    """按指令重采样一个 epoch 的索引(可复现),返回 [N',2] 且按 shard 分组。

    - balance=True: 每条指令先按"帧数相同"为基准,再乘 weights(反制长尾);
      balance=False: 以原始帧数为基准,只乘 weights。
    - total: 目标 epoch 帧数(默认 = 原训练帧数,保证每 epoch 训练量不变)。
    """
    n = len(indices)
    if n == 0:
        return indices
    total = int(total) if total else n
    instr_ids = np.asarray(instr_ids)
    uniq = np.unique(instr_ids)
    base = {int(i): float((instr_ids == i).sum()) for i in uniq}
    if balance:
        mean_n = float(np.mean(list(base.values())))
        base = {k: mean_n for k in base}
    raw = {k: base[k] * float(weights.get(k, 1.0)) for k in base}
    total_w = float(sum(raw.values()))
    target = {k: max(1, int(round(total * raw[k] / total_w))) for k in raw}
    chunks = []
    for i in sorted(target):
        rows = indices[instr_ids == i]
        if len(rows) == 0:
            continue
        if row_weights is not None:
            rw = np.asarray(row_weights)[instr_ids == i].astype(np.float64)
            prob = rw / rw.sum()
            pick = rng.choice(len(rows), size=target[i], replace=True, p=prob)
        else:
            pick = rng.integers(0, len(rows), size=target[i])
        chunks.append(rows[pick])
    if not chunks:
        return indices, {int(k): int(v) for k, v in target.items()}
    out = np.concatenate(chunks, axis=0)
    # 再按 shard 分组打乱(保证每个 shard 每 epoch 只读一次)
    order = rng.permutation(len(out))
    out = out[order]
    shard_ids = np.unique(out[:, 0])
    rng.shuffle(shard_ids)
    grouped = []
    for sid in shard_ids:
        rows = out[out[:, 0] == sid]
        grouped.append(rows[rng.permutation(len(rows))])
    return np.concatenate(grouped, axis=0), {int(k): int(v) for k, v in target.items()}


def shuffle_by_shard(indices, rng):
    """把索引按 shard 分组:先随机 shard 顺序,再 shard 内随机行序。

    目的:每个 shard 只需载入一次(123 MB),而不是每取一个样本就读一遍。
    batch 内仍是同一场景附近的数据,随机性由 shard 内打乱保证。
    """
    if len(indices) == 0:
        return indices
    shard_ids = np.unique(indices[:, 0])
    rng.shuffle(shard_ids)
    chunks = []
    for s in shard_ids:
        rows = indices[indices[:, 0] == s]
        perm = rng.permutation(len(rows))
        chunks.append(rows[perm])
    return np.concatenate(chunks, axis=0)


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def lr_at(step, total_steps, base_lr, warmup_frac=0.03, min_frac=0.1):
    warmup = max(1, int(total_steps * warmup_frac))
    if step < warmup:
        return base_lr * (step + 1) / warmup
    prog = (step - warmup) / max(1, total_steps - warmup)
    cos = 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))
    return base_lr * (min_frac + (1.0 - min_frac) * cos)


@torch.no_grad()
def evaluate(model, loader, device, v_max, w_max, max_batches=None):
    model.eval()
    norm_sq = 0.0
    phys_sq = torch.zeros(2, dtype=torch.float64)
    n = 0
    per_instr = {}
    for bi, batch in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break
        enc = {k: v.to(device) for k, v in batch["enc"].items()}
        target = normalize_actions(batch["actions"], v_max, w_max).to(device)
        aux_sq_sum, aux_n = None, 0
        if getattr(model.cfg, "aux_dim", 0) > 0 and "aux" in batch:
            pred, _feat, aux_pred = model(enc, return_aux=True)
            aux_t = (batch["aux"].to(device) / model.cfg.aux_scale)
            aux_m = batch["aux_mask"].to(device)
            aux_sq = ((aux_pred.float() - aux_t.float()) ** 2) * aux_m[:, None]
            aux_sq_sum = aux_sq.sum(dim=0).double()
            aux_n = int(aux_m.sum().item())
        else:
            pred, _ = model(enc)
        diff = pred.float() - target.float()
        norm_sq += float((diff ** 2).sum())
        phys_diff = diff.clone()
        phys_diff[:, 0] *= v_max
        phys_diff[:, 1] *= w_max
        phys_sq += (phys_diff ** 2).sum(dim=0).double().cpu()
        n += diff.shape[0]
        for j, iid in enumerate(batch["meta"]["instr_id"]):
            v = per_instr.setdefault(int(iid), [0.0, 0])
            v[0] += float((diff[j] ** 2).sum())
            v[1] += 1
    if n == 0:
        return {"n": 0}
    res = {
        "n": n,
        "mse_norm": norm_sq / (2 * n),
        "mse_v": float(phys_sq[0] / n),
        "mse_w": float(phys_sq[1] / n),
        "mse_phys_mean": float((phys_sq[0] + phys_sq[1]) / (2 * n)),
    }
    res["per_instr_mse_norm"] = {str(k): v[0] / (2 * v[1]) for k, v in sorted(per_instr.items())}
    if aux_sq_sum is not None and aux_n > 0:
        res["aux_mse_m"] = [float(x / aux_n) for x in aux_sq_sum]
    model.train()
    return res


def main():
    ap = argparse.ArgumentParser(description="SmolVLM-256M 行为克隆训练")
    ap.add_argument("--data", required=True,
                    help="数据目录(含 shards/*.npz);可用逗号拼接多个目录,如 "
                         "data/dataset_v1,data/dataset_obj")
    ap.add_argument("--instr-weights", default=None,
                    help="按指令加权采样,例如 \"7:2,8:2,9:2,10:2,11:4\"(默认全 1)")
    ap.add_argument("--balance-instr", action="store_true",
                    help="先按指令帧数均衡(每条指令等量)再乘 --instr-weights")
    ap.add_argument("--source-weights", default=None,
                    help="按数据目录加权采样,例如 \"data/dataset_dagger:6\"(可逗号分隔多个);"
                         "只影响加权采样的行选择概率")
    ap.add_argument("--epoch-frames", type=int, default=None,
                    help="加权采样时每个 epoch 的帧数上限(默认=原训练帧数);用于控制单 epoch 时长")
    ap.add_argument("--out", required=True, help="输出目录(ckpt + train_log.json)")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--image-size", type=int, default=256, choices=[128, 192, 256, 320, 384, 448, 512])
    ap.add_argument("--frames", type=int, default=1, choices=[1, 2, 4],
                    help="每个样本输入的历史帧数(1=单帧;2=(t-1,t);4=稀疏历史窗,prompt 里 4 个 <image>)")
    ap.add_argument("--frame-stride", type=int, default=1,
                    help="历史帧步距(1=相邻;2=稀疏窗,frames=4 时取 [t-6,t-4,t-2,t])")
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.0)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--cache-shards", type=int, default=3, help="每个 worker 缓存多少个 shard(每个约 123 MB)")
    ap.add_argument("--no-shard-shuffle", dest="shard_shuffle", action="store_false", default=True,
                    help="关闭按 shard 分组打乱(退化为全局随机,小数据集可关)")
    ap.add_argument("--limit-frames", type=int, default=None, help="均匀抽样帧数上限(冒烟用)")
    ap.add_argument("--max-steps", type=int, default=None, help="每 epoch 最大步数(调试用)")
    ap.add_argument("--val-batches", type=int, default=None)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init-ckpt", default=None,
                    help="从已有 checkpoint 热启动(LoRA+动作头;跨图像分辨率也兼容)")
    ap.add_argument("--resume", default=None,
                    help="从 <out>/ep{N}.pt 断点续训(恢复 LoRA/动作头/优化器状态/epoch/LR 进度)")
    ap.add_argument("--reset-progress", action="store_true",
                    help="配合 --resume/--init-ckpt:只加载权重,epoch/global_step/优化器清零"
                         "(用于换数据集开新一轮微调)")
    ap.add_argument("--amp", choices=["bf16", "fp32"], default="bf16",
                    help="bf16=模型权重/计算走 bf16(默认);fp32 仅调试")
    ap.add_argument("--no-lora", action="store_true", help="不挂 LoRA(只训动作头,速度基线)")
    ap.add_argument("--patience", type=int, default=3, help="验证集连续多少个 epoch 不降就早停")
    ap.add_argument("--stop-after-epoch", type=int, default=None,
                    help="跑完第 N 个 epoch 就正常退出(用于中途评测;之后用 --resume 续训,"
                         "LR 总步数仍按 --epochs 算,不会打乱 schedule)")
    ap.add_argument("--augment", dest="augment", action="store_true", default=True,
                    help="训练期亮度/遮挡增强(默认开)")
    ap.add_argument("--no-augment", dest="augment", action="store_false")
    ap.add_argument("--augment-flip", action="store_true", default=False,
                    help="训练期水平镜像增强(图像翻转 + ω 取反,任务在镜像下等变)")
    ap.add_argument("--aux-loss-weight", type=float, default=0.0,
                    help="辅助回归头(桶相对位置 [dx,dy]x3)的 loss 权重;0=关闭。"
                         "需要数据 shard 里有 aux 键(gen_dataset.py --aux)")
    ap.add_argument("--dist-prompt", action="store_true",
                    help="指令后附目标距离 x.xxm(需要 shard 里有 aux;测距传感器可行性实验)")
    ap.add_argument("--aux-condition", action="store_true",
                    help="两阶段:动作头额外吃 aux 预测(桶相对位置),需同时 >0 的 --aux-loss-weight")
    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        raise SystemExit("需要 GPU(本项目只支持 CUDA 训练/推理)")
    torch.backends.cuda.matmul.allow_tf32 = True

    # ---------------- 数据 ----------------
    (shard_paths, indices, scene_seed, instr_id, step_idx, episode_id, source,
     train_mask, val_mask, data_stats) = make_train_val(args.data, args.limit_frames)
    train_idx = indices[train_mask]
    train_instr_ids = instr_id[train_mask]
    train_source = source[train_mask]
    # 按数据目录加权(DAgger 数据占比低时抬权重)
    dirs_abs = [os.path.abspath(d.strip()) for d in str(args.data).split(",") if d.strip()]
    src_weight_map = {}
    if args.source_weights:
        for part in str(args.source_weights).split(","):
            part = part.strip()
            if not part:
                continue
            if ":" not in part:
                continue
            k, v = part.rsplit(":", 1)
            key = os.path.abspath(k.strip())
            if key not in dirs_abs:
                print(f"[source][warn] {key} 不在 --data 里,忽略")
                continue
            src_weight_map[dirs_abs.index(key)] = float(v)
    train_row_weights = np.ones(len(train_idx), dtype=np.float64)
    for di, mult in src_weight_map.items():
        train_row_weights[train_source == di] = mult
    if src_weight_map:
        print(f"[source] 目录权重: { {dirs_abs[d]: w for d, w in src_weight_map.items()} }; "
              f"各目录训练帧数: "
              f"{ {dirs_abs[d]: int((train_source == d).sum()) for d in range(len(dirs_abs))} }")
    val_idx = indices[val_mask]
    train_ds = ShardDataset(shard_paths, train_idx, cache_size=args.cache_shards,
                            frames=args.frames, frame_stride=args.frame_stride)
    val_ds = ShardDataset(shard_paths, val_idx, cache_size=args.cache_shards,
                          frames=args.frames, frame_stride=args.frame_stride)
    _WORKER_CTX.update({
        "model_path": VLAConfig().base_model,
        "image_size": args.image_size,
        "image_seq_len": VLAConfig.for_image_size(args.image_size).image_seq_len,
        "augment": bool(args.augment),
        "augment_flip": bool(args.augment_flip),
    })
    if not args.no_lora:
        _WORKER_CTX["image_seq_len"] = VLAConfig.for_image_size(args.image_size).image_seq_len
    def make_loader(ds, train_mode):
        return DataLoader(
            ds, batch_size=args.bs,
            shuffle=False,
            num_workers=args.workers,
            collate_fn=collate_batch if train_mode else collate_eval,
            drop_last=len(ds) > args.bs,
            persistent_workers=False,
            prefetch_factor=4 if args.workers > 0 else None,
        )

    train_loader = make_loader(train_ds, True)
    val_loader = make_loader(val_ds, False)

    # ---------------- 模型 ----------------
    cfg = VLAConfig.for_image_size(args.image_size)
    cfg.frames = int(args.frames)
    cfg.frame_stride = int(args.frame_stride)
    global DIST_PROMPT
    DIST_PROMPT = bool(getattr(args, "dist_prompt", False))
    if DIST_PROMPT:
        cfg.dist_prompt = True
    if args.aux_loss_weight and args.aux_loss_weight > 0:
        cfg.aux_dim = 6
        cfg.aux_scale = 3.0
    if getattr(args, "aux_condition", False):
        if cfg.aux_dim <= 0:
            raise SystemExit("--aux-condition 需要同时给 --aux-loss-weight > 0")
        cfg.aux_condition_action = True
    if args.no_lora:
        cfg.lora_r = 0
    cfg.lora_alpha = args.lora_alpha
    cfg.lora_dropout = args.lora_dropout
    t_load = time.time()
    if args.no_lora:
        # 只训练动作头:先建一个 r=1 的 LoRA 再全冻结(peft 不支持 r=0)
        cfg.lora_r = 1
        model = VLAModel(cfg, verbose=True)
        for p in model.lm.parameters():
            p.requires_grad = False
        for p in model.action_head.parameters():
            p.requires_grad = True
    else:
        cfg.lora_r = args.lora_r
        model = VLAModel(cfg, verbose=True)
    start_epoch = 0
    global_step = 0
    best_mse = float("inf")
    bad_epochs = 0
    resume_extra = {}
    if args.init_ckpt:
        payload = torch.load(args.init_ckpt, map_location="cpu", weights_only=False)
        model.load_lora_state(payload["lora_state"])
        skipped = load_action_head_partial(model, payload["head_state"])
        if skipped:
            print(f"[init][warn] 动作头跳过的键: {skipped}")
        if payload.get("aux_state") and getattr(model, "aux_head", None) is not None:
            model.aux_head.load_state_dict(payload["aux_state"])
        print(f"[init] 从 {args.init_ckpt} 热启动 LoRA+动作头 "
              f"({len(payload['lora_state'])} 个 LoRA 张量)")
    if args.resume:
        payload = torch.load(args.resume, map_location="cpu", weights_only=False)
        rcfg = payload.get("config", {})
        if int(rcfg.get("image_size", -1)) != int(cfg.image_size):
            print(f"[resume][warn] checkpoint image_size={rcfg.get('image_size')} "
                  f"与本次 {cfg.image_size} 不同,按 LoRA 权重直接迁移")
        model.load_lora_state(payload["lora_state"])
        _skipped_resume = load_action_head_partial(model, payload["head_state"])
        if _skipped_resume:
            print(f"[resume][warn] 动作头跳过的键: {_skipped_resume}")
        if payload.get("aux_state") and getattr(model, "aux_head", None) is not None:
            model.aux_head.load_state_dict(payload["aux_state"])
        resume_extra = payload.get("extra", {})
        start_epoch = int(resume_extra.get("epoch", 0))
        global_step = int(resume_extra.get("global_step", 0))
        best_mse = float(resume_extra.get("best_mse_so_far", float("inf")))
        bad_epochs = int(resume_extra.get("bad_epochs", 0))
        if args.reset_progress:
            start_epoch = 0
            global_step = 0
            best_mse = float("inf")
            bad_epochs = 0
            print("[resume] --reset-progress:epoch/global_step/最优值全部清零(仅加载权重)")
        print(f"[resume] 从 {args.resume} 恢复: 已完成 {start_epoch} epoch, "
              f"global_step={global_step}, best_mse={best_mse:.5f}")
    model.to(device)
    print(f"[init] 模型加载 {time.time() - t_load:.1f}s, device={device}, "
          f"image={cfg.image_size}px seq={cfg.image_seq_len} "
          f"frames={cfg.frames} stride={cfg.frame_stride}")
    params = model.trainable_parameters()
    opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.95),
                            weight_decay=args.weight_decay)
    if args.resume and not args.reset_progress:
        opt_path = os.path.join(args.out, f"optim_ep{start_epoch}.pt")
        if os.path.exists(opt_path):
            opt.load_state_dict(torch.load(opt_path, map_location="cpu", weights_only=False))
            print(f"[resume] 优化器状态已恢复: {opt_path}")
        else:
            print(f"[resume][warn] 找不到 {opt_path},优化器从零开始(仅影响动量)")

    # 注意:加权采样 + --epoch-frames 时,每个 epoch 的实际 batch 数由 epoch_frames 决定,
    # 而不是原始 train_loader 长度;LR schedule 必须按实际步数算,否则 cosine 会被拉长、
    # 学习率长期停在接近峰值(2026-11-22 v2/v3 首轮踩过这个坑)。
    if (parse_instr_weights(args.instr_weights) or args.balance_instr) and args.epoch_frames:
        steps_per_epoch = max(1, int(args.epoch_frames / args.bs))
    else:
        steps_per_epoch = len(train_loader)
    if args.max_steps:
        steps_per_epoch = min(steps_per_epoch, args.max_steps * args.grad_accum)
    total_steps = max(1, math.ceil(steps_per_epoch / args.grad_accum) * args.epochs)
    print(f"[sched] steps/epoch(有效)={steps_per_epoch} grad_accum={args.grad_accum} "
          f"total_opt_steps={total_steps}")

    log_path = os.path.join(args.out, "train_log.json")
    if args.resume and not args.reset_progress and os.path.exists(log_path):
        with open(log_path, "r", encoding="utf-8") as f:
            log = json.load(f)
        log["args"] = vars(args)
        log["resumed_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[resume] 续写已有 train_log.json({len(log.get('epochs', []))} 条记录)")
    else:
        log = {
            "args": vars(args),
            "config": cfg.to_dict(),
            "data": data_stats,
            "trainable_params": sum(p.numel() for p in params),
            "total_params": sum(numel for numel in [sum(p.numel() for p in model.parameters())]),
            "epochs": [],
            "gpu": torch.cuda.get_device_name(0),
            "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
    print(f"[data] {data_stats['frames_train']} train / {data_stats['frames_val']} val 帧, "
          f"{data_stats['num_shards']} shards; steps/epoch={len(train_loader)} total_opt_steps={total_steps}")

    step_times, data_times = [], []
    gpu_peak = 0.0

    order_rng = np.random.default_rng(args.seed + 1000)
    instr_weights = parse_instr_weights(args.instr_weights)
    if instr_weights or args.balance_instr:
        print(f"[sample] 指令加权采样: weights={instr_weights} balance={args.balance_instr}; "
              f"原分布={ {int(k): int((train_instr_ids == k).sum()) for k in np.unique(train_instr_ids)} }")
    for epoch in range(start_epoch, args.epochs):
        if args.shard_shuffle:
            if instr_weights or args.balance_instr:
                train_ds.indices, sample_dist = weighted_epoch_indices(
                    train_idx, train_instr_ids, instr_weights, order_rng,
                    balance=args.balance_instr,
                    total=args.epoch_frames or len(train_idx),
                    row_weights=train_row_weights)

                if epoch == start_epoch:
                    orig = {int(k): int((train_instr_ids == k).sum())
                            for k in np.unique(train_instr_ids)}
                    print(f"[sample] 加权后 epoch 帧数={len(train_ds.indices)}(原 {len(train_idx)})")
                    print(f"[sample] 原分布: {orig}")
                    print(f"[sample] 采样分布: {sample_dist}")
                    if src_weight_map:
                        n_total = len(train_ds.indices)
                        shares = {}
                        for di, mult in src_weight_map.items():
                            exp_frames = 0.0
                            for i in np.unique(train_instr_ids):
                                m = train_instr_ids == i
                                rw = train_row_weights[m]
                                frac = (rw[train_source[m] == di].sum() / rw.sum()) if rw.sum() > 0 else 0.0
                                exp_frames += sample_dist.get(int(i), 0) * frac
                            shares[dirs_abs[di]] = round(float(exp_frames) / max(1, n_total) * 100.0, 1)
                        print(f"[source] 采样期望占比(%): {shares}")
            else:
                train_ds.indices = shuffle_by_shard(train_idx, order_rng)
            train_loader = make_loader(train_ds, True)
        model.train()
        t_epoch = time.time()
        run_loss, run_n = 0.0, 0
        epoch_loss_sum, epoch_n = 0.0, 0
        t_last = time.time()
        opt.zero_grad(set_to_none=True)
        for it, batch in enumerate(train_loader):
            if args.max_steps is not None and it >= args.max_steps * args.grad_accum:
                break
            t_data = time.time() - t_last
            data_times.append(t_data)
            t0 = time.time()
            enc = {k: v.to(device, non_blocking=True) for k, v in batch["enc"].items()}
            target = normalize_actions(batch["actions"], cfg.v_max, cfg.w_max).to(device)
            aux_loss_val = 0.0
            if cfg.aux_dim > 0 and args.aux_loss_weight > 0 and "aux" in batch:
                pred, _feat, aux_pred = model(enc, return_aux=True)
                aux_t = (batch["aux"].to(device) / cfg.aux_scale)
                aux_m = batch["aux_mask"].to(device)
                aux_sq = ((aux_pred.float() - aux_t.float()) ** 2).mean(dim=-1) * aux_m
                aux_loss = aux_sq.sum() / aux_m.sum().clamp(min=1.0)
                aux_loss_val = float(aux_loss.item())
                loss = (((pred.float() - target.float()) ** 2).mean()
                        + args.aux_loss_weight * aux_loss) / args.grad_accum
            else:
                pred, _ = model(enc)
                loss = ((pred.float() - target.float()) ** 2).mean() / args.grad_accum
            loss.backward()
            batch_mse = float(loss.item()) * args.grad_accum
            run_loss += batch_mse * pred.shape[0]
            run_n += pred.shape[0]
            epoch_loss_sum += batch_mse * pred.shape[0]
            epoch_n += pred.shape[0]
            do_step = ((it + 1) % args.grad_accum == 0) or (it + 1 == len(train_loader))
            if do_step:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                lr = lr_at(global_step, total_steps, args.lr)
                for g in opt.param_groups:
                    g["lr"] = lr
                opt.step()
                opt.zero_grad(set_to_none=True)
                global_step += 1
            torch.cuda.synchronize()
            step_times.append(time.time() - t0)
            t_last = time.time()
            if (it + 1) % args.log_every == 0:
                mem = torch.cuda.max_memory_allocated() / (1024 ** 2)
                gpu_peak = max(gpu_peak, mem)
                print(f"[ep {epoch + 1}/{args.epochs}] it {it + 1}/{len(train_loader)} "
                      f"loss={run_loss / max(1, run_n):.5f} lr={lr:.2e} "
                      f"step={np.mean(step_times[-20:]) * 1000:.0f}ms "
                      f"data={np.mean(data_times[-20:]) * 1000:.0f}ms "
                      f"peak={mem:.0f}MiB", flush=True)
                run_loss, run_n = 0.0, 0

        torch.cuda.synchronize()
        epoch_time = time.time() - t_epoch
        train_loss = epoch_loss_sum / max(1, epoch_n)
        val = evaluate(model, val_loader, device, cfg.v_max, cfg.w_max,
                       max_batches=args.val_batches)
        gpu_peak = max(gpu_peak, torch.cuda.max_memory_allocated() / (1024 ** 2))
        rec = {
            "epoch": epoch + 1,
            "train_loss_tail": float(train_loss),
            "val": val,
            "epoch_time_s": round(epoch_time, 1),
            "mean_step_ms": round(float(np.mean(step_times)) * 1000, 1),
            "gpu_peak_mib": round(gpu_peak, 0),
        }
        log["epochs"].append(rec)
        print(f"[ep {epoch + 1}/{args.epochs}] train_tail_loss={train_loss:.5f} "
              f"val_mse_norm={val.get('mse_norm', -1):.5f} "
              f"val_aux_m={val.get('aux_mse_m', 'n/a')} "
              f"val_mse_v={val.get('mse_v', -1):.6f} val_mse_w={val.get('mse_w', -1):.6f} "
              f"time={epoch_time:.0f}s peak={gpu_peak:.0f}MiB", flush=True)

        score = val.get("mse_norm", float("inf"))
        if score < best_mse:
            best_mse_new = score
        else:
            best_mse_new = best_mse
        extra = {
            "epoch": epoch + 1,
            "global_step": global_step,
            "val": val,
            "data": data_stats,
            "args": vars(args),
            "best_mse_so_far": best_mse_new,
            "bad_epochs": 0 if score < best_mse else bad_epochs + 1,
        }
        # 每个 epoch 都存一份(LoRA+动作头,约 23 MB),便于中途评测/回滚
        ep_path = os.path.join(args.out, f"ep{epoch + 1}.pt")
        save_checkpoint(ep_path, model, extra=extra)
        torch.save(opt.state_dict(), os.path.join(args.out, f"optim_ep{epoch + 1}.pt"))
        print(f"[ckpt] {ep_path} + optim_ep{epoch + 1}.pt 已保存", flush=True)
        if score < best_mse:
            best_mse = score
            bad_epochs = 0
            save_checkpoint(os.path.join(args.out, "best.pt"), model, extra=extra)
            print(f"[ckpt] best.pt 已保存 (val_mse_norm={best_mse:.5f})", flush=True)
        else:
            bad_epochs += 1
        save_checkpoint(os.path.join(args.out, "last.pt"), model, extra=extra)
        with open(os.path.join(args.out, "train_log.json"), "w", encoding="utf-8") as f:
            json.dump(log, f, ensure_ascii=False, indent=2)
        if args.stop_after_epoch is not None and (epoch + 1) >= args.stop_after_epoch:
            print(f"[pause] 已到 --stop-after-epoch={args.stop_after_epoch},正常退出;"
                  f"续训: --resume {args.out}/ep{epoch + 1}.pt ... --epochs {args.epochs}", flush=True)
            break
        if bad_epochs >= args.patience:
            print(f"[early-stop] 验证集连续 {bad_epochs} 个 epoch 未提升,停止训练")
            break

    log["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    log["best_val_mse_norm"] = best_mse
    log["gpu_peak_mib"] = round(gpu_peak, 0)
    log["mean_step_ms"] = round(float(np.mean(step_times)) * 1000, 1)
    log["mean_samples_per_s"] = round(args.bs / float(np.mean(step_times)), 1)
    with open(os.path.join(args.out, "train_log.json"), "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)
    try:
        import subprocess
        smi = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total",
                              "--format=csv,noheader"], capture_output=True, text=True)
        log["nvidia_smi_after"] = smi.stdout.strip()
    except Exception:
        pass
    print(f"[done] best_val_mse_norm={best_mse:.5f} gpu_peak={gpu_peak:.0f}MiB "
          f"mean_step={np.mean(step_times) * 1000:.0f}ms "
          f"samples/s={args.bs / float(np.mean(step_times)):.1f}")
    print(f"[done] 产物: {args.out}/best.pt, {args.out}/last.pt, {args.out}/train_log.json")


if __name__ == "__main__":
    main()
