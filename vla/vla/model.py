"""SmolVLM-256M + 动作头:VLA 行为克隆模型(本项目核心模型定义)。

本模块只做"模型"本身:processor 构造、prompt 模板、动作头、checkpoint 存取。
- 训练:vla/train_bc.py
- 推理封装(VLAPolicy):vla/infer.py
- 数据格式见 specs/VLA-TASK-SPEC.md 第 3 节,模型接口见第 4 节。

关键实现说明
------------
1. transformers 5.x 没有 AutoModelForVision2Seq,改用 AutoModelForImageTextToText,
   加载到的是 Idefics3ForConditionalGeneration(带 lm_head)。
2. 动作只需要"最后一个 token 的 hidden state",不需要 logits,因此训练/推理都直接调用
   内层 Idefics3Model(base.model),避免每步计算 4.9 万词表的 lm_head(实测省约 30% 时间)。
3. SmolVLM 的 processor 默认把图放大到 512x512,得到 64 个 image token(scale_factor=4
   pixel shuffle: (512/16)^2/16 = 64)。本项目源图只有 128x128,512 上采样纯属浪费,
   因此按 spec 第 10 节把 image_size 降到 256(16 token)或 384(36 token),
   同步设置 processor.image_seq_len,否则 image token 数与视觉特征数不匹配。
4. 动作头:last hidden(hidden=576) -> LayerNorm -> Linear(576,256) -> GELU
   -> Linear(256,2) -> tanh,输出归一化动作 [v/0.35, w/1.0]。
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MODEL_PATH = os.path.join(PROJECT_ROOT, "assets", "hf", "SmolVLM-256M-Instruct")

# 12 条指令(spec 1.3),列表下标就是 instr_id
INSTRUCTIONS: List[str] = [
    "保持站立",
    "向前走",
    "向后退",
    "向左转",
    "向右转",
    "原地左旋",
    "原地右旋",
    "面向红桶",
    "靠近红桶",
    "靠近蓝桶",
    "靠近绿桶",
    "绕开红桶前往蓝桶",
]

ACTION_V_MAX = 0.35   # m/s
ACTION_W_MAX = 1.0    # rad/s
PROMPT_TEXT = "指令：{instruction}"

# LoRA 作用在 LM 的 q/k/v/o + gate/up/down(视觉塔冻结,不做 LoRA)
DEFAULT_LORA_TARGETS: Tuple[str, ...] = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


def seq_len_for_image_size(image_size: int) -> int:
    """SmolVLM 的 image token 数 = (image_size / 16)^2 / (scale_factor 4)^2。

    等价于 (image_size / 64)^2:256 -> 16,384 -> 36,512 -> 64。
    """
    patches = int(image_size / 16)
    return int(patches * patches / 16)


@dataclass
class VLAConfig:
    """模型结构 + 预处理配置(checkpoint 里会存一份,推理可完全复现)。"""

    base_model: str = DEFAULT_MODEL_PATH
    image_size: int = 256
    image_seq_len: int = 16
    head_hidden: int = 256
    action_dim: int = 2
    v_max: float = ACTION_V_MAX
    w_max: float = ACTION_W_MAX
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    lora_targets: Tuple[str, ...] = DEFAULT_LORA_TARGETS
    prompt_text: str = PROMPT_TEXT
    use_chat_template: bool = True
    frames: int = 1               # 每个样本输入的历史帧数(1=单帧;2=(t-1,t) 两帧)
    aux_dim: int = 0              # >0 时启用辅助回归头(桶相对位置 [dx,dy]x3,单位 m)
    aux_scale: float = 3.0        # 辅助标签归一化系数(标签 / aux_scale)

    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        d["lora_targets"] = list(self.lora_targets)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "VLAConfig":
        d = dict(d)
        if "lora_targets" in d:
            d["lora_targets"] = tuple(d["lora_targets"])
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})

    @classmethod
    def for_image_size(cls, image_size: int, **kwargs) -> "VLAConfig":
        seq = seq_len_for_image_size(image_size)
        return cls(image_size=image_size, image_seq_len=seq, **kwargs)


# ---------------------------------------------------------------------------
# processor
# ---------------------------------------------------------------------------

def build_processor(model_path: str = DEFAULT_MODEL_PATH, image_size: int = 256,
                    image_seq_len: int | None = None):
    """构造 AutoProcessor,并把图像尺寸/图像 token 数按本项目需求改写。

    - image_processor.size / max_image_size 都设为 image_size,保证 128x128 源图被
      双线性放大到 image_size,且不会切成多张 crop(平板直传 1 张 global 图)。
    - image_seq_len 必须与 (image_size/64)^2 一致,否则 forward 报
      "Image features and image tokens do not match"。
    """
    from transformers import AutoProcessor

    if image_seq_len is None:
        image_seq_len = seq_len_for_image_size(image_size)
    proc = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    edge = {"longest_edge": int(image_size)}
    # 关闭 Idefics3 的切图:默认会把 128 源图先放大到 2048 再切 16x16 个 512 crop,
    # 单帧 input_ids 能到 1.1 万 token。关闭后每帧只有 1 张 global 图 + image_seq_len 个 token。
    proc.image_processor.do_image_splitting = False
    proc.image_processor.max_image_size = dict(edge)
    proc.image_processor.size = dict(edge)
    proc.image_seq_len = int(image_seq_len)
    proc.tokenizer.padding_side = "right"
    return proc


def to_pil(image) -> "object":
    """np.uint8 [H,W,3] 或 PIL -> PIL RGB。"""
    from PIL import Image as PILImage
    if isinstance(image, PILImage.Image):
        return image.convert("RGB")
    arr = np.asarray(image)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return PILImage.fromarray(arr).convert("RGB")


def build_text(processor, instruction: str, n_images: int = 1) -> str:
    """按 Idefics3/SmolVLM chat template 生成 prompt 文本。

    n_images=1 产出形如:
      '<|im_start|>User:<image>指令：向前走<end_of_utterance>\nAssistant:'
    n_images=2(历史帧) 则连续两个 <image>:
      '<|im_start|>User:<image><image>指令：向前走<end_of_utterance>\nAssistant:'
    image token 的实际展开(每图 image_seq_len 个)由 processor 完成。
    """
    content = [{"type": "image"} for _ in range(max(1, int(n_images)))]
    content.append({"type": "text", "text": PROMPT_TEXT.format(instruction=instruction)})
    messages = [{"role": "user", "content": content}]
    return processor.apply_chat_template(messages, add_generation_prompt=True)


def prepare_inputs(processor, images: Sequence, instructions: Sequence[str],
                   device: str | torch.device | None = "cuda") -> Dict[str, torch.Tensor]:
    """把一批 (图像, 指令) 变成模型输入张量。

    images: 长度 = len(instructions);每项可以是单张图(np.uint8 [H,W,3] / PIL),
            也可以是"多帧列表" [prev, cur](历史帧模型,spec 变体)。
    device=None 时留在 CPU(DataLoader worker 里用),否则搬到 device。
    返回 input_ids/attention_mask/pixel_values/pixel_attention_mask。
    """
    assert len(images) == len(instructions), "images 与 instructions 数量必须一致"
    samples = []
    for im in images:
        if isinstance(im, (list, tuple)):
            samples.append([to_pil(x) for x in im])
        else:
            samples.append([to_pil(im)])
    texts = [build_text(processor, ins, n_images=len(smp))
             for ins, smp in zip(instructions, samples)]
    enc = processor(text=texts, images=samples,
                    return_tensors="pt", padding=True)
    keep = ("input_ids", "attention_mask", "pixel_values", "pixel_attention_mask")
    out = {k: enc[k] for k in keep if k in enc}
    if device is not None:
        out = {k: v.to(device) for k, v in out.items()}
    return out


# ---------------------------------------------------------------------------
# 动作头 + 模型
# ---------------------------------------------------------------------------

class ActionHead(nn.Module):
    """LayerNorm -> Linear -> GELU -> Linear -> tanh,输出 2 维归一化动作。"""

    def __init__(self, hidden_size: int = 576, mid: int = 256, action_dim: int = 2):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.fc1 = nn.Linear(hidden_size, mid)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(mid, action_dim)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.fc2(self.act(self.fc1(self.norm(hidden)))))


def load_base_model(model_path: str = DEFAULT_MODEL_PATH, dtype=torch.bfloat16):
    """加载本地 SmolVLM(local_files_only,绝不联网)。"""
    from transformers import AutoModelForImageTextToText
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        local_files_only=True,
        dtype=dtype,
        attn_implementation="sdpa",
    )
    model.config.use_cache = False
    return model


class AuxHead(nn.Module):
    """辅助回归头(LayerNorm -> Linear -> GELU -> Linear),预测桶相对位置。"""

    def __init__(self, hidden_size: int = 576, mid: int = 256, aux_dim: int = 6):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.fc1 = nn.Linear(hidden_size, mid)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(mid, aux_dim)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(self.norm(hidden))))


class VLAModel(nn.Module):
    """SmolVLM-256M(视觉塔冻结 + LM LoRA)+ 动作头(+ 可选辅助回归头)。"""

    def __init__(self, cfg: VLAConfig, base=None, verbose: bool = False):
        super().__init__()
        from peft import LoraConfig, get_peft_model

        self.cfg = cfg
        if base is None:
            base = load_base_model(cfg.base_model)
        # 1) 先全部冻结,再给 LM 加 LoRA;视觉塔永远冻结
        for p in base.parameters():
            p.requires_grad = False
        lora_cfg = LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            bias="none",
            target_modules=list(cfg.lora_targets),
        )
        self.lm = get_peft_model(base, lora_cfg)
        # 2) 内层 Idefics3Model(不含 lm_head),动作只需要 hidden state
        self.inner = base.model
        hidden_size = int(base.config.text_config.hidden_size)
        self.action_head = ActionHead(hidden_size, cfg.head_hidden, cfg.action_dim)
        self.aux_head = AuxHead(hidden_size, cfg.head_hidden, cfg.aux_dim) if cfg.aux_dim > 0 else None
        self.hidden_size = hidden_size
        if verbose:
            n_train = sum(p.numel() for p in self.parameters() if p.requires_grad)
            n_all = sum(p.numel() for p in self.parameters())
            print(f"[VLAModel] trainable={n_train/1e6:.2f}M total={n_all/1e6:.1f}M "
                  f"hidden={hidden_size} image={cfg.image_size}px seq={cfg.image_seq_len}")

    # -- 前向 ---------------------------------------------------------------
    def forward(self, enc: Dict[str, torch.Tensor], return_aux: bool = False):
        """enc: prepare_inputs 的输出。

        默认返回 (归一化动作 [B,2], last hidden [B,H]);
        return_aux=True 且启用了辅助头时返回 (动作, hidden, aux_pred[B,aux_dim])。
        """
        out = self.inner(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            pixel_values=enc["pixel_values"],
            pixel_attention_mask=enc.get("pixel_attention_mask"),
            use_cache=False,
        )
        hidden = out.last_hidden_state                     # [B, T, 576]
        # 右 padding:每条样本最后一个真实 token = attention_mask 之和 - 1
        last_idx = enc["attention_mask"].sum(dim=1) - 1
        feat = hidden[torch.arange(hidden.shape[0], device=hidden.device), last_idx]
        actions = self.action_head(feat.float())           # 动作头用 fp32 更稳
        if return_aux and self.aux_head is not None:
            return actions, feat, self.aux_head(feat.float())
        return actions, feat

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    # -- checkpoint ---------------------------------------------------------
    def lora_state_dict(self) -> Dict[str, torch.Tensor]:
        return {k: v.detach().cpu() for k, v in self.state_dict().items() if "lora_" in k}

    def load_lora_state(self, state: Dict[str, torch.Tensor]) -> None:
        own = self.state_dict()
        missing = [k for k in state if k not in own]
        if missing:
            raise KeyError(f"LoRA 权重与当前结构不匹配,缺少 {len(missing)} 个键,例如 {missing[0]}")
        own.update({k: v for k, v in state.items()})
        self.load_state_dict(own)


def denormalize(actions_norm: torch.Tensor, v_max: float = ACTION_V_MAX,
                w_max: float = ACTION_W_MAX) -> torch.Tensor:
    """归一化动作 [v/0.35, w/1.0] -> 物理量 (v m/s, w rad/s)。"""
    out = actions_norm.clone()
    out[..., 0] = out[..., 0] * v_max
    out[..., 1] = out[..., 1] * w_max
    return out


def normalize_actions(actions_phys: torch.Tensor, v_max: float = ACTION_V_MAX,
                      w_max: float = ACTION_W_MAX) -> torch.Tensor:
    out = actions_phys.clone().float()
    out[..., 0] = out[..., 0] / v_max
    out[..., 1] = out[..., 1] / w_max
    return torch.clamp(out, -1.0, 1.0)


def save_checkpoint(path: str, model: VLAModel, extra: dict | None = None) -> None:
    """只保存可训练部分(LoRA ~几 MB + 动作头),基座从本地缓存重新加载。"""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    payload = {
        "format": "vla_bc_v1",
        "config": model.cfg.to_dict(),
        "lora_state": model.lora_state_dict(),
        "head_state": {k: v.detach().cpu() for k, v in model.action_head.state_dict().items()},
        "extra": extra or {},
    }
    if getattr(model, "aux_head", None) is not None:
        payload["aux_state"] = {k: v.detach().cpu()
                                for k, v in model.aux_head.state_dict().items()}
    torch.save(payload, path)


def load_model_from_checkpoint(path: str, device: str | torch.device = "cuda",
                               verbose: bool = False) -> Tuple[VLAModel, dict]:
    """返回 (eval 模式的 VLAModel, checkpoint 附加信息)。"""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    cfg = VLAConfig.from_dict(payload["config"])
    model = VLAModel(cfg, verbose=verbose)
    model.load_lora_state(payload["lora_state"])
    model.action_head.load_state_dict(payload["head_state"])
    if payload.get("aux_state") and getattr(model, "aux_head", None) is not None:
        model.aux_head.load_state_dict(payload["aux_state"])
    model.to(device)
    model.eval()
    return model, payload.get("extra", {})


def read_stats_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
