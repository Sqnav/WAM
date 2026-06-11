from __future__ import annotations

import argparse
import json
import math
import random
import os
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from PIL import Image
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms

from data.visual_guidance import make_attention_heatmap
from data.teacher_dataset_builder import build_records
from model.config import ModelConfig
from model.losses import summarize_losses, world_model_dit_loss
from model.model import TeacherWorldModelDiT, migrate_legacy_state_dict_keys

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None


_DEFAULT_CFG = ModelConfig()

try:
    import deepspeed
except Exception:
    deepspeed = None


def _str2bool(value: str | bool) -> bool:
    """Parse shell / CLI strings into bool (e.g. true false 1 0 yes no)."""
    if isinstance(value, bool):
        return value
    s = str(value).strip().lower()
    if s in ("1", "true", "t", "yes", "y", "on"):
        return True
    if s in ("0", "false", "f", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {value!r}")


def _cuda_amp_dtype(cfg: ModelConfig) -> torch.dtype:
    dtype_name = str(getattr(cfg, "wan22_torch_dtype", "bfloat16")).lower()
    if dtype_name in ("bf16", "bfloat16"):
        return torch.bfloat16
    if dtype_name in ("fp16", "float16", "half"):
        return torch.float16
    return torch.float32


def _autocast_context(device: torch.device, cfg: ModelConfig):
    if device.type != "cuda":
        return nullcontext()
    amp_dtype = _cuda_amp_dtype(cfg)
    if amp_dtype == torch.float32:
        return nullcontext()
    return torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=True)


def _grad_scaler_enabled(device: torch.device, cfg: ModelConfig, use_deepspeed: bool) -> bool:
    return device.type == "cuda" and (not use_deepspeed) and _cuda_amp_dtype(cfg) == torch.float16


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class TrajectoryDataset(Dataset):
    REQUIRED_KEYS = [
        "target_relative",
        "next_target_relative",
        "prev_actions",
        "expert_action",
    ]
    LEGACY_KEY_ALIASES = {
        "target_relative": "privileged",
        "next_target_relative": "next_privileged",
    }

    def __init__(
        self,
        records: List[Dict[str, Any]],
        image_size: int,
        seq_len: int,
        target_relative_dim: int,
        action_dim: int,
        direction_bins: int = 8,
        distance_bins: int = 6,
        text_context_length: int = 77,
        random_crop: bool = True,
        use_target_visual_guidance: bool = False,
        use_attention_heatmap: bool = True,
        visual_guidance_fov_deg: float = 90.0,
        attention_heatmap_sigma: float = 0.08,
        wan_latent_cache_root: Optional[str] = None,
        action_video_freq_ratio: int = 1,
        use_target_belief_tracker: bool = False,
    ) -> None:
        self.records = records
        self.seq_len = seq_len
        self.action_video_freq_ratio = max(int(action_video_freq_ratio), 1)
        self.use_target_belief_tracker = bool(use_target_belief_tracker)
        self.image_size = int(image_size)
        self.target_relative_dim = target_relative_dim
        self.action_dim = action_dim
        self.direction_bins = direction_bins
        self.distance_bins = distance_bins
        self.text_context_length = text_context_length
        self.random_crop = random_crop
        self.use_target_visual_guidance = bool(use_target_visual_guidance)
        self.use_attention_heatmap = bool(use_attention_heatmap)
        self.visual_guidance_fov_deg = float(visual_guidance_fov_deg)
        self.attention_heatmap_sigma = float(attention_heatmap_sigma)
        self.wan_latent_cache_root = Path(wan_latent_cache_root) if wan_latent_cache_root else None
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
            ]
        )

    def __len__(self) -> int:
        return len(self.records)

    def _record_name(self, record: Dict[str, Any], index: Optional[int] = None) -> str:
        for key in ["id", "trajectory_id", "episode_id", "name"]:
            if key in record:
                return f"record {record[key]}"
        if index is not None:
            return f"record index {index}"
        return "record"

    def _sequence_length(self, record: Dict[str, Any], index: int) -> int:
        rgb_paths = record.get("rgb_paths")
        if rgb_paths is not None:
            return len(rgb_paths)
        for key in ["images", *self.REQUIRED_KEYS]:
            value = record.get(key)
            if value is None and key in self.LEGACY_KEY_ALIASES:
                value = record.get(self.LEGACY_KEY_ALIASES[key])
            if value is None:
                continue
            tensor = torch.load(value, map_location="cpu") if isinstance(value, str) and Path(value).exists() else torch.tensor(value)
            if tensor.ndim >= 1:
                return int(tensor.shape[0])
        raise KeyError(f"{self._record_name(record, index)} 无法推断序列长度。")

    def _select_window(self, length: int) -> tuple[int, int]:
        if length <= 0:
            raise ValueError("trajectory length must be positive.")
        if length >= self.seq_len:
            start = random.randint(0, length - self.seq_len) if self.random_crop else 0
            return start, start + self.seq_len
        return 0, length

    def _load_rgb_sequence(self, record: Dict[str, Any], start: Optional[int] = None, end: Optional[int] = None) -> torch.Tensor:
        if "images" in record:
            value = record["images"]
            images = torch.load(value, map_location="cpu") if isinstance(value, str) else torch.tensor(value)
            if images.ndim != 4:
                raise ValueError("images must have shape [T, C, H, W].")
            if start is not None or end is not None:
                images = images[slice(start, end)]
            return images.float()
        rgb_paths = record.get("rgb_paths")
        if rgb_paths is None:
            raise KeyError("每条样本必须包含 images 或 rgb_paths。")
        if start is not None or end is not None:
            rgb_paths = rgb_paths[slice(start, end)]
        frames = []
        for p in rgb_paths:
            img = Image.open(p).convert("RGB")
            frames.append(self.transform(img))
        if len(frames) == 0:
            raise ValueError("rgb_paths 不能为空。")
        return torch.stack(frames, dim=0)

    def _load_rgb_frame(self, record: Dict[str, Any], frame_index: int) -> torch.Tensor:
        return self._load_rgb_sequence(record, start=frame_index, end=frame_index + 1)[0]

    def _latent_cache_path(self, record: Dict[str, Any], start: int, end: int) -> Optional[Path]:
        if self.wan_latent_cache_root is None:
            return None
        scene = str(record.get("scene_id", "unknown_scene"))
        traj = str(record.get("trajectory_name", record.get("trajectory_id", "unknown_traj")))
        suffix = "" if self.action_video_freq_ratio == 1 else f"_video{self.action_video_freq_ratio}"
        return self.wan_latent_cache_root / scene / traj / f"seq{self.seq_len}{suffix}_start{start:04d}_end{end:04d}.pt"

    def _load_cached_wan_latents(self, record: Dict[str, Any], start: int, end: int) -> Optional[torch.Tensor]:
        path = self._latent_cache_path(record, start, end)
        if path is None or not path.exists():
            return None
        try:
            latents = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            latents = torch.load(path, map_location="cpu")
        if isinstance(latents, dict):
            latents = latents.get("latents")
        if not torch.is_tensor(latents) or latents.ndim != 4:
            raise ValueError(f"Invalid cached Wan latent at {path}: expected [C,T,H,W].")
        return latents.float()

    def _load_tensor_field(self, record: Dict[str, Any], key: str, dtype: torch.dtype) -> Optional[torch.Tensor]:
        if key not in record:
            return None
        value = record[key]
        if isinstance(value, str) and Path(value).exists():
            tensor = torch.load(value, map_location="cpu")
        else:
            tensor = torch.tensor(value)
        return tensor.to(dtype=dtype)

    def _require_tensor_field(self, record: Dict[str, Any], key: str, dtype: torch.dtype, index: int) -> torch.Tensor:
        tensor = self._load_tensor_field(record, key, dtype)
        if tensor is None and key in self.LEGACY_KEY_ALIASES:
            tensor = self._load_tensor_field(record, self.LEGACY_KEY_ALIASES[key], dtype)
        if tensor is None:
            legacy = self.LEGACY_KEY_ALIASES.get(key)
            suffix = f" 或旧字段 `{legacy}`" if legacy is not None else ""
            raise KeyError(f"{self._record_name(record, index)} 缺少必需字段 `{key}`{suffix}。")
        return tensor

    def _text_tokens_or_placeholder(self, record: Dict[str, Any], seq_len: int, index: int) -> Dict[str, Optional[torch.Tensor]]:
        text_tokens = self._load_tensor_field(record, "text_tokens", torch.long)
        attention_mask = self._load_tensor_field(record, "attention_mask", torch.long)
        if text_tokens is not None:
            return {"text_tokens": text_tokens.long(), "attention_mask": None if attention_mask is None else attention_mask.long()}

        if record.get("instructions") is None:
            raise KeyError(f"{self._record_name(record, index)} 需要提供 text_tokens 或 instructions。")
        # Wan2.2 consumes raw instruction strings. Keep placeholder token tensors
        # only to satisfy the shared model call signature and collate path.
        return {
            "text_tokens": torch.zeros(seq_len, 1, dtype=torch.long),
            "attention_mask": torch.ones(seq_len, 1, dtype=torch.long),
        }

    def _ensure_2d(self, x: torch.Tensor, length: int, dim: int, key: str) -> torch.Tensor:
        if x.ndim == 1:
            if dim == 1 and x.numel() == length:
                x = x.unsqueeze(-1)
            elif x.numel() == dim:
                x = x.unsqueeze(0).expand(length, -1)
            else:
                raise ValueError(f"`{key}` shape {tuple(x.shape)} cannot be aligned to [T={length}, D={dim}].")
        elif x.ndim == 2:
            if x.shape == (length, dim):
                pass
            elif x.shape == (1, dim):
                x = x.expand(length, -1)
            else:
                raise ValueError(f"`{key}` must have shape [T={length}, D={dim}] or [1, D], got {tuple(x.shape)}.")
        else:
            raise ValueError(f"`{key}` must be 1D or 2D, got shape {tuple(x.shape)}.")
        return x.float()

    def _ensure_1d_bins(self, x: torch.Tensor, length: int, key: str, num_bins: int) -> torch.Tensor:
        if x.ndim == 2 and x.size(-1) == 1:
            x = x.squeeze(-1)
        if x.ndim == 0:
            x = x.view(1).expand(length)
        elif x.ndim == 1:
            if x.numel() == length:
                pass
            elif x.numel() == 1:
                x = x.expand(length)
            else:
                raise ValueError(f"`{key}` must have length T={length} or length 1, got {tuple(x.shape)}.")
        else:
            raise ValueError(f"`{key}` must be 1D or [T,1], got shape {tuple(x.shape)}.")
        x = x.long()
        if torch.any((x < 0) | (x >= num_bins)):
            raise ValueError(f"`{key}` contains values outside [0, {num_bins - 1}].")
        return x

    def _slice_time_tensor(self, x: Optional[torch.Tensor], full_length: int, start: int, end: int) -> Optional[torch.Tensor]:
        if x is None:
            return None
        if x.ndim >= 1 and x.shape[0] == full_length:
            return x[start:end]
        return x

    def _crop_or_pad(self, item: Dict[str, Optional[torch.Tensor]]) -> Dict[str, Optional[torch.Tensor]]:
        length = item["images"].shape[0]  # type: ignore[union-attr]
        static_keys = {"reference_images", "reference_target_relative"}
        if length >= self.seq_len:
            start = random.randint(0, length - self.seq_len) if self.random_crop else 0
            end = start + self.seq_len
            cropped: Dict[str, Optional[torch.Tensor]] = {}
            for k, v in item.items():
                if not isinstance(v, torch.Tensor):
                    cropped[k] = v
                elif k in static_keys:
                    cropped[k] = v
                elif v.ndim >= 1 and v.shape[0] == length:
                    cropped[k] = v[start:end]
                else:
                    cropped[k] = v
            cropped["valid_mask"] = torch.ones(self.seq_len, dtype=torch.float32)
            return cropped

        pad = self.seq_len - length
        padded: Dict[str, Optional[torch.Tensor]] = {}
        for k, v in item.items():
            if not isinstance(v, torch.Tensor):
                padded[k] = v
                continue
            if k in static_keys:
                padded[k] = v
                continue
            if v.ndim >= 1 and v.shape[0] == length:
                pad_value = v[-1:].expand(pad, *v.shape[1:])
                padded[k] = torch.cat([v, pad_value], dim=0)
            else:
                padded[k] = v
        valid_mask = torch.cat([torch.ones(length), torch.zeros(pad)], dim=0)
        padded["valid_mask"] = valid_mask.float()
        return padded

    def __getitem__(self, index: int) -> Dict[str, Optional[torch.Tensor]]:
        record = self.records[index]
        for key in self.REQUIRED_KEYS:
            legacy = self.LEGACY_KEY_ALIASES.get(key)
            if key not in record and (legacy is None or legacy not in record):
                suffix = f" 或旧字段 `{legacy}`" if legacy is not None else ""
                raise KeyError(f"{self._record_name(record, index)} 缺少必需字段 `{key}`{suffix}。")

        full_len = self._sequence_length(record, index)
        start, end = self._select_window(full_len)
        video_latents = None
        images = None
        if end - start == self.seq_len:
            video_latents = self._load_cached_wan_latents(record, start=start, end=end)
        if video_latents is None:
            images = self._load_rgb_sequence(record, start=start, end=end)
        else:
            images = torch.zeros(end - start, 3, self.image_size, self.image_size, dtype=torch.float32)

        text = self._text_tokens_or_placeholder(record, full_len, index)
        raw_instructions = record.get("instructions")
        if isinstance(raw_instructions, list):
            instruction_text = str(raw_instructions[start] if start < len(raw_instructions) else raw_instructions[0])
        elif raw_instructions is None:
            instruction_text = ""
        else:
            instruction_text = str(raw_instructions)
        target_relative = self._ensure_2d(
            self._require_tensor_field(record, "target_relative", torch.float32, index),
            full_len,
            self.target_relative_dim,
            "target_relative",
        )
        prev_actions = self._ensure_2d(
            self._require_tensor_field(record, "prev_actions", torch.float32, index),
            full_len,
            self.action_dim,
            "prev_actions",
        )
        next_target_relative = self._ensure_2d(
            self._require_tensor_field(record, "next_target_relative", torch.float32, index),
            full_len,
            self.target_relative_dim,
            "next_target_relative",
        )
        expert_action = self._ensure_2d(
            self._require_tensor_field(record, "expert_action", torch.float32, index),
            full_len,
            self.action_dim,
            "expert_action",
        )
        item: Dict[str, Optional[torch.Tensor]] = {
            "images": images.float(),
            "text_tokens": self._slice_time_tensor(text["text_tokens"].long(), full_len, start, end),  # type: ignore[union-attr]
            "attention_mask": None if text["attention_mask"] is None else self._slice_time_tensor(text["attention_mask"].long(), full_len, start, end),
            "target_relative": target_relative[start:end].float(),
            "next_target_relative": next_target_relative[start:end].float(),
            "prev_actions": prev_actions[start:end].float(),
            "expert_action": expert_action[start:end].float(),
            "instructions": instruction_text,
        }
        if self.use_target_belief_tracker:
            item["reference_images"] = self._load_rgb_frame(record, 0).float()
            item["reference_target_relative"] = target_relative[0].float()
        if video_latents is not None:
            item["video_latents"] = video_latents
        if self.use_target_visual_guidance:
            if self.use_attention_heatmap:
                item["attention_heatmaps"] = make_attention_heatmap(
                    item["target_relative"].float(),  # type: ignore[union-attr]
                    image_hw=(images.shape[-2], images.shape[-1]),
                    fov_deg=self.visual_guidance_fov_deg,
                    sigma=self.attention_heatmap_sigma,
                )
        return self._crop_or_pad(item)


def collate_fn(batch: List[Dict[str, Optional[torch.Tensor]]]) -> Dict[str, Optional[torch.Tensor]]:
    out: Dict[str, Optional[torch.Tensor]] = {}
    for key in batch[0].keys():
        values = [x[key] for x in batch]
        if all(isinstance(v, str) for v in values):
            out[key] = values  # type: ignore[assignment]
            continue
        if all(v is None for v in values):
            out[key] = None
        elif any(v is None for v in values):
            raise ValueError(f"Batch 中 `{key}` 有的样本为 None、有的不是 None，请统一数据格式。")
        else:
            out[key] = torch.stack(values, dim=0)  # type: ignore[arg-type]
    return out


def move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


def _format_metrics(metrics: Dict[str, float]) -> str:
    order = [
        "total",
        "action",
        "video",
        "next_target_relative",
        "prior_next_target_relative",
        "kl",
    ]
    hidden_when_zero = {
        "kl",
        "next_target_relative",
        "prior_next_target_relative",
        "video_x0",
        "x0_action",
    }
    parts = []
    for key in order:
        if key in metrics and not (key in hidden_when_zero and abs(metrics[key]) < 1e-12):
            parts.append(f"{key}={metrics[key]:.4f}")
    for key in sorted(metrics.keys()):
        if key in order:
            continue
        if key in hidden_when_zero and abs(metrics[key]) < 1e-12:
            continue
        parts.append(f"{key}={metrics[key]:.4f}")
    return " | ".join(parts)


def _tqdm_train_postfix(avg: Dict[str, float]) -> Dict[str, str]:
    """Active running-average loss keys for tqdm, four decimal places, stable order."""
    order = [
        "total",
        "action",
        "video",
        "next_target_relative",
        "prior_next_target_relative",
        "kl",
    ]
    hidden_when_zero = {
        "kl",
        "next_target_relative",
        "prior_next_target_relative",
        "video_x0",
        "x0_action",
    }
    out: Dict[str, str] = {}
    for key in order:
        if key in avg and not (key in hidden_when_zero and abs(avg[key]) < 1e-12):
            out[key] = f"{avg[key]:.4f}"
    for key in sorted(avg.keys()):
        if key in out:
            continue
        if key in hidden_when_zero and abs(avg[key]) < 1e-12:
            continue
        out[key] = f"{avg[key]:.4f}"
    return out


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    if hasattr(model, "module"):
        return model.module
    if isinstance(model, (torch.nn.DataParallel, DDP)):
        return model.module
    return model


def _trainable_state_dict(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    """Checkpoint only trainable params to avoid repeatedly writing frozen encoders."""
    unwrapped = _unwrap_model(model)
    trainable_names = {name for name, param in unwrapped.named_parameters() if param.requires_grad}
    state = unwrapped.state_dict()
    return {name: tensor for name, tensor in state.items() if name in trainable_names}


def _wan_latent_cache_stats(
    records: List[Dict[str, Any]],
    cache_root: str,
    seq_len: int,
    action_video_freq_ratio: int = 1,
) -> Optional[Dict[str, int]]:
    if not cache_root:
        return None
    root = Path(cache_root)
    if not root.exists():
        return {"records": len(records), "windows": 0, "hits": 0}
    windows = 0
    hits = 0
    for record in records:
        rgb_paths = record.get("rgb_paths") or []
        length = len(rgb_paths)
        starts = range(0, length - seq_len + 1) if length >= seq_len else range(0, 1)
        for start in starts:
            end = min(start + seq_len, length)
            if end - start != seq_len:
                continue
            windows += 1
            scene = str(record.get("scene_id", "unknown_scene"))
            traj = str(record.get("trajectory_name", record.get("trajectory_id", "unknown_traj")))
            suffix = "" if int(action_video_freq_ratio) <= 1 else f"_video{int(action_video_freq_ratio)}"
            path = root / scene / traj / f"seq{seq_len}{suffix}_start{start:04d}_end{end:04d}.pt"
            if path.exists():
                hits += 1
    return {"records": len(records), "windows": windows, "hits": hits}


def _ddp_is_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def _get_rank() -> int:
    return int(os.environ.get("RANK", "0"))


def _get_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def _get_world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def _is_main_process() -> bool:
    return _get_rank() == 0


def _reduce_metrics(metrics: Dict[str, float], device: torch.device, distributed: bool) -> Dict[str, float]:
    if not distributed or not metrics:
        return metrics
    keys = sorted(metrics.keys())
    values = torch.tensor([float(metrics[k]) for k in keys], device=device, dtype=torch.float32)
    dist.all_reduce(values, op=dist.ReduceOp.SUM)
    values = values / float(_get_world_size())
    return {k: float(v.item()) for k, v in zip(keys, values)}


def _make_deepspeed_config(args: argparse.Namespace) -> Dict[str, Any]:
    world_size = max(_get_world_size(), 1)
    grad_accum = max(int(args.gradient_accumulation_steps), 1)
    micro_batch = int(args.batch_size)
    zero_optimization = {
        "stage": 1,
        "offload_param": {"device": "none"},
        "overlap_comm": False,
        "contiguous_gradients": False,
        "reduce_bucket_size": 2e8,
        "allgather_bucket_size": 2e8,
    }
    if bool(getattr(args, "deepspeed_offload_optimizer", False)):
        zero_optimization["offload_optimizer"] = {"device": "cpu", "pin_memory": True}
    return {
        "train_micro_batch_size_per_gpu": micro_batch,
        "gradient_accumulation_steps": grad_accum,
        "train_batch_size": micro_batch * grad_accum * world_size,
        "bf16": {"enabled": True},
        "zero_optimization": zero_optimization,
        "zero_force_ds_cpu_optimizer": False,
        "gradient_clipping": float(args.grad_clip),
        "steps_per_print": 1000000,
    }


def _cosine_epoch_lr(base_lr: float, epoch: int, total_epochs: int, eta_min: float = 0.0) -> float:
    if total_epochs <= 0:
        return float(base_lr)
    progress = min(max(float(epoch) / float(total_epochs), 0.0), 1.0)
    return float(eta_min + 0.5 * (float(base_lr) - float(eta_min)) * (1.0 + math.cos(math.pi * progress)))


def _cosine_step_lr(base_lr: float, step: int, total_steps: int, eta_min: float = 0.0) -> float:
    if total_steps <= 0:
        return float(base_lr)
    progress = min(max(float(step) / float(total_steps), 0.0), 1.0)
    return float(eta_min + 0.5 * (float(base_lr) - float(eta_min)) * (1.0 + math.cos(math.pi * progress)))


def _set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = float(lr)


def _init_swanlab(args: argparse.Namespace, cfg: ModelConfig, run_name: str):
    if not bool(getattr(args, "use_swanlab", False)) or not _is_main_process():
        return None
    try:
        import swanlab
    except Exception as exc:
        print(f"[swanlab] disabled: import failed ({exc})")
        return None
    try:
        return swanlab.init(
            project=args.swanlab_project,
            workspace=args.swanlab_workspace or None,
            experiment_name=run_name,
            logdir=args.swanlab_log_dir or None,
            mode=args.swanlab_mode,
            config={
                **cfg.__dict__,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "scene_list": args.scene_list,
                "trajectory_range": args.trajectory_range,
            },
        )
    except Exception as exc:
        print(f"[swanlab] disabled: init failed ({exc})")
        return None


def _swanlab_log(run, metrics: Dict[str, float], step: int, prefix: str) -> None:
    if run is None or not _is_main_process():
        return
    hidden_when_zero = {
        "kl",
        "next_target_relative",
        "prior_next_target_relative",
        "video_x0",
        "x0_action",
    }
    active = {
        k: float(v)
        for k, v in metrics.items()
        if not (k in hidden_when_zero and abs(float(v)) < 1e-12)
    }
    if not active:
        return
    try:
        import swanlab
        swanlab.log({f"{prefix}/{k}": v for k, v in active.items()}, step=step)
    except Exception as exc:
        print(f"[swanlab] log skipped: {exc}")


def _swanlab_finish(run) -> None:
    if run is None or not _is_main_process():
        return
    try:
        import swanlab
        swanlab.finish()
    except Exception as exc:
        print(f"[swanlab] finish skipped: {exc}")


@torch.no_grad()
def evaluate(model: TeacherWorldModelDiT, loader: DataLoader, cfg: ModelConfig, device: torch.device) -> Dict[str, float]:
    model.eval()
    acc: Dict[str, float] = {}
    count = 0
    val_iter = loader
    if tqdm is not None:
        val_iter = tqdm(loader, desc="val", leave=False, dynamic_ncols=True)
    for batch in val_iter:
        batch = move_batch_to_device(batch, device)
        outputs = model(
            images=batch["images"],
            text_tokens=batch["text_tokens"],
            target_relative=batch["target_relative"],
            prev_actions=batch["prev_actions"],
            attention_mask=batch["attention_mask"],
            attention_heatmaps=batch.get("attention_heatmaps"),
            expert_action=batch["expert_action"],
            valid_mask=batch["valid_mask"],
            done=batch.get("done"),
            instructions=batch.get("instructions"),
            video_latents=batch.get("video_latents"),
            reference_target_relative=batch.get("reference_target_relative"),
            reference_images=batch.get("reference_images"),
        )
        losses = world_model_dit_loss(outputs, batch, cfg, valid_mask=batch["valid_mask"])
        summary = summarize_losses(losses)
        for k, v in summary.items():
            acc[k] = acc.get(k, 0.0) + v
        count += 1
    return {k: v / max(count, 1) for k, v in acc.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=str, required=True)
    parser.add_argument("--scene-list", type=str, required=True)
    parser.add_argument("--trajectory-range", type=str, default="")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--max-vel", type=float, default=_DEFAULT_CFG.max_vel, help="Physical max velocity for action normalization.")
    parser.add_argument("--max-yaw-rate", type=float, default=_DEFAULT_CFG.max_yaw_rate, help="Physical max yaw rate for action normalization.")
    parser.add_argument("--max-speed-norm", type=float, default=_DEFAULT_CFG.max_speed_norm, help="Physical speed-norm cap used both in training targets and online action execution.")
    parser.add_argument("--save-dir", type=str, required=True)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--target-relative-dim", type=int, default=3)
    parser.add_argument("--action-dim", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--max-train-steps", type=int, default=0, help="Stop after this many optimizer update steps; 0 keeps epoch-based training.")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--wan-latent-cache-root", type=str, default="")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--action-sequence-horizon", type=int, default=_DEFAULT_CFG.action_sequence_horizon)
    parser.add_argument("--action-video-freq-ratio", type=int, default=_DEFAULT_CFG.fastwam_action_video_freq_ratio)
    parser.add_argument("--diffusion-steps", type=int, default=20)
    parser.add_argument("--sampling-steps", type=int, default=20)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save-every-epochs", type=int, default=1, help="Write last.pt every N epochs; always save on final epoch.")
    parser.add_argument("--save-best-checkpoint", type=_str2bool, default=True, help="Whether to write best.pt on checkpoint epochs.")
    parser.add_argument("--save-optimizer-state", type=_str2bool, default=True, help="Include optimizer/scheduler state in checkpoints when not using DeepSpeed.")
    parser.add_argument(
        "--multi-gpu",
        action="store_true",
        help="Enable multi-GPU training. Prefer --deepspeed; legacy DDP/DataParallel remains available when --deepspeed is off.",
    )
    parser.add_argument("--deepspeed", action="store_true", help="Use DeepSpeed engine instead of DDP/DataParallel.")
    parser.add_argument("--deepspeed-config", type=str, default=None, help="Path to DeepSpeed JSON config.")
    parser.add_argument("--deepspeed-offload-optimizer", type=_str2bool, default=False)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--local_rank", type=int, default=-1, help="Passed by DeepSpeed launcher.")
    parser.add_argument("--use-swanlab", type=_str2bool, default=False)
    parser.add_argument("--swanlab-project", type=str, default="WAM-FastWAM")
    parser.add_argument("--swanlab-experiment-name", type=str, default=None)
    parser.add_argument("--swanlab-workspace", type=str, default="")
    parser.add_argument("--swanlab-log-dir", type=str, default=None)
    parser.add_argument("--swanlab-mode", type=str, default="cloud", choices=["cloud", "local", "offline", "disabled"])
    parser.add_argument("--use-wan22-encoders", type=_str2bool, default=_DEFAULT_CFG.use_wan22_encoders)
    parser.add_argument("--wan22-model-base-path", type=str, default=_DEFAULT_CFG.wan22_model_base_path)
    parser.add_argument("--wan22-fastwam-src-path", type=str, default=_DEFAULT_CFG.wan22_fastwam_src_path)
    parser.add_argument("--wan22-skip-download", type=_str2bool, default=_DEFAULT_CFG.wan22_skip_download)
    parser.add_argument("--wan22-text-context-length", type=int, default=_DEFAULT_CFG.wan22_text_context_length)
    parser.add_argument("--wan22-text-encode-batch-size", type=int, default=_DEFAULT_CFG.wan22_text_encode_batch_size)
    parser.add_argument(
        "--use-diffusion-actor",
        type=_str2bool,
        default=True,
        help="true/false: true=DiT diffusion denoising actor；false=MLP direct action head。",
    )
    parser.add_argument(
        "--use-fastwam-mot",
        type=_str2bool,
        default=_DEFAULT_CFG.use_fastwam_mot,
        help="true/false: true=FastWAM video/action MoT；false=legacy MLP/DiT actor path。",
    )
    parser.add_argument(
        "--target-token-fusion-mode",
        type=str,
        default=_DEFAULT_CFG.target_token_fusion_mode,
        choices=["attention", "concat"],
        help="attention=null target token participates in cross-attention；concat=append null target embedding after image/text fusion。",
    )
    parser.add_argument("--train-next-target-relative", type=_str2bool, default=_DEFAULT_CFG.train_next_target_relative)
    parser.add_argument("--train-rollout", type=_str2bool, default=False, help="Deprecated; prediction-head rollout supervision is disabled.")
    parser.add_argument("--next-target-relative-loss-weight", type=float, default=_DEFAULT_CFG.next_target_relative_loss_weight)
    parser.add_argument("--prior-target-relative-loss-weight", type=float, default=_DEFAULT_CFG.prior_target_relative_loss_weight)
    parser.add_argument("--rollout-loss-weight", type=float, default=_DEFAULT_CFG.rollout_loss_weight)
    parser.add_argument("--rollout-horizon", type=int, default=_DEFAULT_CFG.rollout_horizon)
    parser.add_argument("--direct-action-loss-weight", type=float, default=1.0)
    parser.add_argument("--action-yaw-loss-weight", type=float, default=_DEFAULT_CFG.action_yaw_loss_weight)
    parser.add_argument("--x0-action-loss-weight", type=float, default=_DEFAULT_CFG.x0_action_loss_weight)
    parser.add_argument("--use-target-visual-guidance", type=_str2bool, default=_DEFAULT_CFG.use_target_visual_guidance)
    parser.add_argument("--use-attention-heatmap", type=_str2bool, default=_DEFAULT_CFG.use_attention_heatmap)
    parser.add_argument("--visual-guidance-fov-deg", type=float, default=_DEFAULT_CFG.visual_guidance_fov_deg)
    parser.add_argument("--attention-heatmap-sigma", type=float, default=_DEFAULT_CFG.attention_heatmap_sigma)
    parser.add_argument("--use-heatmap-tensor-encoder", type=_str2bool, default=_DEFAULT_CFG.use_heatmap_tensor_encoder)
    parser.add_argument("--heatmap-token-scale", type=float, default=_DEFAULT_CFG.heatmap_token_scale)
    parser.add_argument("--fastwam-heatmap-context-grid", type=int, default=_DEFAULT_CFG.fastwam_heatmap_context_grid)
    parser.add_argument("--use-target-belief-tracker", type=_str2bool, default=_DEFAULT_CFG.use_target_belief_tracker)
    parser.add_argument("--target-belief-token-scale", type=float, default=_DEFAULT_CFG.target_belief_token_scale)
    parser.add_argument("--target-belief-update-rate", type=float, default=_DEFAULT_CFG.target_belief_update_rate)
    parser.add_argument("--target-belief-min-confidence", type=float, default=_DEFAULT_CFG.target_belief_min_confidence)
    parser.add_argument("--target-belief-temperature", type=float, default=_DEFAULT_CFG.target_belief_temperature)
    parser.add_argument("--target-belief-loss-weight", type=float, default=_DEFAULT_CFG.target_belief_loss_weight)
    parser.add_argument("--target-belief-motion-weight", type=float, default=_DEFAULT_CFG.target_belief_motion_weight)
    parser.add_argument("--target-belief-update-sharpness", type=float, default=_DEFAULT_CFG.target_belief_update_sharpness)
    parser.add_argument("--fastwam-lambda-action", type=float, default=_DEFAULT_CFG.fastwam_lambda_action)
    parser.add_argument("--fastwam-lambda-video", type=float, default=_DEFAULT_CFG.fastwam_lambda_video)
    parser.add_argument("--fastwam-skip-dit-load-from-pretrain", type=_str2bool, default=_DEFAULT_CFG.fastwam_skip_dit_load_from_pretrain)
    parser.add_argument("--fastwam-action-dit-pretrained-path", type=str, default=_DEFAULT_CFG.fastwam_action_dit_pretrained_path)
    parser.add_argument("--fastwam-mot-checkpoint-mixed-attn", type=_str2bool, default=_DEFAULT_CFG.fastwam_mot_checkpoint_mixed_attn)
    args = parser.parse_args()

    seed_everything(args.seed + _get_rank())
    save_dir = Path(args.save_dir)
    if _is_main_process():
        save_dir.mkdir(parents=True, exist_ok=True)

    use_deepspeed = bool(args.deepspeed)
    if use_deepspeed and deepspeed is None:
        raise ImportError("DeepSpeed is not installed in this environment.")
    use_distributed = (use_deepspeed or args.multi_gpu) and torch.cuda.is_available() and _get_world_size() > 1
    use_ddp = (not use_deepspeed) and args.multi_gpu and torch.cuda.is_available() and _get_world_size() > 1
    if use_deepspeed:
        torch.cuda.set_device(_get_local_rank())
        if dist.is_available() and not dist.is_initialized() and _get_world_size() > 1:
            deepspeed.init_distributed(dist_backend="nccl")
        device = torch.device("cuda", _get_local_rank())
    elif use_ddp:
        torch.cuda.set_device(_get_local_rank())
        dist.init_process_group(backend="nccl", init_method="env://")
        device = torch.device("cuda", _get_local_rank())
    else:
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    cfg = ModelConfig(
        image_size=args.image_size,
        text_context_length=args.wan22_text_context_length if args.use_wan22_encoders else _DEFAULT_CFG.text_context_length,
        use_wan22_encoders=args.use_wan22_encoders,
        wan22_model_base_path=args.wan22_model_base_path,
        wan22_fastwam_src_path=args.wan22_fastwam_src_path,
        wan22_skip_download=args.wan22_skip_download,
        wan22_text_context_length=args.wan22_text_context_length,
        wan22_text_encode_batch_size=args.wan22_text_encode_batch_size,
        target_relative_dim=args.target_relative_dim,
        action_dim=args.action_dim,
        action_sequence_horizon=args.action_sequence_horizon,
        fastwam_action_video_freq_ratio=max(int(args.action_video_freq_ratio), 1),
        action_diffusion_steps=args.diffusion_steps,
        action_sampling_steps=args.sampling_steps,
        max_vel=args.max_vel,
        max_yaw_rate=args.max_yaw_rate,
        max_speed_norm=args.max_speed_norm,
        target_token_fusion_mode=args.target_token_fusion_mode,
        use_target_visual_guidance=args.use_target_visual_guidance,
        use_attention_heatmap=args.use_attention_heatmap,
        visual_guidance_fov_deg=args.visual_guidance_fov_deg,
        attention_heatmap_sigma=args.attention_heatmap_sigma,
        use_heatmap_tensor_encoder=args.use_heatmap_tensor_encoder,
        heatmap_token_scale=args.heatmap_token_scale,
        fastwam_heatmap_context_grid=args.fastwam_heatmap_context_grid,
        use_target_belief_tracker=args.use_target_belief_tracker,
        target_belief_token_scale=args.target_belief_token_scale,
        target_belief_update_rate=args.target_belief_update_rate,
        target_belief_min_confidence=args.target_belief_min_confidence,
        target_belief_temperature=args.target_belief_temperature,
        target_belief_loss_weight=args.target_belief_loss_weight,
        target_belief_motion_weight=args.target_belief_motion_weight,
        target_belief_update_sharpness=args.target_belief_update_sharpness,
        use_diffusion_actor=args.use_diffusion_actor,
        use_fastwam_mot=args.use_fastwam_mot,
        use_rssm=False,
        train_kl=False,
        train_direct_action=True,
        train_next_target_relative=args.train_next_target_relative,
        train_rollout=False,
        next_target_relative_loss_weight=args.next_target_relative_loss_weight,
        prior_target_relative_loss_weight=args.prior_target_relative_loss_weight,
        rollout_loss_weight=args.rollout_loss_weight,
        rollout_horizon=args.rollout_horizon,
        direct_action_loss_weight=args.direct_action_loss_weight,
        action_yaw_loss_weight=args.action_yaw_loss_weight,
        x0_action_loss_weight=args.x0_action_loss_weight,
        fastwam_lambda_action=args.fastwam_lambda_action,
        fastwam_lambda_video=args.fastwam_lambda_video,
        fastwam_skip_dit_load_from_pretrain=args.fastwam_skip_dit_load_from_pretrain,
        fastwam_action_dit_pretrained_path=args.fastwam_action_dit_pretrained_path,
        fastwam_mot_checkpoint_mixed_attn=args.fastwam_mot_checkpoint_mixed_attn,
    )
    action_video_freq_ratio = max(int(args.action_video_freq_ratio), 1)
    if (args.seq_len - 1) % action_video_freq_ratio != 0:
        raise ValueError(
            "--seq-len must satisfy (seq_len - 1) % action_video_freq_ratio == 0; "
            f"got seq_len={args.seq_len}, action_video_freq_ratio={action_video_freq_ratio}."
        )
    sampled_video_len = (args.seq_len - 1) // action_video_freq_ratio + 1
    if sampled_video_len % 4 != 1:
        raise ValueError(
            "Sampled video frame count must satisfy T % 4 == 1 for Wan VAE; "
            f"got sampled_video_len={sampled_video_len}."
        )

    scene_list = [s.strip() for s in args.scene_list.split(",") if s.strip()]
    if not scene_list:
        raise ValueError("--scene-list is empty.")
    records = build_records(
        Path(args.dataset_root),
        scene_list,
        args.trajectory_range.strip(),
        max_vel=args.max_vel,
        max_yaw_rate=args.max_yaw_rate,
        max_speed_norm=args.max_speed_norm,
    )
    if not records:
        raise RuntimeError("No trajectory selected. Check --scene-list / --trajectory-range.")
    rng = random.Random(args.split_seed)
    rng.shuffle(records)
    val_n = int(len(records) * args.val_ratio)
    if args.val_ratio > 0.0 and len(records) > 1:
        val_n = max(1, val_n)
    val_n = min(val_n, max(len(records) - 1, 0))
    val_records = records[:val_n]
    train_records = records[val_n:] if val_n > 0 else records
    if _is_main_process():
        if val_records:
            print(f"[dataset] total={len(records)}, train={len(train_records)}, val={len(val_records)}")
        else:
            print(f"[dataset] total={len(records)}, train={len(train_records)}")
        cache_stats = _wan_latent_cache_stats(records, args.wan_latent_cache_root, args.seq_len, action_video_freq_ratio)
        if cache_stats is not None:
            hits = cache_stats["hits"]
            windows = cache_stats["windows"]
            ratio = (hits / windows) if windows else 0.0
            print(
                f"[wan-latents] cache_root={args.wan_latent_cache_root} "
                f"seq_len={args.seq_len} video_ratio={action_video_freq_ratio} hits={hits}/{windows} ({ratio:.1%})"
            )
            if windows > 0 and hits == 0:
                print("[wan-latents] WARNING: no matching cached latents; training will encode RGB videos online.")
        print(
            "[cfg curriculum] "
            f"diffusion={cfg.use_diffusion_actor} | "
            f"architecture=fastwam_video_action_mot_no_rssm fusion={cfg.target_token_fusion_mode} | "
            f"fastwam_mot={cfg.use_fastwam_mot} | "
            f"action_video_freq_ratio={cfg.fastwam_action_video_freq_ratio} | "
            f"kl={cfg.train_kl} direct_action={cfg.train_direct_action} | "
            f"action_w={cfg.direct_action_loss_weight} yaw_w={cfg.action_yaw_loss_weight} | "
            f"WAM auxiliary: next_target_relative={cfg.train_next_target_relative} rollout_head=false | "
            f"visual_guidance={cfg.use_target_visual_guidance} heatmap={cfg.use_attention_heatmap} | "
            f"target_belief_tracker={cfg.use_target_belief_tracker} belief_w={cfg.target_belief_loss_weight}"
        )
    train_dataset = TrajectoryDataset(
        records=train_records,
        image_size=args.image_size,
        seq_len=args.seq_len,
        target_relative_dim=args.target_relative_dim,
        action_dim=args.action_dim,
        direction_bins=cfg.direction_bins,
        distance_bins=cfg.distance_bins,
        text_context_length=cfg.text_context_length,
        random_crop=True,
        use_target_visual_guidance=cfg.use_target_visual_guidance,
        use_attention_heatmap=cfg.use_attention_heatmap,
        visual_guidance_fov_deg=cfg.visual_guidance_fov_deg,
        attention_heatmap_sigma=cfg.attention_heatmap_sigma,
        wan_latent_cache_root=args.wan_latent_cache_root if args.wan_latent_cache_root else None,
        action_video_freq_ratio=cfg.fastwam_action_video_freq_ratio,
        use_target_belief_tracker=cfg.use_target_belief_tracker,
    )
    train_sampler = (
        DistributedSampler(
            train_dataset,
            num_replicas=_get_world_size(),
            rank=_get_rank(),
            shuffle=True,
            drop_last=False,
        )
        if use_distributed
        else None
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=2 if args.num_workers > 0 else None,
        collate_fn=collate_fn,
    )

    # For simplicity: run validation only on rank0 under DDP.
    val_loader = None
    if val_records and ((not use_distributed) or _is_main_process()):
        val_dataset = TrajectoryDataset(
            records=val_records,
            image_size=args.image_size,
            seq_len=args.seq_len,
            target_relative_dim=args.target_relative_dim,
            action_dim=args.action_dim,
            direction_bins=cfg.direction_bins,
            distance_bins=cfg.distance_bins,
            text_context_length=cfg.text_context_length,
            random_crop=False,
            use_target_visual_guidance=cfg.use_target_visual_guidance,
            use_attention_heatmap=cfg.use_attention_heatmap,
            visual_guidance_fov_deg=cfg.visual_guidance_fov_deg,
            attention_heatmap_sigma=cfg.attention_heatmap_sigma,
            wan_latent_cache_root=args.wan_latent_cache_root if args.wan_latent_cache_root else None,
            action_video_freq_ratio=cfg.fastwam_action_video_freq_ratio,
            use_target_belief_tracker=cfg.use_target_belief_tracker,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=args.num_workers > 0,
            prefetch_factor=2 if args.num_workers > 0 else None,
            collate_fn=collate_fn,
        )

    model = TeacherWorldModelDiT(cfg).to(device)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    if use_deepspeed:
        ds_config = args.deepspeed_config if args.deepspeed_config is not None else _make_deepspeed_config(args)
        model, optimizer, _, _ = deepspeed.initialize(
            args=args,
            model=model,
            model_parameters=trainable_params,
            optimizer=optimizer,
            config=ds_config,
        )
        if _is_main_process():
            print(f"[train] DeepSpeed enabled on world_size={_get_world_size()} (local_rank={_get_local_rank()})")
    elif use_ddp:
        model = DDP(
            model,
            device_ids=[_get_local_rank()],
            output_device=_get_local_rank(),
            find_unused_parameters=False,
            broadcast_buffers=False,
        )
        if _is_main_process():
            print(f"[train] DDP enabled on world_size={_get_world_size()} (local_rank={_get_local_rank()})")
    else:
        use_dp = args.multi_gpu and device.type == "cuda" and torch.cuda.device_count() > 1
        if use_dp:
            model = torch.nn.DataParallel(model)
            print(f"[train] DataParallel enabled on {torch.cuda.device_count()} GPUs")
        else:
            print(f"[train] Device: {device}")
    scaler = torch.amp.GradScaler("cuda", enabled=_grad_scaler_enabled(device, cfg, use_deepspeed))
    if _is_main_process() and device.type == "cuda":
        print(f"[train] AMP dtype: {_cuda_amp_dtype(cfg)}, grad_scaler={scaler.is_enabled()}")

    start_epoch = 0
    global_step = 0
    best_val = math.inf
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        missing, unexpected = _unwrap_model(model).load_state_dict(
            migrate_legacy_state_dict_keys(ckpt["model"]),
            strict=False,
        )
        if _is_main_process() and (missing or unexpected):
            print(f"[resume] load strict=False: missing={len(missing)} unexpected={len(unexpected)}")
        if not use_deepspeed:
            if ckpt.get("optimizer") and ckpt.get("scheduler"):
                optimizer.load_state_dict(ckpt["optimizer"])
                scheduler.load_state_dict(ckpt["scheduler"])
            elif _is_main_process():
                print("[resume] optimizer/scheduler state missing; restarting optimizer state.")
        start_epoch = ckpt["epoch"] + 1
        global_step = int(ckpt.get("global_step", 0))
        best_val = ckpt.get("best_val", best_val)

    total_pbar = None
    run_name = args.swanlab_experiment_name or save_dir.name
    swanlab_run = _init_swanlab(args, cfg, run_name)
    if _is_main_process():
        print(
            "[running-model] "
            f"model={save_dir.name} | run={run_name} | save_dir={save_dir} | "
            f"target_belief_tracker={cfg.use_target_belief_tracker} | "
            f"visual_guidance={cfg.use_target_visual_guidance} | "
            f"fastwam_mot={cfg.use_fastwam_mot}"
        )
    if tqdm is not None and _is_main_process():
        if int(args.max_train_steps) > 0:
            total_steps = max(int(args.max_train_steps) - int(global_step), 0)
            desc = f"train steps {global_step}->{int(args.max_train_steps)}"
        else:
            total_steps = max(args.epochs - start_epoch, 0) * max(len(train_loader), 1)
            desc = f"train {start_epoch:03d}->{args.epochs - 1:03d}"
        total_pbar = tqdm(
            total=total_steps,
            desc=desc,
            leave=True,
            dynamic_ncols=True,
        )

    try:
        reached_max_steps = False
        for epoch in range(start_epoch, args.epochs):
            if int(args.max_train_steps) > 0 and global_step >= int(args.max_train_steps):
                break
            model.train()
            if use_deepspeed:
                epoch_lr = (
                    _cosine_step_lr(args.lr, global_step, int(args.max_train_steps))
                    if int(args.max_train_steps) > 0
                    else _cosine_epoch_lr(args.lr, epoch, args.epochs)
                )
                _set_optimizer_lr(optimizer, epoch_lr)
            else:
                epoch_lr = float(optimizer.param_groups[0]["lr"])
            running: Dict[str, float] = {}
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            num_train_batches = 0
            for step, batch in enumerate(train_loader):
                if int(args.max_train_steps) > 0 and global_step >= int(args.max_train_steps):
                    reached_max_steps = True
                    break
                if int(args.max_train_steps) > 0:
                    step_lr = _cosine_step_lr(args.lr, global_step, int(args.max_train_steps))
                    _set_optimizer_lr(optimizer, step_lr)
                    epoch_lr = step_lr
                batch = move_batch_to_device(batch, device)
                if use_deepspeed:
                    model.zero_grad()
                else:
                    optimizer.zero_grad(set_to_none=True)
                amp_ctx = nullcontext() if use_deepspeed else _autocast_context(device, cfg)
                with amp_ctx:
                    outputs = model(
                        images=batch["images"],
                        text_tokens=batch["text_tokens"],
                        target_relative=batch["target_relative"],
                        prev_actions=batch["prev_actions"],
                        attention_mask=batch["attention_mask"],
                        attention_heatmaps=batch.get("attention_heatmaps"),
                        expert_action=batch["expert_action"],
                        valid_mask=batch["valid_mask"],
                        done=batch.get("done"),
                        instructions=batch.get("instructions"),
                        video_latents=batch.get("video_latents"),
                        reference_target_relative=batch.get("reference_target_relative"),
                        reference_images=batch.get("reference_images"),
                    )
                    losses = world_model_dit_loss(
                        outputs,
                        batch,
                        cfg,
                        valid_mask=batch["valid_mask"],
                    )
                    loss = losses["total"]

                if use_deepspeed:
                    model.backward(loss)
                    model.step()
                else:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    clip_grad_norm_(model.parameters(), args.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()

                summary = summarize_losses(losses)
                for k, v in summary.items():
                    running[k] = running.get(k, 0.0) + v

                num_train_batches += 1
                avg = {k: v / (step + 1) for k, v in running.items()}
                global_step += 1
                if total_pbar is not None:
                    postfix = {"epoch": f"{epoch:03d}", "step": global_step, **_tqdm_train_postfix(avg)}
                    total_pbar.set_postfix(**postfix)
                    total_pbar.update(1)
                elif _is_main_process() and (step + 1) % 20 == 0:
                    print(f"[Epoch {epoch:03d} | Step {step + 1:05d}] {_format_metrics(avg)}")

            if not use_deepspeed:
                if int(args.max_train_steps) <= 0:
                    scheduler.step()
            train_avg = {k: v / max(num_train_batches, 1) for k, v in running.items()}
            train_avg = _reduce_metrics(train_avg, device, use_distributed)
            if _is_main_process():
                msg = f">>> Epoch {epoch:03d} train: {_format_metrics(train_avg)} | lr={epoch_lr:.6g} | global_step={global_step}"
                tqdm.write(msg) if tqdm is not None else print(msg)
                _swanlab_log(swanlab_run, {**train_avg, "lr": float(epoch_lr), "global_step": int(global_step)}, step=global_step, prefix="train")

            val_avg = None
            if val_loader is not None:
                val_avg = evaluate(_unwrap_model(model), val_loader, cfg, device)
                if _is_main_process():
                    msg = f">>> Epoch {epoch:03d} val:   {_format_metrics(val_avg)}"
                    tqdm.write(msg) if tqdm is not None else print(msg)
                    _swanlab_log(swanlab_run, val_avg, step=epoch, prefix="val")

            metric = train_avg["total"] if val_avg is None else val_avg["total"]
            should_save = (
                _is_main_process()
                and (
                    (
                        int(args.max_train_steps) <= 0
                        and int(args.save_every_epochs) > 0
                        and (((epoch + 1) % int(args.save_every_epochs) == 0) or (epoch + 1 == args.epochs))
                    )
                    or (
                        int(args.max_train_steps) > 0
                        and global_step >= int(args.max_train_steps)
                    )
                )
            )
            if should_save:
                is_best = metric < best_val
                if is_best:
                    best_val = metric
                ckpt = {
                    "epoch": epoch,
                    "global_step": int(global_step),
                    "max_train_steps": int(args.max_train_steps),
                    "model": _trainable_state_dict(model),
                    "model_state_format": "trainable_only",
                    "optimizer": {} if (use_deepspeed or not args.save_optimizer_state) else optimizer.state_dict(),
                    "scheduler": {} if (use_deepspeed or not args.save_optimizer_state) else scheduler.state_dict(),
                    "cfg": cfg.__dict__,
                    "best_val": best_val,
                }
                torch.save(ckpt, save_dir / "last.pt")
                if bool(args.save_best_checkpoint) and is_best:
                    torch.save(ckpt, save_dir / "best.pt")

            if use_distributed:
                dist.barrier()
            if reached_max_steps or (int(args.max_train_steps) > 0 and global_step >= int(args.max_train_steps)):
                break
        if _is_main_process():
            done_marker = {
                "status": "complete",
                "epochs": int(args.epochs),
                "global_step": int(global_step),
                "max_train_steps": int(args.max_train_steps),
                "best_val": float(best_val),
            }
            with open(save_dir / "done.marker", "w", encoding="utf-8") as f:
                json.dump(done_marker, f, indent=2, ensure_ascii=False)
    finally:
        if total_pbar is not None:
            total_pbar.close()
        _swanlab_finish(swanlab_run)
        if use_distributed and _ddp_is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
