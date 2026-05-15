from __future__ import annotations

import argparse
import math
import random
import os
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

from data.teacher_dataset_builder import build_records
from model.config import ModelConfig
from model.losses import summarize_losses, world_model_dit_loss
from model.model import PrivilegedTeacherWorldModelDiT

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None


_DEFAULT_CFG = ModelConfig()
LOCAL_CLIP_MODEL_PATH = "/data1/ysq/Worldmodel/model/clip-vit-base-patch32"
LOCAL_DINOV2_MODEL_PATH = "/data1/ysq/Worldmodel/model/dinov2-base"

try:
    from transformers import CLIPTokenizerFast
except Exception:
    CLIPTokenizerFast = None


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class TrajectoryDataset(Dataset):
    REQUIRED_KEYS = [
        "privileged",
        "prev_actions",
        "expert_action",
        "reward",
    ]

    def __init__(
        self,
        records: List[Dict[str, Any]],
        image_size: int,
        seq_len: int,
        privileged_dim: int,
        action_dim: int,
        direction_bins: int = 8,
        distance_bins: int = 6,
        tokenizer_name: Optional[str] = None,
        text_context_length: int = 77,
        random_crop: bool = True,
    ) -> None:
        self.records = records
        self.seq_len = seq_len
        self.privileged_dim = privileged_dim
        self.action_dim = action_dim
        self.direction_bins = direction_bins
        self.distance_bins = distance_bins
        self.text_context_length = text_context_length
        self.random_crop = random_crop
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
            ]
        )
        self.tokenizer = None
        if tokenizer_name is not None:
            if CLIPTokenizerFast is None:
                raise ImportError("transformers 未安装，无法对原始 instructions 做 CLIP tokenization。")
            self.tokenizer = CLIPTokenizerFast.from_pretrained(
                tokenizer_name,
                local_files_only=True,
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

    def _load_rgb_sequence(self, record: Dict[str, Any]) -> torch.Tensor:
        if "images" in record:
            value = record["images"]
            images = torch.load(value, map_location="cpu") if isinstance(value, str) else torch.tensor(value)
            if images.ndim != 4:
                raise ValueError("images must have shape [T, C, H, W].")
            return images.float()
        rgb_paths = record.get("rgb_paths")
        if rgb_paths is None:
            raise KeyError("每条样本必须包含 images 或 rgb_paths。")
        frames = []
        for p in rgb_paths:
            img = Image.open(p).convert("RGB")
            frames.append(self.transform(img))
        if len(frames) == 0:
            raise ValueError("rgb_paths 不能为空。")
        return torch.stack(frames, dim=0)

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
        if tensor is None:
            raise KeyError(f"{self._record_name(record, index)} 缺少必需字段 `{key}`。")
        return tensor

    def _tokenize_if_needed(self, record: Dict[str, Any], seq_len: int, index: int) -> Dict[str, Optional[torch.Tensor]]:
        text_tokens = self._load_tensor_field(record, "text_tokens", torch.long)
        attention_mask = self._load_tensor_field(record, "attention_mask", torch.long)
        if text_tokens is not None:
            return {"text_tokens": text_tokens.long(), "attention_mask": None if attention_mask is None else attention_mask.long()}

        instructions = record.get("instructions")
        if instructions is None:
            raise KeyError(f"{self._record_name(record, index)} 需要提供 text_tokens 或 instructions。")
        if self.tokenizer is None:
            raise ValueError("发现原始 instructions，但未提供 --tokenizer-name。")

        # instructions can be a single string or a per-timestep list of strings.
        enc = self.tokenizer(
            instructions,
            padding="max_length",
            truncation=True,
            max_length=self.text_context_length,
            return_tensors="pt",
        )
        return {"text_tokens": enc["input_ids"].long(), "attention_mask": enc["attention_mask"].long()}

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

    def _crop_or_pad(self, item: Dict[str, Optional[torch.Tensor]]) -> Dict[str, Optional[torch.Tensor]]:
        length = item["images"].shape[0]  # type: ignore[union-attr]
        if length >= self.seq_len:
            start = random.randint(0, length - self.seq_len) if self.random_crop else 0
            end = start + self.seq_len
            cropped: Dict[str, Optional[torch.Tensor]] = {}
            for k, v in item.items():
                if not isinstance(v, torch.Tensor):
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
            if key not in record:
                raise KeyError(f"{self._record_name(record, index)} 缺少必需字段 `{key}`。")

        images = self._load_rgb_sequence(record)
        seq_len = images.shape[0]

        text = self._tokenize_if_needed(record, seq_len, index)
        privileged = self._ensure_2d(
            self._require_tensor_field(record, "privileged", torch.float32, index),
            seq_len,
            self.privileged_dim,
            "privileged",
        )
        prev_actions = self._ensure_2d(
            self._require_tensor_field(record, "prev_actions", torch.float32, index),
            seq_len,
            self.action_dim,
            "prev_actions",
        )
        expert_action = self._ensure_2d(
            self._require_tensor_field(record, "expert_action", torch.float32, index),
            seq_len,
            self.action_dim,
            "expert_action",
        )
        reward = self._ensure_2d(self._require_tensor_field(record, "reward", torch.float32, index), seq_len, 1, "reward")
        item: Dict[str, Optional[torch.Tensor]] = {
            "images": images.float(),
            "text_tokens": text["text_tokens"].long(),  # type: ignore[union-attr]
            "attention_mask": None if text["attention_mask"] is None else text["attention_mask"].long(),
            "privileged": privileged.float(),
            "prev_actions": prev_actions.float(),
            "expert_action": expert_action.float(),
            "reward": reward.float(),
        }
        return self._crop_or_pad(item)


def collate_fn(batch: List[Dict[str, Optional[torch.Tensor]]]) -> Dict[str, Optional[torch.Tensor]]:
    out: Dict[str, Optional[torch.Tensor]] = {}
    for key in batch[0].keys():
        values = [x[key] for x in batch]
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
        "kl",
        "reward",
        "prior_reward",
    ]
    parts = []
    for key in order:
        if key in metrics:
            parts.append(f"{key}={metrics[key]:.6f}")
    for key in sorted(metrics.keys()):
        if key not in order:
            parts.append(f"{key}={metrics[key]:.6f}")
    return " | ".join(parts)


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    if isinstance(model, (torch.nn.DataParallel, DDP)):
        return model.module
    return model


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


def _reduce_metrics(metrics: Dict[str, float], device: torch.device, use_ddp: bool) -> Dict[str, float]:
    if not use_ddp or not metrics:
        return metrics
    keys = sorted(metrics.keys())
    values = torch.tensor([float(metrics[k]) for k in keys], device=device, dtype=torch.float32)
    dist.all_reduce(values, op=dist.ReduceOp.SUM)
    values = values / float(_get_world_size())
    return {k: float(v.item()) for k, v in zip(keys, values)}


@torch.no_grad()
def evaluate(model: PrivilegedTeacherWorldModelDiT, loader: DataLoader, cfg: ModelConfig, device: torch.device) -> Dict[str, float]:
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
            privileged=batch["privileged"],
            prev_actions=batch["prev_actions"],
            attention_mask=batch["attention_mask"],
            expert_action=batch["expert_action"],
            valid_mask=batch["valid_mask"],
            done=batch.get("done"),
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
    parser.add_argument("--save-dir", type=str, required=True)
    parser.add_argument("--tokenizer-name", type=str, default=LOCAL_CLIP_MODEL_PATH)
    parser.add_argument("--clip-text-model-name", type=str, default=LOCAL_CLIP_MODEL_PATH)
    parser.add_argument("--dinov2-model-name", type=str, default=LOCAL_DINOV2_MODEL_PATH)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--privileged-dim", type=int, default=3)
    parser.add_argument("--action-dim", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--diffusion-steps", type=int, default=20)
    parser.add_argument("--sampling-steps", type=int, default=20)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--multi-gpu",
        action="store_true",
        help="Enable multi-GPU training. With torchrun this uses DDP; otherwise falls back to DataParallel.",
    )
    parser.add_argument("--freeze-clip-text", action="store_true", default=True)
    parser.add_argument("--finetune-clip-text", action="store_false", dest="freeze_clip_text")
    parser.add_argument("--freeze-dinov2", action="store_true", default=True)
    parser.add_argument("--finetune-dinov2", action="store_false", dest="freeze_dinov2")
    args = parser.parse_args()

    seed_everything(args.seed + _get_rank())
    save_dir = Path(args.save_dir)
    if _is_main_process():
        save_dir.mkdir(parents=True, exist_ok=True)

    # DDP init (torchrun sets LOCAL_RANK/RANK/WORLD_SIZE)
    use_ddp = args.multi_gpu and torch.cuda.is_available() and _get_world_size() > 1
    if use_ddp:
        torch.cuda.set_device(_get_local_rank())
        dist.init_process_group(backend="nccl", init_method="env://")
        device = torch.device("cuda", _get_local_rank())
    else:
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    cfg = ModelConfig(
        image_size=args.image_size,
        dinov2_model_name=args.dinov2_model_name,
        dinov2_freeze=args.freeze_dinov2,
        clip_text_model_name=args.clip_text_model_name,
        clip_text_freeze=args.freeze_clip_text,
        privileged_dim=args.privileged_dim,
        action_dim=args.action_dim,
        action_diffusion_steps=args.diffusion_steps,
        action_sampling_steps=args.sampling_steps,
        max_vel=args.max_vel,
        max_yaw_rate=args.max_yaw_rate,
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
    )
    if not records:
        raise RuntimeError("No trajectory selected. Check --scene-list / --trajectory-range.")
    rng = random.Random(args.split_seed)
    rng.shuffle(records)
    val_n = max(1, int(len(records) * args.val_ratio)) if len(records) > 1 else 0
    val_records = records[:val_n]
    train_records = records[val_n:] if val_n > 0 else records
    if _is_main_process():
        print(f"[dataset] total={len(records)}, train={len(train_records)}, val={len(val_records)}")

    train_dataset = TrajectoryDataset(
        records=train_records,
        image_size=args.image_size,
        seq_len=args.seq_len,
        privileged_dim=args.privileged_dim,
        action_dim=args.action_dim,
        direction_bins=cfg.direction_bins,
        distance_bins=cfg.distance_bins,
        tokenizer_name=args.tokenizer_name,
        text_context_length=cfg.text_context_length,
        random_crop=True,
    )
    train_sampler = (
        DistributedSampler(
            train_dataset,
            num_replicas=_get_world_size(),
            rank=_get_rank(),
            shuffle=True,
            drop_last=False,
        )
        if use_ddp
        else None
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    val_dataset = TrajectoryDataset(
        records=val_records,
        image_size=args.image_size,
        seq_len=args.seq_len,
        privileged_dim=args.privileged_dim,
        action_dim=args.action_dim,
        direction_bins=cfg.direction_bins,
        distance_bins=cfg.distance_bins,
        tokenizer_name=args.tokenizer_name,
        text_context_length=cfg.text_context_length,
        random_crop=False,
    )
    # For simplicity: run validation only on rank0 under DDP.
    val_loader = None
    if (not use_ddp) or _is_main_process():
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
        )

    model = PrivilegedTeacherWorldModelDiT(cfg).to(device)
    if use_ddp:
        model = DDP(
            model,
            device_ids=[_get_local_rank()],
            output_device=_get_local_rank(),
            find_unused_parameters=True,
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

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    start_epoch = 0
    best_val = math.inf
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        _unwrap_model(model).load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_val = ckpt.get("best_val", best_val)

    try:
        for epoch in range(start_epoch, args.epochs):
            model.train()
            running: Dict[str, float] = {}
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            train_iter = train_loader
            if tqdm is not None and _is_main_process():
                train_iter = tqdm(train_loader, desc=f"Epoch {epoch:03d} train", leave=False, dynamic_ncols=True)
            for step, batch in enumerate(train_iter):
                batch = move_batch_to_device(batch, device)
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(device_type="cuda", enabled=device.type == "cuda"):
                    outputs = model(
                        images=batch["images"],
                        text_tokens=batch["text_tokens"],
                        privileged=batch["privileged"],
                        prev_actions=batch["prev_actions"],
                        attention_mask=batch["attention_mask"],
                        expert_action=batch["expert_action"],
                        valid_mask=batch["valid_mask"],
                        done=batch.get("done"),
                    )
                    global_step = epoch * len(train_loader) + step

                    losses = world_model_dit_loss(
                        outputs,
                        batch,
                        cfg,
                        valid_mask=batch["valid_mask"],
                        global_step=global_step,
                    )
                    loss = losses["total"]

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()

                summary = summarize_losses(losses)
                for k, v in summary.items():
                    running[k] = running.get(k, 0.0) + v

                avg = {k: v / (step + 1) for k, v in running.items()}
                if tqdm is not None and _is_main_process():
                    train_iter.set_postfix(
                        total=f"{avg.get('total', 0.0):.6f}",
                        action=f"{avg.get('action', 0.0):.6f}",
                        kl=f"{avg.get('kl', 0.0):.6f}",
                    )
                elif _is_main_process() and (step + 1) % 20 == 0:
                    print(f"[Epoch {epoch:03d} | Step {step + 1:05d}] {_format_metrics(avg)}")

            scheduler.step()
            train_avg = {k: v / max(len(train_loader), 1) for k, v in running.items()}
            train_avg = _reduce_metrics(train_avg, device, use_ddp)
            if _is_main_process():
                print(f">>> Epoch {epoch:03d} train: {_format_metrics(train_avg)}")

            val_avg = None
            if val_loader is not None:
                val_avg = evaluate(_unwrap_model(model), val_loader, cfg, device)
                if _is_main_process():
                    print(f">>> Epoch {epoch:03d} val:   {_format_metrics(val_avg)}")

            metric = train_avg["total"] if val_avg is None else val_avg["total"]
            ckpt = {
                "epoch": epoch,
                "model": _unwrap_model(model).state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "cfg": cfg.__dict__,
                "best_val": best_val,
            }
            if _is_main_process():
                torch.save(ckpt, save_dir / "last.pt")
                if metric < best_val:
                    best_val = metric
                    ckpt["best_val"] = best_val
                    torch.save(ckpt, save_dir / "best.pt")

            if use_ddp:
                dist.barrier()
    finally:
        if use_ddp and _ddp_is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
