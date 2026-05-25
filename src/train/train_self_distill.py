from __future__ import annotations

import argparse
from dataclasses import fields
import json
import math
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from data.teacher_dataset_builder import build_records
from model.config import ModelConfig
from model.action_loss_utils import weighted_mean_action_squared_error
from model.losses import summarize_losses, world_model_dit_loss
from model.model import PrivilegedTeacherWorldModelDiT
from train.train_teacher import TrajectoryDataset, collate_fn, move_batch_to_device

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None


LOCAL_CLIP_MODEL_PATH = "/data1/ysq/Worldmodel/model/clip-vit-base-patch32"
LOCAL_DINOV2_MODEL_PATH = "/data1/ysq/Worldmodel/model/dinov2-base"
_DEFAULT_CFG = ModelConfig()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    if isinstance(model, (torch.nn.DataParallel, DDP)):
        return model.module
    return model


def _reduce_metrics(metrics: Dict[str, float], device: torch.device, use_ddp: bool) -> Dict[str, float]:
    if not use_ddp or not metrics:
        return metrics
    keys = sorted(metrics.keys())
    values = torch.tensor([float(metrics[k]) for k in keys], device=device, dtype=torch.float32)
    dist.all_reduce(values, op=dist.ReduceOp.SUM)
    values = values / float(_get_world_size())
    return {k: float(v.item()) for k, v in zip(keys, values)}


def _format_metrics(metrics: Dict[str, float]) -> str:
    order = [
        "total",
        "sup_total",
        "feat_distill",
        "action_distill",
        "dit_noise_distill",
        "sup_action",
        "sup_kl",
        "sup_privileged",
        "sup_prior_privileged",
        "sup_rollout_privileged",
    ]
    parts = []
    for key in order:
        if key in metrics:
            parts.append(f"{key}={metrics[key]:.4f}")
    for key in sorted(metrics.keys()):
        if key not in order:
            parts.append(f"{key}={metrics[key]:.4f}")
    return " | ".join(parts)


def _str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    s = str(value).strip().lower()
    if s in ("1", "true", "t", "yes", "y", "on"):
        return True
    if s in ("0", "false", "f", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {value!r}")


def _load_checkpoint_cfg(ckpt_path: str) -> Dict[str, Any]:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    out = dict(ckpt["cfg"]) if isinstance(ckpt, dict) and isinstance(ckpt.get("cfg"), dict) else {}
    if isinstance(ckpt, dict) and "action_sequence_horizon" not in out:
        state = ckpt.get("model", {}) or {}
        token = state.get("actor.action_token_embed")
        if token is None:
            token = state.get("module.actor.action_token_embed")
        if token is not None and getattr(token, "ndim", 0) == 3:
            action_dim = int(out.get("action_dim", ModelConfig().action_dim))
            out["action_sequence_horizon"] = max(int(token.shape[1]) // max(action_dim, 1), 1)
    return out


def make_cfg(args: argparse.Namespace, checkpoint_cfg: Optional[Dict[str, Any]] = None) -> ModelConfig:
    valid_fields = {f.name for f in fields(ModelConfig)}
    cfg_kwargs = {k: v for k, v in (checkpoint_cfg or {}).items() if k in valid_fields}

    # Runtime/path arguments stay script-controlled, while WAM switches and loss
    # weights are inherited from the teacher checkpoint by default.
    cfg_kwargs.update(
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
        max_speed_norm=args.max_speed_norm,
    )
    if args.privileged_fusion_mode is not None:
        cfg_kwargs["privileged_fusion_mode"] = str(args.privileged_fusion_mode)
    if args.action_sequence_horizon is not None:
        cfg_kwargs["action_sequence_horizon"] = int(args.action_sequence_horizon)
    if args.use_target_visual_guidance is not None:
        cfg_kwargs["use_target_visual_guidance"] = bool(args.use_target_visual_guidance)
    if args.use_attention_heatmap is not None:
        cfg_kwargs["use_attention_heatmap"] = bool(args.use_attention_heatmap)
    if args.visual_guidance_fov_deg is not None:
        cfg_kwargs["visual_guidance_fov_deg"] = float(args.visual_guidance_fov_deg)
    if args.attention_heatmap_sigma is not None:
        cfg_kwargs["attention_heatmap_sigma"] = float(args.attention_heatmap_sigma)
    return ModelConfig(**cfg_kwargs)


def load_model_state(model: torch.nn.Module, ckpt_path: str, strict: bool = True) -> Dict[str, Any]:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state, strict=strict)
    if (missing or unexpected) and strict:
        raise RuntimeError(f"Checkpoint load mismatch. missing={missing}, unexpected={unexpected}")
    return ckpt if isinstance(ckpt, dict) else {"model": state}


def freeze_model(model: torch.nn.Module) -> torch.nn.Module:
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def rssm_belief_feat(outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Use deterministic RSSM belief feature: concat(deter, posterior mean).

    This avoids forcing the student to match the teacher's sampled stochastic state,
    which contains random sampling noise.
    """
    posts = outputs["posts"]
    return torch.cat([posts["deter"], posts["mean"]], dim=-1)


def masked_mse_lastdim(x: torch.Tensor, y: torch.Tensor, valid_mask: Optional[torch.Tensor]) -> torch.Tensor:
    per_item = (x - y).pow(2).mean(dim=-1)
    if valid_mask is None:
        return per_item.mean()
    mask = valid_mask.float()
    while mask.ndim < per_item.ndim:
        mask = mask.unsqueeze(-1)
    return (per_item * mask).sum() / mask.sum().clamp(min=1.0)


def dit_noise_distillation_loss(
    student_model: PrivilegedTeacherWorldModelDiT,
    teacher_model: PrivilegedTeacherWorldModelDiT,
    student_feat: torch.Tensor,
    teacher_feat: torch.Tensor,
    expert_action: torch.Tensor,
    valid_mask: Optional[torch.Tensor],
    cfg: ModelConfig,
) -> torch.Tensor:
    """Distill the DiT action head at the noise-prediction level.

    We use the same noisy action x_t and diffusion timestep for teacher and student.
    This is better than comparing sampled actions because the current sample()
    method is inference-only and decorated with torch.no_grad().
    """
    if expert_action.ndim != 3:
        raise ValueError("expert_action must have shape [B, T, A].")
    batch, seq_len, action_dim = expert_action.shape
    device = expert_action.device
    horizon = max(int(getattr(cfg, "action_sequence_horizon", 1)), 1)

    if horizon > 1:
        seq_targets = []
        for k in range(horizon):
            seq_targets.append(
                torch.cat(
                    [expert_action[:, k:], expert_action[:, -1:].expand(-1, k, -1)],
                    dim=1,
                )
            )
        expert_action = torch.stack(seq_targets, dim=2).reshape(batch, seq_len, horizon * action_dim)

    flat_action = expert_action.reshape(batch * seq_len, -1)
    flat_student_feat = student_feat.reshape(batch * seq_len, student_feat.size(-1))
    flat_teacher_feat = teacher_feat.reshape(batch * seq_len, teacher_feat.size(-1))

    flat_mask = None
    if valid_mask is not None:
        flat_mask = valid_mask.reshape(batch * seq_len).float()

    # Use the student's scheduler. Teacher and student should share the same config.
    actor_s = student_model.actor
    actor_t = teacher_model.actor
    t = torch.randint(0, actor_s.num_steps, (flat_action.shape[0],), device=device)
    noise = torch.randn_like(flat_action)
    xt = actor_s.q_sample(flat_action, t, noise)

    student_pred_noise = actor_s(flat_student_feat, xt, t)
    with torch.no_grad():
        teacher_pred_noise = actor_t(flat_teacher_feat, xt, t)

    per_item = weighted_mean_action_squared_error(student_pred_noise, teacher_pred_noise, cfg)
    if flat_mask is None:
        return per_item.mean()
    return (per_item * flat_mask).sum() / flat_mask.sum().clamp(min=1.0)


def action_distillation_loss(
    student_out: Dict[str, torch.Tensor],
    teacher_out: Dict[str, torch.Tensor],
    valid_mask: Optional[torch.Tensor],
    cfg: ModelConfig,
) -> torch.Tensor:
    student_action = student_out.get("policy_action_sequence")
    teacher_action = teacher_out.get("policy_action_sequence")
    device = next(iter(student_out["posts"].values())).device
    dtype = next(iter(student_out["posts"].values())).dtype
    if student_action is not None and teacher_action is not None:
        with torch.no_grad():
            target = teacher_action.detach()
        terms = []
        for k in range(student_action.size(2)):
            per_t = weighted_mean_action_squared_error(student_action[:, :, k], target[:, :, k], cfg).unsqueeze(-1)
            terms.append(masked_mean(per_t, valid_mask))
        return torch.stack(terms).mean()

    student_action = student_out.get("policy_action")
    teacher_action = teacher_out.get("policy_action")
    if student_action is None or teacher_action is None:
        return torch.zeros((), device=device, dtype=dtype)

    with torch.no_grad():
        target = teacher_action.detach()
    per_t = weighted_mean_action_squared_error(student_action, target, cfg).unsqueeze(-1)
    return masked_mean(per_t, valid_mask)


def self_distill_losses(
    student_out: Dict[str, torch.Tensor],
    teacher_out: Dict[str, torch.Tensor],
    student_model: PrivilegedTeacherWorldModelDiT,
    teacher_model: PrivilegedTeacherWorldModelDiT,
    batch: Dict[str, Any],
    cfg: ModelConfig,
    args: argparse.Namespace,
) -> Dict[str, torch.Tensor]:
    valid_mask = batch.get("valid_mask")
    sup = world_model_dit_loss(student_out, batch, cfg, valid_mask=valid_mask)

    losses: Dict[str, torch.Tensor] = {f"sup_{k}": v for k, v in sup.items()}

    student_belief = rssm_belief_feat(student_out)
    with torch.no_grad():
        teacher_belief = rssm_belief_feat(teacher_out)

    feat_loss = masked_mse_lastdim(student_belief, teacher_belief, valid_mask)
    action_loss = action_distillation_loss(student_out, teacher_out, valid_mask, cfg)

    if args.dit_noise_distill_weight > 0.0 and cfg.use_diffusion_actor:
        dit_loss = dit_noise_distillation_loss(
            student_model=student_model,
            teacher_model=teacher_model,
            student_feat=student_belief,
            teacher_feat=teacher_belief,
            expert_action=batch["expert_action"],
            valid_mask=valid_mask,
            cfg=cfg,
        )
    else:
        dit_loss = torch.zeros((), device=feat_loss.device, dtype=feat_loss.dtype)

    total = (
        args.sup_weight * sup["total"]
        + args.feat_distill_weight * feat_loss
        + args.action_distill_weight * action_loss
        + args.dit_noise_distill_weight * dit_loss
    )

    losses["feat_distill"] = feat_loss
    losses["action_distill"] = action_loss
    losses["dit_noise_distill"] = dit_loss
    losses["total"] = total
    return losses


@torch.no_grad()
def evaluate_distill(
    student: PrivilegedTeacherWorldModelDiT,
    teacher: PrivilegedTeacherWorldModelDiT,
    loader: DataLoader,
    cfg: ModelConfig,
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, float]:
    student.eval()
    teacher.eval()
    acc: Dict[str, float] = {}
    count = 0

    val_iter = loader
    if tqdm is not None:
        val_iter = tqdm(loader, desc="val", leave=False, dynamic_ncols=True)

    for batch in val_iter:
        batch = move_batch_to_device(batch, device)

        teacher_out = teacher(
            images=batch["images"],
            text_tokens=batch["text_tokens"],
            privileged=batch["privileged"],
            prev_actions=batch["prev_actions"],
            attention_mask=batch["attention_mask"],
            attention_heatmaps=batch.get("attention_heatmaps"),
            expert_action=batch["expert_action"],
            valid_mask=batch["valid_mask"],
            done=batch.get("done"),
        )

        student_out = student(
            images=batch["images"],
            text_tokens=batch["text_tokens"],
            privileged=batch["privileged"],
            prev_actions=batch["prev_actions"],
            attention_mask=batch["attention_mask"],
            attention_heatmaps=batch.get("attention_heatmaps"),
            expert_action=batch["expert_action"],
            valid_mask=batch["valid_mask"],
            done=batch.get("done"),
        )

        losses = self_distill_losses(
            student_out=student_out,
            teacher_out=teacher_out,
            student_model=student,
            teacher_model=teacher,
            batch=batch,
            cfg=cfg,
            args=args,
        )
        summary = summarize_losses(losses)
        for k, v in summary.items():
            acc[k] = acc.get(k, 0.0) + v
        count += 1

    return {k: v / max(count, 1) for k, v in acc.items()}


def build_loaders(
    args: argparse.Namespace,
    cfg: ModelConfig,
    use_ddp: bool,
) -> tuple[DataLoader, Optional[DataLoader], int, int, int]:
    scene_list = [s.strip() for s in args.scene_list.split(",") if s.strip()]
    if not scene_list:
        raise ValueError("--scene-list is empty.")

    records = build_records(
        Path(args.dataset_root),
        scene_list,
        args.trajectory_range.strip(),
        max_vel=cfg.max_vel,
        max_yaw_rate=cfg.max_yaw_rate,
        max_speed_norm=cfg.max_speed_norm,
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

    train_dataset = TrajectoryDataset(
        records=train_records,
        image_size=cfg.image_size,
        seq_len=args.seq_len,
        privileged_dim=cfg.privileged_dim,
        action_dim=cfg.action_dim,
        direction_bins=cfg.direction_bins,
        distance_bins=cfg.distance_bins,
        tokenizer_name=args.tokenizer_name,
        text_context_length=cfg.text_context_length,
        random_crop=True,
        use_target_visual_guidance=cfg.use_target_visual_guidance,
        use_attention_heatmap=cfg.use_attention_heatmap,
        visual_guidance_fov_deg=cfg.visual_guidance_fov_deg,
        attention_heatmap_sigma=cfg.attention_heatmap_sigma,
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
        persistent_workers=args.num_workers > 0,
        prefetch_factor=2 if args.num_workers > 0 else None,
        collate_fn=collate_fn,
    )
    train_loader.sampler_for_epoch = train_sampler  # type: ignore[attr-defined]

    val_loader = None
    if val_records and ((not use_ddp) or _is_main_process()):
        val_dataset = TrajectoryDataset(
            records=val_records,
            image_size=cfg.image_size,
            seq_len=args.seq_len,
            privileged_dim=cfg.privileged_dim,
            action_dim=cfg.action_dim,
            direction_bins=cfg.direction_bins,
            distance_bins=cfg.distance_bins,
            tokenizer_name=args.tokenizer_name,
            text_context_length=cfg.text_context_length,
            random_crop=False,
            use_target_visual_guidance=cfg.use_target_visual_guidance,
            use_attention_heatmap=cfg.use_attention_heatmap,
            visual_guidance_fov_deg=cfg.visual_guidance_fov_deg,
            attention_heatmap_sigma=cfg.attention_heatmap_sigma,
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

    return train_loader, val_loader, len(records), len(train_records), len(val_records)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Privileged self-distillation for Teacher World Model + DiT action head.")

    # Dataset / paths
    parser.add_argument("--dataset-root", type=str, required=True)
    parser.add_argument("--scene-list", type=str, required=True)
    parser.add_argument("--trajectory-range", type=str, default="")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--save-dir", type=str, required=True)
    parser.add_argument("--teacher-ckpt", type=str, required=True)
    parser.add_argument("--resume", type=str, default=None)

    # Model paths
    parser.add_argument("--tokenizer-name", type=str, default=LOCAL_CLIP_MODEL_PATH)
    parser.add_argument("--clip-text-model-name", type=str, default=LOCAL_CLIP_MODEL_PATH)
    parser.add_argument("--dinov2-model-name", type=str, default=LOCAL_DINOV2_MODEL_PATH)

    # Model / data config
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--privileged-dim", type=int, default=3)
    parser.add_argument("--action-dim", type=int, default=4)
    parser.add_argument("--max-vel", type=float, default=_DEFAULT_CFG.max_vel)
    parser.add_argument("--max-yaw-rate", type=float, default=_DEFAULT_CFG.max_yaw_rate)
    parser.add_argument("--max-speed-norm", type=float, default=_DEFAULT_CFG.max_speed_norm)
    parser.add_argument("--action-sequence-horizon", type=int, default=None)
    parser.add_argument("--diffusion-steps", type=int, default=20)
    parser.add_argument("--sampling-steps", type=int, default=20)
    parser.add_argument(
        "--privileged-fusion-mode",
        type=str,
        default=None,
        choices=["attention", "concat"],
        help="Override teacher checkpoint cfg.privileged_fusion_mode. Default uses checkpoint cfg.",
    )
    parser.add_argument("--use-target-visual-guidance", type=_str2bool, default=None)
    parser.add_argument("--use-attention-heatmap", type=_str2bool, default=None)
    parser.add_argument("--visual-guidance-fov-deg", type=float, default=None)
    parser.add_argument("--attention-heatmap-sigma", type=float, default=None)

    # Training config
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--multi-gpu", action="store_true")

    # Encoder freezing
    parser.add_argument("--freeze-clip-text", action="store_true", default=True)
    parser.add_argument("--finetune-clip-text", action="store_false", dest="freeze_clip_text")
    parser.add_argument("--freeze-dinov2", action="store_true", default=False)

    # Distillation weights. Keep the first version clean.
    parser.add_argument("--sup-weight", type=float, default=1.0)
    parser.add_argument("--feat-distill-weight", type=float, default=0.1)
    parser.add_argument("--action-distill-weight", type=float, default=0.5)
    parser.add_argument("--dit-noise-distill-weight", type=float, default=0.0)

    # Student initialization
    parser.add_argument(
        "--init-student-from-teacher",
        action="store_true",
        default=False,
        help="Initialize student from teacher checkpoint before self-distillation.",
    )
    parser.add_argument(
        "--student-init-random",
        action="store_false",
        dest="init_student_from_teacher",
        help="Do not initialize the student from teacher checkpoint.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    seed_everything(args.seed + _get_rank())

    save_dir = Path(args.save_dir)
    if _is_main_process():
        save_dir.mkdir(parents=True, exist_ok=True)

    use_ddp = args.multi_gpu and torch.cuda.is_available() and _get_world_size() > 1
    if use_ddp:
        torch.cuda.set_device(_get_local_rank())
        dist.init_process_group(backend="nccl", init_method="env://")
        device = torch.device("cuda", _get_local_rank())
    else:
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    teacher_cfg = _load_checkpoint_cfg(args.teacher_ckpt)
    cfg = make_cfg(args, teacher_cfg)

    train_loader, val_loader, total_n, train_n, val_n = build_loaders(args, cfg, use_ddp=use_ddp)
    train_sampler = getattr(train_loader, "sampler_for_epoch", None)

    if _is_main_process():
        if val_n > 0:
            print(f"[dataset] total={total_n}, train={train_n}, val={val_n}")
        else:
            print(f"[dataset] total={total_n}, train={train_n}")
        print(
            f"[cfg] use_diffusion_actor={cfg.use_diffusion_actor}, "
            f"privileged_input=disabled, fusion={cfg.privileged_fusion_mode}, "
            f"train_next_privileged={cfg.train_next_privileged}, train_rollout={cfg.train_rollout}"
        )
        print(
            f"[distill] sup={args.sup_weight}, feat={args.feat_distill_weight}, "
            f"action={args.action_distill_weight}, dit_noise={args.dit_noise_distill_weight}"
        )

    teacher = PrivilegedTeacherWorldModelDiT(cfg).to(device)
    load_model_state(teacher, args.teacher_ckpt, strict=False)
    freeze_model(teacher)
    if _is_main_process():
        print(f"[teacher] loaded frozen teacher: {args.teacher_ckpt}")

    student = PrivilegedTeacherWorldModelDiT(cfg).to(device)
    if args.init_student_from_teacher:
        load_model_state(student, args.teacher_ckpt, strict=False)
        if _is_main_process():
            print("[student] initialized from teacher checkpoint")
    else:
        if _is_main_process():
            print("[student] random initialization")

    optimizer = torch.optim.AdamW(
        [p for p in student.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    start_epoch = 0
    best_val = math.inf
    history: List[Dict[str, Any]] = []

    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        student.load_state_dict(ckpt["model"], strict=True)
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = int(ckpt["epoch"]) + 1
        best_val = float(ckpt.get("best_val", best_val))
        history = list(ckpt.get("history", []))
        if _is_main_process():
            print(f"[resume] {args.resume}, start_epoch={start_epoch}, best_val={best_val:.6f}")

    if use_ddp:
        student = DDP(
            student,
            device_ids=[_get_local_rank()],
            output_device=_get_local_rank(),
            find_unused_parameters=False,
            broadcast_buffers=False,
        )
        if _is_main_process():
            print(f"[train] DDP enabled on world_size={_get_world_size()} (local_rank={_get_local_rank()})")
    else:
        if args.multi_gpu and device.type == "cuda" and torch.cuda.device_count() > 1:
            student = torch.nn.DataParallel(student)
            if _is_main_process():
                print(f"[train] DataParallel enabled on {torch.cuda.device_count()} GPUs")
        else:
            if _is_main_process():
                print(f"[train] Device: {device}")

    for epoch in range(start_epoch, args.epochs):
        _unwrap_model(student).train()
        teacher.eval()
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        running: Dict[str, float] = {}
        train_iter = train_loader
        if tqdm is not None and _is_main_process():
            train_iter = tqdm(train_loader, desc=f"Epoch {epoch:03d} self-distill", leave=False, dynamic_ncols=True)

        for step, batch in enumerate(train_iter):
            batch = move_batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type="cuda", enabled=device.type == "cuda"):
                with torch.no_grad():
                    teacher_out = teacher(
                        images=batch["images"],
                        text_tokens=batch["text_tokens"],
                        privileged=batch["privileged"],
                        prev_actions=batch["prev_actions"],
                        attention_mask=batch["attention_mask"],
                        attention_heatmaps=batch.get("attention_heatmaps"),
                        expert_action=batch["expert_action"],
                        valid_mask=batch["valid_mask"],
                        done=batch.get("done"),
                    )

                student_out = student(
                    images=batch["images"],
                    text_tokens=batch["text_tokens"],
                    privileged=batch["privileged"],
                    prev_actions=batch["prev_actions"],
                    attention_mask=batch["attention_mask"],
                    attention_heatmaps=batch.get("attention_heatmaps"),
                    expert_action=batch["expert_action"],
                    valid_mask=batch["valid_mask"],
                    done=batch.get("done"),
                )

                losses = self_distill_losses(
                    student_out=student_out,
                    teacher_out=teacher_out,
                    student_model=_unwrap_model(student),
                    teacher_model=teacher,
                    batch=batch,
                    cfg=cfg,
                    args=args,
                )
                loss = losses["total"]

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            clip_grad_norm_(student.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            summary = summarize_losses(losses)
            for k, v in summary.items():
                running[k] = running.get(k, 0.0) + v

            avg = {k: v / (step + 1) for k, v in running.items()}
            if tqdm is not None and _is_main_process():
                train_iter.set_postfix(
                    total=f"{avg.get('total', 0.0):.4f}",
                    sup=f"{avg.get('sup_total', 0.0):.4f}",
                    feat=f"{avg.get('feat_distill', 0.0):.4f}",
                    action=f"{avg.get('action_distill', 0.0):.4f}",
                    dit=f"{avg.get('dit_noise_distill', 0.0):.4f}",
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
            val_avg = evaluate_distill(
                student=_unwrap_model(student),
                teacher=teacher,
                loader=val_loader,
                cfg=cfg,
                args=args,
                device=device,
            )
            if _is_main_process():
                print(f">>> Epoch {epoch:03d} val:   {_format_metrics(val_avg)}")

        metric = train_avg["total"] if val_avg is None else val_avg["total"]
        if _is_main_process():
            history.append({
                "epoch": epoch,
                "train": train_avg,
                "val": val_avg,
                "metric": metric,
            })
            ckpt = {
                "epoch": epoch,
                "model": _unwrap_model(student).state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "cfg": cfg.__dict__,
                "args": vars(args),
                "best_val": best_val,
                "history": history,
            }
            torch.save(ckpt, save_dir / "last.pt")

            if metric < best_val:
                best_val = metric
                ckpt["best_val"] = best_val
                torch.save(ckpt, save_dir / "best.pt")
                print(f"[save] best.pt updated: metric={metric:.6f}")

            with open(save_dir / "history.json", "w", encoding="utf-8") as f:
                json.dump(history, f, indent=2, ensure_ascii=False)

        if use_ddp:
            dist.barrier()

    if use_ddp and _ddp_is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
