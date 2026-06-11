from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import fields, replace
import json
import math
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from data.teacher_dataset_builder import build_records
from model.config import ModelConfig, migrate_legacy_config
from model.action_loss_utils import weighted_mean_action_squared_error
from model.losses import summarize_losses, world_model_dit_loss
from model.model import TeacherWorldModelDiT, migrate_legacy_state_dict_keys
from train.train_teacher import TrajectoryDataset, _wan_latent_cache_stats, collate_fn, move_batch_to_device

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None

try:
    import deepspeed
except Exception:
    deepspeed = None


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


def _reduce_metrics(metrics: Dict[str, float], device: torch.device, use_ddp: bool) -> Dict[str, float]:
    if not use_ddp or not metrics or not _ddp_is_initialized():
        return metrics
    keys = sorted(metrics.keys())
    values = torch.tensor([float(metrics[k]) for k in keys], device=device, dtype=torch.float32)
    dist.all_reduce(values, op=dist.ReduceOp.SUM)
    values = values / float(_get_world_size())
    return {k: float(v.item()) for k, v in zip(keys, values)}


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
                "teacher_ckpt": args.teacher_ckpt,
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
        "sup_kl",
        "sup_next_target_relative",
        "sup_prior_next_target_relative",
        "sup_video_x0",
        "sup_x0_action",
        "action_distill",
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


def _format_metrics(metrics: Dict[str, float]) -> str:
    order = [
        "total",
        "sup_total",
        "feat_distill",
        "action_distill",
        "sup_action",
        "sup_video",
        "sup_kl",
        "sup_next_target_relative",
        "sup_prior_next_target_relative",
    ]
    hidden_when_zero = {
        "action_distill",
        "sup_kl",
        "sup_next_target_relative",
        "sup_prior_next_target_relative",
        "sup_video_x0",
        "sup_x0_action",
    }
    parts = []
    for key in order:
        if key in metrics and not (key in hidden_when_zero and abs(metrics[key]) < 1e-12):
            parts.append(f"{key}={metrics[key]:.4f}")
    for key in sorted(metrics.keys()):
        if key not in order:
            if key in hidden_when_zero and abs(metrics[key]) < 1e-12:
                continue
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


def _grad_scaler_enabled(device: torch.device, cfg: ModelConfig, use_deepspeed: bool = False) -> bool:
    return device.type == "cuda" and (not use_deepspeed) and _cuda_amp_dtype(cfg) == torch.float16


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


def _set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = float(lr)


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


def _finite_debug_summary(losses: Dict[str, torch.Tensor], outputs: Dict[str, torch.Tensor]) -> str:
    parts = []
    bad_losses = []
    for name, value in losses.items():
        if torch.is_tensor(value) and value.is_floating_point() and not torch.isfinite(value.detach()).all():
            bad_losses.append(name)
    if bad_losses:
        parts.append("bad losses=" + ",".join(bad_losses))

    bad_outputs = []
    for name, value in outputs.items():
        if torch.is_tensor(value) and value.is_floating_point() and not torch.isfinite(value.detach()).all():
            bad_outputs.append(f"{name}{tuple(value.shape)}")
    if bad_outputs:
        parts.append("bad outputs=" + ",".join(bad_outputs[:12]))
    return "; ".join(parts) if parts else "no non-finite tensor located"


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
    checkpoint_cfg = migrate_legacy_config(checkpoint_cfg or {})
    cfg_kwargs = {k: v for k, v in checkpoint_cfg.items() if k in valid_fields}

    # Runtime/path arguments stay script-controlled, while WAM switches and loss
    # weights are inherited from the teacher checkpoint by default.
    cfg_kwargs.update(
        image_size=args.image_size,
        target_relative_dim=args.target_relative_dim,
        action_dim=args.action_dim,
        action_diffusion_steps=args.diffusion_steps,
        action_sampling_steps=args.sampling_steps,
        max_vel=args.max_vel,
        max_yaw_rate=args.max_yaw_rate,
        max_speed_norm=args.max_speed_norm,
    )
    if args.target_token_fusion_mode is not None:
        cfg_kwargs["target_token_fusion_mode"] = str(args.target_token_fusion_mode)
    if args.action_sequence_horizon is not None:
        cfg_kwargs["action_sequence_horizon"] = int(args.action_sequence_horizon)
    if args.action_video_freq_ratio is not None:
        cfg_kwargs["fastwam_action_video_freq_ratio"] = max(int(args.action_video_freq_ratio), 1)
    if args.use_target_visual_guidance is not None:
        cfg_kwargs["use_target_visual_guidance"] = bool(args.use_target_visual_guidance)
    if args.use_attention_heatmap is not None:
        cfg_kwargs["use_attention_heatmap"] = bool(args.use_attention_heatmap)
    if args.visual_guidance_fov_deg is not None:
        cfg_kwargs["visual_guidance_fov_deg"] = float(args.visual_guidance_fov_deg)
    if args.attention_heatmap_sigma is not None:
        cfg_kwargs["attention_heatmap_sigma"] = float(args.attention_heatmap_sigma)
    if args.use_wan22_encoders is not None:
        cfg_kwargs["use_wan22_encoders"] = bool(args.use_wan22_encoders)
    if args.wan22_model_base_path is not None:
        cfg_kwargs["wan22_model_base_path"] = str(args.wan22_model_base_path)
    if args.wan22_fastwam_src_path is not None:
        cfg_kwargs["wan22_fastwam_src_path"] = str(args.wan22_fastwam_src_path)
    if args.wan22_skip_download is not None:
        cfg_kwargs["wan22_skip_download"] = bool(args.wan22_skip_download)
    if args.wan22_text_context_length is not None:
        cfg_kwargs["wan22_text_context_length"] = int(args.wan22_text_context_length)
        if bool(cfg_kwargs.get("use_wan22_encoders", False)):
            cfg_kwargs["text_context_length"] = int(args.wan22_text_context_length)
    if args.wan22_text_encode_batch_size is not None:
        cfg_kwargs["wan22_text_encode_batch_size"] = int(args.wan22_text_encode_batch_size)
    if args.fastwam_heatmap_context_grid is not None:
        cfg_kwargs["fastwam_heatmap_context_grid"] = int(args.fastwam_heatmap_context_grid)
    if args.use_target_belief_tracker is not None:
        cfg_kwargs["use_target_belief_tracker"] = bool(args.use_target_belief_tracker)
    if args.target_belief_token_scale is not None:
        cfg_kwargs["target_belief_token_scale"] = float(args.target_belief_token_scale)
    if args.target_belief_update_rate is not None:
        cfg_kwargs["target_belief_update_rate"] = float(args.target_belief_update_rate)
    if args.target_belief_min_confidence is not None:
        cfg_kwargs["target_belief_min_confidence"] = float(args.target_belief_min_confidence)
    if args.target_belief_temperature is not None:
        cfg_kwargs["target_belief_temperature"] = float(args.target_belief_temperature)
    if args.target_belief_loss_weight is not None:
        cfg_kwargs["target_belief_loss_weight"] = float(args.target_belief_loss_weight)
    if args.target_belief_motion_weight is not None:
        cfg_kwargs["target_belief_motion_weight"] = float(args.target_belief_motion_weight)
    if args.target_belief_update_sharpness is not None:
        cfg_kwargs["target_belief_update_sharpness"] = float(args.target_belief_update_sharpness)
    if args.fastwam_skip_dit_load_from_pretrain is not None:
        cfg_kwargs["fastwam_skip_dit_load_from_pretrain"] = bool(args.fastwam_skip_dit_load_from_pretrain)
    if args.fastwam_action_dit_pretrained_path is not None:
        cfg_kwargs["fastwam_action_dit_pretrained_path"] = str(args.fastwam_action_dit_pretrained_path)
    if args.fastwam_mot_checkpoint_mixed_attn is not None:
        cfg_kwargs["fastwam_mot_checkpoint_mixed_attn"] = bool(args.fastwam_mot_checkpoint_mixed_attn)
    return ModelConfig(**cfg_kwargs)


def make_student_cfg(args: argparse.Namespace, teacher_cfg: ModelConfig) -> ModelConfig:
    updates: Dict[str, Any] = {}
    if args.student_use_target_visual_guidance is not None:
        updates["use_target_visual_guidance"] = bool(args.student_use_target_visual_guidance)
    if args.student_use_attention_heatmap is not None:
        updates["use_attention_heatmap"] = bool(args.student_use_attention_heatmap)
    if args.student_use_target_belief_tracker is not None:
        updates["use_target_belief_tracker"] = bool(args.student_use_target_belief_tracker)
    return replace(teacher_cfg, **updates)


def load_model_state(model: torch.nn.Module, ckpt_path: str, strict: bool = True) -> Dict[str, Any]:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    state = migrate_legacy_state_dict_keys(state)
    missing, unexpected = model.load_state_dict(state, strict=strict)
    if (missing or unexpected) and strict:
        raise RuntimeError(f"Checkpoint load mismatch. missing={missing}, unexpected={unexpected}")
    return ckpt if isinstance(ckpt, dict) else {"model": state}


def freeze_model(model: torch.nn.Module) -> torch.nn.Module:
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def belief_feat(outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Use the model latent feature for distillation.

    RSSM checkpoints expose posterior belief states; Fast-WAM-style checkpoints
    expose direct observation features and have no RSSM state.
    """
    posts = outputs.get("posts")
    if posts is not None:
        return torch.cat([posts["deter"], posts["mean"]], dim=-1)
    return outputs["feat"]


def masked_mse_lastdim(x: torch.Tensor, y: torch.Tensor, valid_mask: Optional[torch.Tensor]) -> torch.Tensor:
    if x.shape[:-1] != y.shape[:-1]:
        if x.ndim >= 3 and y.ndim >= 3 and x.size(0) == y.size(0):
            dst_t = min(x.size(1), y.size(1))
            if x.size(1) != dst_t:
                idx = torch.linspace(0, x.size(1) - 1, dst_t, device=x.device).round().long()
                x = x[:, idx]
            if y.size(1) != dst_t:
                idx = torch.linspace(0, y.size(1) - 1, dst_t, device=y.device).round().long()
                y = y[:, idx]
        else:
            raise ValueError(f"Cannot align feature shapes {tuple(x.shape)} and {tuple(y.shape)}.")
    per_item = (x - y).pow(2).mean(dim=-1)
    if valid_mask is None:
        return per_item.mean()
    mask = valid_mask.float()
    if mask.ndim >= 2 and per_item.ndim >= 2 and mask.size(1) != per_item.size(1):
        src_t = mask.size(1)
        dst_t = per_item.size(1)
        idx = torch.linspace(0, src_t - 1, dst_t, device=mask.device).round().long()
        mask = mask[:, idx]
    while mask.ndim < per_item.ndim:
        mask = mask.unsqueeze(-1)
    return (per_item * mask).sum() / mask.sum().clamp(min=1.0)


def align_time_mask(valid_mask: Optional[torch.Tensor], length: int, device: torch.device) -> Optional[torch.Tensor]:
    if valid_mask is None:
        return None
    mask = valid_mask.float().to(device)
    if mask.ndim >= 2 and mask.size(1) != length:
        if mask.size(1) > length:
            mask = mask[:, :length]
        else:
            pad = mask[:, -1:].expand(-1, length - mask.size(1))
            mask = torch.cat([mask, pad], dim=1)
    return mask


def _make_action_sequence_target(
    expert_action: torch.Tensor,
    horizon: int,
    action_dim: int,
) -> torch.Tensor:
    if expert_action.ndim != 3:
        raise ValueError("expert_action must have shape [B, T, A].")
    batch, seq_len, action_dim = expert_action.shape
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
    return expert_action


def _weighted_flat_action_sequence_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    cfg: ModelConfig,
) -> torch.Tensor:
    action_dim = int(getattr(cfg, "action_dim", pred.shape[-1]))
    if pred.shape[-1] % action_dim != 0:
        return (pred.float() - target.float()).pow(2).mean(dim=-1)
    horizon = pred.shape[-1] // action_dim
    pred_seq = pred.reshape(pred.shape[0], horizon, action_dim)
    tgt_seq = target.reshape(target.shape[0], horizon, action_dim)
    return weighted_mean_action_squared_error(pred_seq, tgt_seq, cfg).mean(dim=-1)


def action_distillation_loss(
    student_out: Dict[str, torch.Tensor],
    teacher_out: Dict[str, torch.Tensor],
    valid_mask: Optional[torch.Tensor],
    cfg: ModelConfig,
) -> torch.Tensor:
    ref = student_out.get("feat", student_out.get("obs_embed"))
    device = ref.device
    dtype = ref.dtype
    student_action = student_out.get("policy_action_sequence")
    teacher_action = teacher_out.get("policy_action_sequence")
    if student_action is not None and teacher_action is not None:
        with torch.no_grad():
            target = teacher_action.detach()
        if student_action.ndim == 4 and teacher_action.ndim == 4:
            horizon = min(student_action.size(2), teacher_action.size(2))
            mask = align_time_mask(valid_mask, student_action.size(1), student_action.device)
            terms = []
            for k in range(horizon):
                per_t = weighted_mean_action_squared_error(student_action[:, :, k], target[:, :, k], cfg).unsqueeze(-1)
                terms.append(masked_mean(per_t, mask))
            return torch.stack(terms).mean()
        if student_action.ndim == 4 and teacher_action.ndim == 3:
            target = teacher_action.unsqueeze(2)
        if student_action.ndim == 3 and teacher_action.ndim == 4:
            target = teacher_action[:, :, 0]
        if student_action.ndim == 3 and target.ndim == 3:
            per_t = weighted_mean_action_squared_error(student_action, target, cfg).unsqueeze(-1)
            return masked_mean(per_t, align_time_mask(valid_mask, student_action.size(1), student_action.device))
        terms = []
        mask = align_time_mask(valid_mask, student_action.size(1), student_action.device)
        horizon = min(student_action.size(2), target.size(2))
        for k in range(horizon):
            per_t = weighted_mean_action_squared_error(student_action[:, :, k], target[:, :, k], cfg).unsqueeze(-1)
            terms.append(masked_mean(per_t, mask))
        return torch.stack(terms).mean()

    student_action = student_out.get("policy_action")
    teacher_action = teacher_out.get("policy_action")
    if student_action is None or teacher_action is None:
        return torch.zeros((), device=device, dtype=dtype)

    with torch.no_grad():
        target = teacher_action.detach()
    per_t = weighted_mean_action_squared_error(student_action, target, cfg).unsqueeze(-1)
    return masked_mean(per_t, align_time_mask(valid_mask, student_action.size(1), student_action.device))


def self_distill_losses(
    student_out: Dict[str, torch.Tensor],
    teacher_out: Dict[str, torch.Tensor],
    student_model: TeacherWorldModelDiT,
    teacher_model: TeacherWorldModelDiT,
    batch: Dict[str, Any],
    cfg: ModelConfig,
    args: argparse.Namespace,
) -> Dict[str, torch.Tensor]:
    valid_mask = batch.get("valid_mask")
    sup = world_model_dit_loss(student_out, batch, cfg, valid_mask=valid_mask)

    losses: Dict[str, torch.Tensor] = {f"sup_{k}": v for k, v in sup.items()}

    student_belief = belief_feat(student_out)
    with torch.no_grad():
        teacher_belief = belief_feat(teacher_out)

    feat_loss = masked_mse_lastdim(student_belief, teacher_belief, valid_mask)
    use_fastwam = bool(getattr(cfg, "use_fastwam_mot", False))
    action_loss = torch.zeros((), device=feat_loss.device, dtype=feat_loss.dtype)
    action_loss = action_distillation_loss(student_out, teacher_out, valid_mask, cfg)
    total = (
        args.sup_weight * sup["total"]
        + args.feat_distill_weight * feat_loss
        + args.action_distill_weight * action_loss
    )

    losses["feat_distill"] = feat_loss
    losses["action_distill"] = action_loss
    losses["total"] = total
    return losses


@torch.no_grad()
def evaluate_distill(
    student: TeacherWorldModelDiT,
    teacher: TeacherWorldModelDiT,
    loader: DataLoader,
    student_cfg: ModelConfig,
    teacher_cfg: ModelConfig,
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

        teacher_policy_target = batch["expert_action"] if args.action_distill_weight > 0.0 else None
        teacher_out = teacher(
            images=batch["images"],
            text_tokens=batch["text_tokens"],
            target_relative=batch["target_relative"],
            prev_actions=batch["prev_actions"],
            attention_mask=batch["attention_mask"],
            attention_heatmaps=batch.get("attention_heatmaps"),
            expert_action=teacher_policy_target,
            valid_mask=batch["valid_mask"] if teacher_policy_target is not None else None,
            done=batch.get("done"),
            instructions=batch.get("instructions"),
            video_latents=batch.get("video_latents"),
            reference_target_relative=batch.get("reference_target_relative"),
            reference_images=batch.get("reference_images"),
        )

        student_out = student(
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

        losses = self_distill_losses(
            student_out=student_out,
            teacher_out=teacher_out,
            student_model=student,
            teacher_model=teacher,
            batch=batch,
            cfg=student_cfg,
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
        target_relative_dim=cfg.target_relative_dim,
        action_dim=cfg.action_dim,
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
            target_relative_dim=cfg.target_relative_dim,
            action_dim=cfg.action_dim,
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

    return train_loader, val_loader, len(records), len(train_records), len(val_records)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Self-distillation for FastWAM teacher/student models.")

    # Dataset / paths
    parser.add_argument("--dataset-root", type=str, required=True)
    parser.add_argument("--scene-list", type=str, required=True)
    parser.add_argument("--trajectory-range", type=str, default="")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--save-dir", type=str, required=True)
    parser.add_argument("--teacher-ckpt", type=str, required=True)
    parser.add_argument("--resume", type=str, default=None)

    # Model / data config
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--target-relative-dim", type=int, default=3)
    parser.add_argument("--action-dim", type=int, default=4)
    parser.add_argument("--max-vel", type=float, default=_DEFAULT_CFG.max_vel)
    parser.add_argument("--max-yaw-rate", type=float, default=_DEFAULT_CFG.max_yaw_rate)
    parser.add_argument("--max-speed-norm", type=float, default=_DEFAULT_CFG.max_speed_norm)
    parser.add_argument("--action-sequence-horizon", type=int, default=None)
    parser.add_argument("--action-video-freq-ratio", type=int, default=None)
    parser.add_argument("--diffusion-steps", type=int, default=20)
    parser.add_argument("--sampling-steps", type=int, default=20)
    parser.add_argument(
        "--target-token-fusion-mode",
        type=str,
        default=None,
        choices=["attention", "concat"],
        help="Override teacher checkpoint cfg.target_token_fusion_mode. Default uses checkpoint cfg.",
    )
    parser.add_argument("--use-target-visual-guidance", type=_str2bool, default=None)
    parser.add_argument("--use-attention-heatmap", type=_str2bool, default=None)
    parser.add_argument("--student-use-target-visual-guidance", type=_str2bool, default=None)
    parser.add_argument("--student-use-attention-heatmap", type=_str2bool, default=None)
    parser.add_argument("--use-target-belief-tracker", type=_str2bool, default=None)
    parser.add_argument("--student-use-target-belief-tracker", type=_str2bool, default=None)
    parser.add_argument("--target-belief-token-scale", type=float, default=None)
    parser.add_argument("--target-belief-update-rate", type=float, default=None)
    parser.add_argument("--target-belief-min-confidence", type=float, default=None)
    parser.add_argument("--target-belief-temperature", type=float, default=None)
    parser.add_argument("--target-belief-loss-weight", type=float, default=None)
    parser.add_argument("--target-belief-motion-weight", type=float, default=None)
    parser.add_argument("--target-belief-update-sharpness", type=float, default=None)
    parser.add_argument("--visual-guidance-fov-deg", type=float, default=None)
    parser.add_argument("--attention-heatmap-sigma", type=float, default=None)
    parser.add_argument("--use-wan22-encoders", type=_str2bool, default=None)
    parser.add_argument("--wan22-model-base-path", type=str, default=None)
    parser.add_argument("--wan22-fastwam-src-path", type=str, default=None)
    parser.add_argument("--wan22-skip-download", type=_str2bool, default=None)
    parser.add_argument("--wan22-text-context-length", type=int, default=None)
    parser.add_argument("--wan22-text-encode-batch-size", type=int, default=None)
    parser.add_argument("--fastwam-heatmap-context-grid", type=int, default=None)
    parser.add_argument("--fastwam-skip-dit-load-from-pretrain", type=_str2bool, default=None)
    parser.add_argument("--fastwam-action-dit-pretrained-path", type=str, default=None)
    parser.add_argument("--fastwam-mot-checkpoint-mixed-attn", type=_str2bool, default=None)

    # Training config
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--max-train-steps", type=int, default=0, help="Stop after this many optimizer update steps; 0 keeps epoch-based training.")
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--wan-latent-cache-root", type=str, default="")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save-every-epochs", type=int, default=1, help="Write last.pt every N epochs; always save on final epoch.")
    parser.add_argument("--save-best-checkpoint", type=_str2bool, default=True)
    parser.add_argument("--save-optimizer-state", type=_str2bool, default=True)
    parser.add_argument("--multi-gpu", action="store_true")
    parser.add_argument("--local_rank", "--local-rank", type=int, default=-1, help="Passed by DeepSpeed/torchrun launcher.")
    parser.add_argument("--deepspeed", action="store_true", help="Use DeepSpeed ZeRO optimizer offload for the student model.")
    parser.add_argument("--deepspeed-config", type=str, default=None)
    parser.add_argument("--deepspeed-offload-optimizer", type=_str2bool, default=False)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--use-swanlab", type=_str2bool, default=False)
    parser.add_argument("--swanlab-project", type=str, default="WAM-FastWAM")
    parser.add_argument("--swanlab-experiment-name", type=str, default=None)
    parser.add_argument("--swanlab-workspace", type=str, default="")
    parser.add_argument("--swanlab-log-dir", type=str, default=None)
    parser.add_argument("--swanlab-mode", type=str, default="cloud", choices=["cloud", "local", "offline", "disabled"])

    # Distillation weights. Keep the first version clean.
    parser.add_argument("--sup-weight", type=float, default=1.0)
    parser.add_argument("--feat-distill-weight", type=float, default=0.1)
    parser.add_argument("--action-distill-weight", type=float, default=0.0)

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

    use_deepspeed = bool(args.deepspeed)
    if use_deepspeed and deepspeed is None:
        raise ImportError("DeepSpeed requested but not installed in this environment.")
    use_distributed = (use_deepspeed or args.multi_gpu) and torch.cuda.is_available() and _get_world_size() > 1
    use_ddp = (not use_deepspeed) and args.multi_gpu and torch.cuda.is_available() and _get_world_size() > 1
    if use_deepspeed:
        if _get_world_size() > 1 and not _ddp_is_initialized():
            deepspeed.init_distributed(dist_backend="nccl")
        torch.cuda.set_device(_get_local_rank())
        device = torch.device("cuda", _get_local_rank())
    elif use_ddp:
        torch.cuda.set_device(_get_local_rank())
        dist.init_process_group(backend="nccl", init_method="env://")
        device = torch.device("cuda", _get_local_rank())
    else:
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    teacher_ckpt_cfg = _load_checkpoint_cfg(args.teacher_ckpt)
    teacher_cfg = make_cfg(args, teacher_ckpt_cfg)
    student_cfg = make_student_cfg(args, teacher_cfg)
    action_video_freq_ratio = max(int(getattr(student_cfg, "fastwam_action_video_freq_ratio", 1)), 1)
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

    dataset_cfg = replace(
        student_cfg,
        use_target_visual_guidance=bool(teacher_cfg.use_target_visual_guidance or student_cfg.use_target_visual_guidance),
        use_attention_heatmap=bool(teacher_cfg.use_attention_heatmap or student_cfg.use_attention_heatmap),
        use_target_belief_tracker=bool(
            teacher_cfg.use_target_belief_tracker or student_cfg.use_target_belief_tracker
        ),
    )
    train_loader, val_loader, total_n, train_n, val_n = build_loaders(args, dataset_cfg, use_ddp=use_distributed)
    train_sampler = getattr(train_loader, "sampler_for_epoch", None)

    if _is_main_process():
        if val_n > 0:
            print(f"[dataset] total={total_n}, train={train_n}, val={val_n}")
        else:
            print(f"[dataset] total={total_n}, train={train_n}")
        records_for_cache = build_records(
            Path(args.dataset_root),
            [s.strip() for s in args.scene_list.split(",") if s.strip()],
            args.trajectory_range.strip(),
            max_vel=args.max_vel,
            max_yaw_rate=args.max_yaw_rate,
            max_speed_norm=args.max_speed_norm,
        )
        cache_stats = _wan_latent_cache_stats(records_for_cache, args.wan_latent_cache_root, args.seq_len, action_video_freq_ratio)
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
            f"[teacher cfg] guidance={teacher_cfg.use_target_visual_guidance}, heatmap={teacher_cfg.use_attention_heatmap}, "
            f"use_fastwam_mot={teacher_cfg.use_fastwam_mot}, use_wan22_encoders={teacher_cfg.use_wan22_encoders}"
        )
        print(
            f"[student cfg] guidance={student_cfg.use_target_visual_guidance}, heatmap={student_cfg.use_attention_heatmap}, "
            f"low_dim_target_input=off, fusion={student_cfg.target_token_fusion_mode}, "
            f"train_next_target_relative={student_cfg.train_next_target_relative}, rollout_head=false"
        )
        print(
            f"[distill] sup={args.sup_weight}, feat={args.feat_distill_weight}, "
            f"action={args.action_distill_weight}"
        )

    teacher = TeacherWorldModelDiT(teacher_cfg).to(device)
    load_model_state(teacher, args.teacher_ckpt, strict=False)
    freeze_model(teacher)
    if _is_main_process():
        print(f"[teacher] loaded frozen teacher: {args.teacher_ckpt}")

    student = TeacherWorldModelDiT(student_cfg).to(device)
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
    if use_deepspeed:
        ds_config = args.deepspeed_config if args.deepspeed_config is not None else _make_deepspeed_config(args)
        student, optimizer, _, _ = deepspeed.initialize(
            model=student,
            model_parameters=[p for p in student.parameters() if p.requires_grad],
            optimizer=optimizer,
            config=ds_config,
        )
        if _is_main_process():
            print(f"[train] DeepSpeed enabled on world_size={_get_world_size()} (local_rank={_get_local_rank()})")
    scaler = torch.amp.GradScaler("cuda", enabled=_grad_scaler_enabled(device, student_cfg, use_deepspeed))
    if _is_main_process() and device.type == "cuda":
        amp_dtype = _cuda_amp_dtype(student_cfg)
        print(f"[train] AMP dtype: {amp_dtype}, grad_scaler={scaler.is_enabled()}")

    start_epoch = 0
    global_step = 0
    best_val = math.inf
    history: List[Dict[str, Any]] = []

    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        _unwrap_model(student).load_state_dict(ckpt["model"], strict=True)
        if not use_deepspeed:
            if ckpt.get("optimizer") and ckpt.get("scheduler"):
                optimizer.load_state_dict(ckpt["optimizer"])
                scheduler.load_state_dict(ckpt["scheduler"])
            elif _is_main_process():
                print("[resume] optimizer/scheduler state missing; restarting optimizer state.")
        start_epoch = int(ckpt["epoch"]) + 1
        global_step = int(ckpt.get("global_step", 0))
        best_val = float(ckpt.get("best_val", best_val))
        history = list(ckpt.get("history", []))
        if _is_main_process():
            print(f"[resume] {args.resume}, start_epoch={start_epoch}, best_val={best_val:.6f}")

    run_name = args.swanlab_experiment_name or save_dir.name
    swanlab_run = _init_swanlab(args, student_cfg, run_name)
    total_pbar = None

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

    if _is_main_process():
        print(
            "[running-model] "
            f"model={save_dir.name} | run={run_name} | save_dir={save_dir} | "
            f"teacher_ckpt={args.teacher_ckpt} | "
            f"teacher_target_belief_tracker={teacher_cfg.use_target_belief_tracker} | "
            f"student_target_belief_tracker={student_cfg.use_target_belief_tracker}"
        )

    if tqdm is not None and _is_main_process():
        if int(args.max_train_steps) > 0:
            total_steps = max(int(args.max_train_steps) - int(global_step), 0)
            desc = f"self-distill steps {global_step}->{int(args.max_train_steps)}"
        else:
            total_steps = max(args.epochs - start_epoch, 0) * max(len(train_loader), 1)
            desc = f"self-distill {start_epoch:03d}->{args.epochs - 1:03d}"
        total_pbar = tqdm(
            total=total_steps,
            desc=desc,
            leave=True,
            dynamic_ncols=True,
        )

    reached_max_steps = False
    for epoch in range(start_epoch, args.epochs):
        if int(args.max_train_steps) > 0 and global_step >= int(args.max_train_steps):
            break
        _unwrap_model(student).train()
        teacher.eval()
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
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
                student.zero_grad()
            else:
                optimizer.zero_grad(set_to_none=True)

            amp_ctx = nullcontext() if use_deepspeed else _autocast_context(device, student_cfg)
            with amp_ctx:
                with torch.no_grad():
                    teacher_policy_target = batch["expert_action"] if args.action_distill_weight > 0.0 else None
                    teacher_out = teacher(
                        images=batch["images"],
                        text_tokens=batch["text_tokens"],
                        target_relative=batch["target_relative"],
                        prev_actions=batch["prev_actions"],
                        attention_mask=batch["attention_mask"],
                        attention_heatmaps=batch.get("attention_heatmaps"),
                        expert_action=teacher_policy_target,
                        valid_mask=batch["valid_mask"] if teacher_policy_target is not None else None,
                        done=batch.get("done"),
                        instructions=batch.get("instructions"),
                        video_latents=batch.get("video_latents"),
                        reference_target_relative=batch.get("reference_target_relative"),
                        reference_images=batch.get("reference_images"),
                    )

                student_out = student(
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

                losses = self_distill_losses(
                    student_out=student_out,
                    teacher_out=teacher_out,
                    student_model=_unwrap_model(student),
                    teacher_model=teacher,
                    batch=batch,
                    cfg=student_cfg,
                    args=args,
                )
                loss = losses["total"]

            if torch.is_tensor(loss) and not torch.isfinite(loss.detach()).all():
                debug = _finite_debug_summary(losses, student_out)
                raise RuntimeError(f"Non-finite self-distill loss at epoch={epoch}, step={step}: {debug}")

            if use_deepspeed:
                student.backward(loss)
                student.step()
            else:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                clip_grad_norm_(student.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()

            summary = summarize_losses(losses)
            for k, v in summary.items():
                running[k] = running.get(k, 0.0) + v

            num_train_batches += 1
            avg = {k: v / (step + 1) for k, v in running.items()}
            global_step += 1
            if total_pbar is not None:
                postfix = {
                    "epoch": f"{epoch:03d}",
                    "step": global_step,
                    "total": f"{avg.get('total', 0.0):.4f}",
                    "sup": f"{avg.get('sup_total', 0.0):.4f}",
                    "feat": f"{avg.get('feat_distill', 0.0):.4f}",
                }
                if abs(avg.get("action_distill", 0.0)) >= 1e-12:
                    postfix["action"] = f"{avg['action_distill']:.4f}"
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
            msg = f">>> Epoch {epoch:03d} train: {_format_metrics(train_avg)}"
            tqdm.write(msg) if tqdm is not None else print(msg)
            _swanlab_log(swanlab_run, {**train_avg, "lr": epoch_lr, "global_step": int(global_step)}, step=global_step, prefix="train")

        val_avg = None
        if val_loader is not None:
            val_avg = evaluate_distill(
                student=_unwrap_model(student),
                teacher=teacher,
                loader=val_loader,
                student_cfg=student_cfg,
                teacher_cfg=teacher_cfg,
                args=args,
                device=device,
            )
            if _is_main_process():
                msg = f">>> Epoch {epoch:03d} val:   {_format_metrics(val_avg)}"
                tqdm.write(msg) if tqdm is not None else print(msg)
                _swanlab_log(swanlab_run, val_avg, step=epoch, prefix="val")

        metric = train_avg["total"] if val_avg is None else val_avg["total"]
        if _is_main_process():
            history.append({
                "epoch": epoch,
                "train": train_avg,
                "val": val_avg,
                "metric": metric,
                "global_step": int(global_step),
            })

            should_save = (
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
            if should_save:
                is_best = metric < best_val
                if is_best:
                    best_val = metric
                ckpt = {
                    "epoch": epoch,
                    "global_step": int(global_step),
                    "max_train_steps": int(args.max_train_steps),
                    "model": _trainable_state_dict(student),
                    "model_state_format": "trainable_only",
                    "optimizer": {} if (use_deepspeed or not args.save_optimizer_state) else optimizer.state_dict(),
                    "scheduler": {} if (use_deepspeed or not args.save_optimizer_state) else scheduler.state_dict(),
                    "cfg": student_cfg.__dict__,
                    "teacher_cfg": teacher_cfg.__dict__,
                    "args": vars(args),
                    "best_val": best_val,
                    "history": history,
                }
                torch.save(ckpt, save_dir / "last.pt")

                if bool(args.save_best_checkpoint) and is_best:
                    torch.save(ckpt, save_dir / "best.pt")
                    print(f"[save] best.pt updated: metric={metric:.6f}")

            with open(save_dir / "history.json", "w", encoding="utf-8") as f:
                json.dump(history, f, indent=2, ensure_ascii=False)

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

    try:
        if total_pbar is not None:
            total_pbar.close()
        _swanlab_finish(swanlab_run)
    finally:
        if use_distributed and _ddp_is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
