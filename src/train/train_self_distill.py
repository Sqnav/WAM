from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
from dataclasses import fields, replace
import json
import math
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from data.teacher_dataset_builder import build_records
from model.config import ModelConfig, migrate_legacy_config
from model.losses import masked_mean, summarize_losses, world_model_dit_loss
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
                "val_scene_list": args.val_scene_list,
                "val_trajectory_range": args.val_trajectory_range,
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
    if args.use_target_relative_context is not None:
        cfg_kwargs["use_target_relative_context"] = bool(args.use_target_relative_context)
    if args.target_relative_context_scale is not None:
        cfg_kwargs["target_relative_context_scale"] = float(args.target_relative_context_scale)
    if args.target_relative_token_scale is not None:
        cfg_kwargs["target_relative_token_scale"] = float(args.target_relative_token_scale)
    if args.target_relative_context_hidden_dim is not None:
        cfg_kwargs["target_relative_context_hidden_dim"] = int(args.target_relative_context_hidden_dim)
    if args.use_tracker_center_context is not None:
        cfg_kwargs["use_tracker_center_context"] = bool(args.use_tracker_center_context)
    if args.tracker_center_context_hidden_dim is not None:
        cfg_kwargs["tracker_center_context_hidden_dim"] = int(args.tracker_center_context_hidden_dim)
    if args.tracker_center_token_scale is not None:
        cfg_kwargs["tracker_center_token_scale"] = float(args.tracker_center_token_scale)
    if args.tracker_heatmap_target_mode is not None:
        cfg_kwargs["tracker_heatmap_target_mode"] = str(args.tracker_heatmap_target_mode)
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
    if args.fastwam_skip_dit_load_from_pretrain is not None:
        cfg_kwargs["fastwam_skip_dit_load_from_pretrain"] = bool(args.fastwam_skip_dit_load_from_pretrain)
    if args.fastwam_action_dit_pretrained_path is not None:
        cfg_kwargs["fastwam_action_dit_pretrained_path"] = str(args.fastwam_action_dit_pretrained_path)
    if args.fastwam_mot_checkpoint_mixed_attn is not None:
        cfg_kwargs["fastwam_mot_checkpoint_mixed_attn"] = bool(args.fastwam_mot_checkpoint_mixed_attn)
    if args.use_fastwam_heatmap_guidance is not None:
        cfg_kwargs["use_fastwam_heatmap_guidance"] = bool(args.use_fastwam_heatmap_guidance)
    if args.fastwam_heatmap_guidance_scale is not None:
        cfg_kwargs["fastwam_heatmap_guidance_scale"] = float(args.fastwam_heatmap_guidance_scale)
    if args.fastwam_heatmap_guidance_sigma is not None:
        cfg_kwargs["fastwam_heatmap_guidance_sigma"] = float(args.fastwam_heatmap_guidance_sigma)
    if args.fastwam_heatmap_guidance_fov_deg is not None:
        cfg_kwargs["fastwam_heatmap_guidance_fov_deg"] = float(args.fastwam_heatmap_guidance_fov_deg)
    if args.fastwam_ortrack_consistency_loss_weight is not None:
        cfg_kwargs["fastwam_ortrack_consistency_loss_weight"] = float(args.fastwam_ortrack_consistency_loss_weight)
    if args.teacher_use_tracker_bias:
        cfg_kwargs.update(
            use_fastwam_attention_bias=True,
            fastwam_heatmap_source="tracker",
            use_fastwam_heatmap_guidance=True,
        )
    return ModelConfig(**cfg_kwargs)


def make_student_cfg(args: argparse.Namespace, teacher_cfg: ModelConfig) -> ModelConfig:
    updates: Dict[str, Any] = {
        "use_fastwam_attention_bias": False,
        "use_gt_center_attention_bias": False,
        "use_fastwam_heatmap_guidance": False,
    }
    if not args.student_keep_tracker_supervision:
        updates.update(
            use_fastwam_attention_heatmap_loss=False,
            use_fastwam_tracker_heatmap_loss=False,
            fastwam_heatmap_source="none",
        )
    if args.student_use_target_relative_context is not None:
        updates["use_target_relative_context"] = bool(args.student_use_target_relative_context)
    if args.student_use_tracker_center_context is not None:
        updates["use_tracker_center_context"] = bool(args.student_use_tracker_center_context)
    return replace(teacher_cfg, **updates)


def student_tracker_inputs(
    batch: Dict[str, Any],
    cfg: ModelConfig,
) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    use_center = bool(getattr(cfg, "use_tracker_center_context", False))
    use_heatmap = bool(getattr(cfg, "use_fastwam_tracker_heatmap_loss", False)) or bool(
        getattr(cfg, "use_fastwam_attention_bias", False)
    )
    return (
        batch.get("guidance_heatmap") if use_heatmap else None,
        batch.get("guidance_confidence") if (use_center or use_heatmap) else None,
        batch.get("tracker_center") if use_center else None,
    )


@torch.no_grad()
def sample_student_policy_actions(
    student: TeacherWorldModelDiT,
    batch: Dict[str, Any],
    cfg: ModelConfig,
    args: argparse.Namespace,
    sampling_offset: int = 0,
) -> torch.Tensor:
    batch_size = int(batch["target_relative"].size(0))
    horizon = max(int(cfg.action_sequence_horizon), 1)
    device = batch["target_relative"].device
    generator = torch.Generator(device=device)
    generator.manual_seed(int(args.student_sampling_seed) + int(sampling_offset))
    initial_noise = torch.randn(
        batch_size,
        horizon,
        int(cfg.action_dim),
        device=device,
        dtype=torch.float32,
        generator=generator,
    )
    targets = student.sample_distillation_targets(
        images=batch["images"],
        text_tokens=batch["text_tokens"],
        target_relative=batch["target_relative"],
        attention_mask=batch["attention_mask"],
        instructions=batch.get("instructions"),
        video_latents=batch.get("video_latents"),
        guidance_heatmap=None,
        guidance_confidence=None,
        tracker_center=None,
        num_steps=int(args.student_sampling_steps),
        initial_action_noise=initial_noise,
        return_attention_maps=False,
    )
    return targets["teacher_action_sequence"].detach()


def shared_flow_state(
    model: TeacherWorldModelDiT,
    sampled_action: torch.Tensor,
    batch: Dict[str, Any],
) -> Dict[str, Optional[torch.Tensor]]:
    fastwam = model.fastwam
    if fastwam is None:
        raise RuntimeError("FastWAM is required for velocity distillation.")
    action_noise = torch.randn_like(sampled_action)
    action_timestep = fastwam.action_scheduler.sample_training_t(
        sampled_action.size(0), sampled_action.device, sampled_action.dtype
    )

    video_latents = batch.get("video_latents")
    if video_latents is None:
        raise RuntimeError(
            "Shared video-velocity distillation requires cached video_latents."
        )
    video_noise = torch.randn_like(video_latents)
    video_timestep = fastwam.video_scheduler.sample_training_t(
        video_latents.size(0), video_latents.device, video_latents.dtype
    )
    return {
        "noise_action": action_noise,
        "t_action": action_timestep,
        "noise_video": video_noise,
        "t_video": video_timestep,
    }


@contextmanager
def deterministic_student_kd_forward(model: torch.nn.Module):
    dropout_types = (
        torch.nn.Dropout,
        torch.nn.Dropout1d,
        torch.nn.Dropout2d,
        torch.nn.Dropout3d,
        torch.nn.AlphaDropout,
        torch.nn.FeatureAlphaDropout,
    )
    modules = [module for module in _unwrap_model(model).modules() if isinstance(module, dropout_types)]
    states = [module.training for module in modules]
    try:
        for module in modules:
            module.train(False)
        yield
    finally:
        for module, training in zip(modules, states):
            module.train(training)


def forward_student_policy_target(
    student: torch.nn.Module,
    batch: Dict[str, Any],
    cfg: ModelConfig,
    action_target: torch.Tensor,
    capture_attention: bool,
    capture_flow_predictions: bool = False,
    flow_state: Optional[Dict[str, Optional[torch.Tensor]]] = None,
) -> Dict[str, torch.Tensor]:
    student_heatmap, student_confidence, student_center = student_tracker_inputs(batch, cfg)
    return student(
        images=batch["images"],
        text_tokens=batch["text_tokens"],
        target_relative=batch["target_relative"],
        prev_actions=batch["prev_actions"],
        attention_mask=batch["attention_mask"],
        expert_action=action_target,
        valid_mask=batch["valid_mask"],
        done=batch.get("done"),
        instructions=batch.get("instructions"),
        video_latents=batch.get("video_latents"),
        guidance_heatmap=student_heatmap,
        guidance_confidence=student_confidence,
        tracker_center=student_center,
        capture_fastwam_attention=capture_attention,
        capture_fastwam_flow_predictions=capture_flow_predictions,
        fastwam_noise_video_override=None if flow_state is None else flow_state.get("noise_video"),
        fastwam_t_video_override=None if flow_state is None else flow_state.get("t_video"),
        fastwam_noise_action_override=None if flow_state is None else flow_state.get("noise_action"),
        fastwam_t_action_override=None if flow_state is None else flow_state.get("t_action"),
    )


def forward_teacher_velocity(
    teacher: torch.nn.Module,
    batch: Dict[str, Any],
    action_target: torch.Tensor,
    flow_state: Dict[str, Optional[torch.Tensor]],
    capture_attention: bool = False,
) -> Dict[str, torch.Tensor]:
    return teacher(
        images=batch["images"],
        text_tokens=batch["text_tokens"],
        target_relative=batch["target_relative"],
        prev_actions=batch["prev_actions"],
        attention_mask=batch["attention_mask"],
        expert_action=action_target,
        valid_mask=batch["valid_mask"],
        done=batch.get("done"),
        instructions=batch.get("instructions"),
        video_latents=batch.get("video_latents"),
        guidance_heatmap=batch.get("guidance_heatmap"),
        guidance_confidence=batch.get("guidance_confidence"),
        tracker_center=batch.get("tracker_center"),
        capture_fastwam_attention=capture_attention,
        capture_fastwam_flow_predictions=True,
        fastwam_noise_video_override=flow_state.get("noise_video"),
        fastwam_t_video_override=flow_state.get("t_video"),
        fastwam_noise_action_override=flow_state.get("noise_action"),
        fastwam_t_action_override=flow_state.get("t_action"),
    )


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


def sampled_action_as_sequence_target(
    sampled_action: torch.Tensor,
    expert_action: torch.Tensor,
) -> torch.Tensor:
    """Append one shape-only padding step to a sampled FastWAM action trajectory."""
    if sampled_action.ndim != 3 or expert_action.ndim != 3:
        raise ValueError("Sampled and expert actions must have shape [B,T,A].")
    if sampled_action.size(0) != expert_action.size(0) or sampled_action.size(2) != expert_action.size(2):
        raise ValueError(
            "Sampled action batch/dim must match expert_action; "
            f"got {tuple(sampled_action.shape)} and {tuple(expert_action.shape)}."
        )
    transitions = max(int(expert_action.size(1)) - 1, 1)
    if sampled_action.size(1) < transitions:
        pad = sampled_action[:, -1:].expand(-1, transitions - sampled_action.size(1), -1)
        sampled_action = torch.cat([sampled_action, pad], dim=1)
    sampled_action = sampled_action[:, :transitions].detach()
    # FastWAM creates action queries from action_target[:, :-1]. Keep the final
    # slot as an explicit non-label padding sentinel instead of duplicating the
    # last real action and suggesting a false equality constraint.
    padding = torch.zeros_like(sampled_action[:, -1:])
    return torch.cat([sampled_action, padding], dim=1)


def action_query_valid_mask(
    valid_mask: Optional[torch.Tensor],
    query_count: int,
    *,
    device: torch.device,
) -> Optional[torch.Tensor]:
    if valid_mask is None:
        return None
    valid = valid_mask.to(device=device, dtype=torch.bool)
    if valid.ndim == 3 and valid.size(-1) == 1:
        valid = valid.squeeze(-1)
    if valid.ndim != 2:
        raise ValueError("valid_mask must have shape [B,T] or [B,action_queries].")
    if valid.size(1) == query_count + 1:
        valid = valid[:, :-1] & valid[:, 1:]
    elif valid.size(1) != query_count:
        raise ValueError(
            f"valid_mask length {valid.size(1)} does not match {query_count} action queries."
        )
    return valid


def action_velocity_distillation_loss(
    student_velocity_out: Optional[Dict[str, torch.Tensor]],
    teacher_velocity_out: Optional[Dict[str, torch.Tensor]],
    valid_mask: Optional[torch.Tensor],
    reference: torch.Tensor,
) -> torch.Tensor:
    if student_velocity_out is None or teacher_velocity_out is None:
        return reference.sum() * 0.0
    student_velocity = student_velocity_out["policy_action_sequence"].float()
    teacher_velocity = teacher_velocity_out["policy_action_sequence"].detach().float()
    if student_velocity.shape != teacher_velocity.shape:
        raise ValueError(
            "Teacher/Student action velocity shapes differ: "
            f"{tuple(teacher_velocity.shape)} vs {tuple(student_velocity.shape)}."
        )
    per_token = (student_velocity - teacher_velocity).pow(2).mean(dim=(-1, -2))
    valid = action_query_valid_mask(
        valid_mask, int(per_token.size(1)), device=per_token.device
    )
    if valid is not None:
        weight = valid.to(dtype=per_token.dtype)
        return (per_token * weight).sum() / weight.sum().clamp_min(1.0)
    return per_token.mean()


def video_velocity_distillation_loss(
    student_velocity_out: Optional[Dict[str, torch.Tensor]],
    teacher_velocity_out: Optional[Dict[str, torch.Tensor]],
    reference: torch.Tensor,
) -> torch.Tensor:
    if student_velocity_out is None or teacher_velocity_out is None:
        return reference.sum() * 0.0
    student_velocity = student_velocity_out["video_velocity"].float()
    teacher_velocity = teacher_velocity_out["video_velocity"].detach().float()
    if student_velocity.shape != teacher_velocity.shape:
        raise ValueError(
            "Teacher/Student video velocity shapes differ: "
            f"{tuple(teacher_velocity.shape)} vs {tuple(student_velocity.shape)}."
        )
    return F.mse_loss(student_velocity, teacher_velocity)


def all_query_attention_distillation_loss(
    student_velocity_out: Optional[Dict[str, torch.Tensor]],
    teacher_velocity_out: Optional[Dict[str, torch.Tensor]],
    valid_mask: Optional[torch.Tensor],
    reference: torch.Tensor,
) -> torch.Tensor:
    """Distill per-query spatial attention over the first 7x7 visual tokens."""
    if student_velocity_out is None or teacher_velocity_out is None:
        return reference.sum() * 0.0
    student_attention = student_velocity_out.get("last_guided_action_attention")
    if student_attention is None:
        student_attention = student_velocity_out.get("last_action_attention")
    # A privileged center-bias teacher must distill the post-bias attention,
    # otherwise its GT spatial prior is absent from the KD target.
    teacher_attention = teacher_velocity_out.get("last_guided_action_attention")
    if teacher_attention is None:
        teacher_attention = teacher_velocity_out.get("last_action_attention")
    if student_attention is None or teacher_attention is None:
        raise RuntimeError("Attention distillation requested without captured last-layer attention.")
    if student_attention.ndim != 4 or teacher_attention.ndim != 4:
        raise ValueError("Last-layer attention must have shape [B, heads, queries, video_tokens].")
    if student_attention.shape[:3] != teacher_attention.shape[:3]:
        raise ValueError(
            "Teacher/Student attention batch/head/query shapes differ: "
            f"{tuple(teacher_attention.shape)} vs {tuple(student_attention.shape)}."
        )

    image_tokens = 7 * 7
    if student_attention.size(3) < image_tokens or teacher_attention.size(3) < image_tokens:
        raise ValueError(
            "Attention distillation requires at least 49 first-frame visual tokens in both models; "
            f"got teacher={tuple(teacher_attention.shape)}, student={tuple(student_attention.shape)}."
        )
    student = student_attention[..., :image_tokens].float().mean(dim=1).clamp_min(1.0e-8)
    teacher = teacher_attention[..., :image_tokens].detach().float().mean(dim=1).clamp_min(1.0e-8)
    student = student / student.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)
    teacher = teacher / teacher.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)
    per_query = (teacher * (teacher.log() - student.log())).sum(dim=-1)

    valid = action_query_valid_mask(
        valid_mask, int(per_query.size(1)), device=per_query.device
    )
    if valid is not None:
        if not valid.any():
            return student_attention.sum() * 0.0
        return per_query[valid].mean()
    return per_query.mean()


def self_distill_losses(
    student_out: Dict[str, torch.Tensor],
    teacher_out: Optional[Dict[str, torch.Tensor]],
    offline_student_velocity_out: Optional[Dict[str, torch.Tensor]],
    offline_teacher_velocity_out: Optional[Dict[str, torch.Tensor]],
    sampled_student_velocity_out: Optional[Dict[str, torch.Tensor]],
    sampled_teacher_velocity_out: Optional[Dict[str, torch.Tensor]],
    student_model: TeacherWorldModelDiT,
    teacher_model: TeacherWorldModelDiT,
    batch: Dict[str, Any],
    cfg: ModelConfig,
    args: argparse.Namespace,
) -> Dict[str, torch.Tensor]:
    valid_mask = batch.get("valid_mask")
    sup = world_model_dit_loss(student_out, batch, cfg, valid_mask=valid_mask)

    losses: Dict[str, torch.Tensor] = {f"sup_{k}": v for k, v in sup.items()}

    del student_model, teacher_model
    student_belief = belief_feat(student_out)
    feat_loss = student_belief.sum() * 0.0
    if args.feat_distill_weight > 0.0:
        if teacher_out is None:
            raise RuntimeError("Feature distillation requested without a teacher training forward.")
        with torch.no_grad():
            teacher_belief = belief_feat(teacher_out)
        feat_loss = masked_mse_lastdim(student_belief, teacher_belief, valid_mask)
    offline_action_loss = action_velocity_distillation_loss(
        offline_student_velocity_out,
        offline_teacher_velocity_out,
        valid_mask,
        student_belief,
    )
    offline_video_velocity_loss = video_velocity_distillation_loss(
        offline_student_velocity_out,
        offline_teacher_velocity_out,
        student_belief,
    )
    sampled_action_loss = action_velocity_distillation_loss(
        sampled_student_velocity_out,
        sampled_teacher_velocity_out,
        valid_mask,
        student_belief,
    )
    sampled_video_velocity_loss = student_belief.sum() * 0.0
    attention_loss = all_query_attention_distillation_loss(
        offline_student_velocity_out,
        offline_teacher_velocity_out,
        valid_mask,
        student_belief,
    ) if args.attention_distill_weight > 0.0 else student_belief.sum() * 0.0
    flow_distill_loss = offline_action_loss + offline_video_velocity_loss + sampled_action_loss
    total = (
        args.sup_weight * sup["total"]
        + args.feat_distill_weight * feat_loss
        + args.action_distill_weight * flow_distill_loss
        + args.attention_distill_weight * attention_loss
    )

    losses["feat_distill"] = feat_loss
    losses["offline_action_velocity_distill"] = offline_action_loss
    losses["offline_video_velocity_distill"] = offline_video_velocity_loss
    losses["sampled_action_velocity_distill"] = sampled_action_loss
    losses["sampled_video_velocity_distill"] = sampled_video_velocity_loss
    losses["action_distill"] = offline_action_loss + sampled_action_loss
    losses["video_velocity_distill"] = offline_video_velocity_loss
    losses["flow_distill"] = flow_distill_loss
    losses["attention_distill"] = attention_loss
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

    for batch_index, batch in enumerate(val_iter):
        batch = move_batch_to_device(batch, device)
        teacher_out = None
        if args.feat_distill_weight > 0.0:
            teacher_out = teacher(
                images=batch["images"],
                text_tokens=batch["text_tokens"],
                target_relative=batch["target_relative"],
                prev_actions=batch["prev_actions"],
                attention_mask=batch["attention_mask"],
                expert_action=None,
                valid_mask=None,
                done=batch.get("done"),
                instructions=batch.get("instructions"),
                video_latents=batch.get("video_latents"),
                guidance_confidence=batch.get("guidance_confidence"),
                tracker_center=batch.get("tracker_center"),
            )

        student_out = forward_student_policy_target(
            student,
            batch,
            student_cfg,
            batch["expert_action"],
            capture_attention=False,
        )
        offline_student_velocity_out = None
        offline_teacher_velocity_out = None
        sampled_student_velocity_out = None
        sampled_teacher_velocity_out = None
        if args.action_distill_weight > 0.0 or args.attention_distill_weight > 0.0:
            expert_action = batch["expert_action"]
            offline_flow_state = shared_flow_state(student, expert_action[:, :-1], batch)
            offline_teacher_velocity_out = forward_teacher_velocity(
                teacher, batch, expert_action, offline_flow_state,
                capture_attention=args.attention_distill_weight > 0.0,
            )
            with deterministic_student_kd_forward(student):
                offline_student_velocity_out = forward_student_policy_target(
                    student,
                    batch,
                    student_cfg,
                    expert_action,
                    capture_attention=args.attention_distill_weight > 0.0,
                    capture_flow_predictions=True,
                    flow_state=offline_flow_state,
                )
        if args.action_distill_weight > 0.0:
            sampled_action = sample_student_policy_actions(
                student, batch, student_cfg, args, sampling_offset=batch_index
            )
            sampled_action_target = sampled_action_as_sequence_target(
                sampled_action,
                batch["expert_action"],
            )
            flow_state = shared_flow_state(student, sampled_action, batch)
            sampled_teacher_velocity_out = forward_teacher_velocity(
                teacher, batch, sampled_action_target, flow_state,
                capture_attention=False,
            )
            with deterministic_student_kd_forward(student):
                sampled_student_velocity_out = forward_student_policy_target(
                    student,
                    batch,
                    student_cfg,
                    sampled_action_target,
                    capture_attention=False,
                    capture_flow_predictions=True,
                    flow_state=flow_state,
                )

        losses = self_distill_losses(
            student_out=student_out,
            teacher_out=teacher_out,
            offline_student_velocity_out=offline_student_velocity_out,
            offline_teacher_velocity_out=offline_teacher_velocity_out,
            sampled_student_velocity_out=sampled_student_velocity_out,
            sampled_teacher_velocity_out=sampled_teacher_velocity_out,
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
    teacher_requires_gt_center: bool = False,
) -> tuple[DataLoader, Optional[DataLoader], int, int, int]:
    scene_list = [s.strip() for s in args.scene_list.split(",") if s.strip()]
    if not scene_list:
        raise ValueError("--scene-list is empty.")

    train_records = build_records(
        Path(args.dataset_root),
        scene_list,
        args.trajectory_range.strip(),
        max_vel=cfg.max_vel,
        max_yaw_rate=cfg.max_yaw_rate,
        max_speed_norm=cfg.max_speed_norm,
    )
    if not train_records:
        raise RuntimeError("No trajectory selected. Check --scene-list / --trajectory-range.")
    explicit_val = bool(args.val_scene_list.strip() or args.val_trajectory_range.strip())
    if explicit_val and not (args.val_scene_list.strip() and args.val_trajectory_range.strip()):
        raise ValueError("--val-scene-list and --val-trajectory-range must be provided together.")
    if explicit_val:
        val_scenes = [s.strip() for s in args.val_scene_list.split(",") if s.strip()]
        val_records = build_records(
            Path(args.dataset_root),
            val_scenes,
            args.val_trajectory_range.strip(),
            max_vel=cfg.max_vel,
            max_yaw_rate=cfg.max_yaw_rate,
            max_speed_norm=cfg.max_speed_norm,
        )
        if not val_records:
            raise RuntimeError(
                "No validation trajectory selected. Check --val-scene-list / --val-trajectory-range."
            )
        train_keys = {(record["scene_id"], record["trajectory_name"]) for record in train_records}
        val_keys = {(record["scene_id"], record["trajectory_name"]) for record in val_records}
        overlap = sorted(train_keys.intersection(val_keys))
        if overlap:
            raise ValueError(f"Training and validation trajectories overlap: {overlap[:8]}")
    else:
        rng = random.Random(args.split_seed)
        rng.shuffle(train_records)
        val_n = int(len(train_records) * args.val_ratio)
        if args.val_ratio > 0.0 and len(train_records) > 1:
            val_n = max(1, val_n)
        val_n = min(val_n, max(len(train_records) - 1, 0))
        val_records = train_records[:val_n]
        train_records = train_records[val_n:] if val_n > 0 else train_records
    random.Random(args.split_seed).shuffle(train_records)

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
        wan_latent_cache_root=args.wan_latent_cache_root if args.wan_latent_cache_root else None,
        action_video_freq_ratio=cfg.fastwam_action_video_freq_ratio,
        ortrack_cache_root=args.ortrack_cache_root,
        require_ortrack_cache=(
            args.teacher_use_tracker_bias
            or bool(args.use_tracker_center_context)
            or cfg.use_tracker_center_context
            or cfg.use_fastwam_tracker_heatmap_loss
        ),
        canonical_heatmap_sigma=cfg.fastwam_attention_heatmap_sigma,
        tracker_heatmap_target_mode=cfg.tracker_heatmap_target_mode,
        guidance_heatmap_source=("gt" if teacher_requires_gt_center else None),
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
            wan_latent_cache_root=args.wan_latent_cache_root if args.wan_latent_cache_root else None,
            action_video_freq_ratio=cfg.fastwam_action_video_freq_ratio,
            ortrack_cache_root=args.ortrack_cache_root,
            require_ortrack_cache=(
                args.teacher_use_tracker_bias
                or bool(args.use_tracker_center_context)
                or cfg.use_tracker_center_context
                or cfg.use_fastwam_tracker_heatmap_loss
            ),
            canonical_heatmap_sigma=cfg.fastwam_attention_heatmap_sigma,
            tracker_heatmap_target_mode=cfg.tracker_heatmap_target_mode,
            guidance_heatmap_source=("gt" if teacher_requires_gt_center else None),
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

    return train_loader, val_loader, len(train_records) + len(val_records), len(train_records), len(val_records)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Self-distillation for FastWAM teacher/student models.")

    # Dataset / paths
    parser.add_argument("--dataset-root", type=str, required=True)
    parser.add_argument("--scene-list", type=str, required=True)
    parser.add_argument("--trajectory-range", type=str, default="")
    parser.add_argument("--val-scene-list", type=str, default="")
    parser.add_argument("--val-trajectory-range", type=str, default="")
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
    parser.add_argument("--use-target-relative-context", type=_str2bool, default=None)
    parser.add_argument("--student-use-target-relative-context", type=_str2bool, default=None)
    parser.add_argument("--target-relative-context-scale", type=float, default=None)
    parser.add_argument("--target-relative-token-scale", type=float, default=None)
    parser.add_argument("--target-relative-context-hidden-dim", type=int, default=None)
    parser.add_argument("--use-tracker-center-context", type=_str2bool, default=None)
    parser.add_argument("--student-use-tracker-center-context", type=_str2bool, default=None)
    parser.add_argument("--tracker-center-context-hidden-dim", type=int, default=None)
    parser.add_argument("--tracker-center-token-scale", type=float, default=None)
    parser.add_argument(
        "--tracker-heatmap-target-mode",
        choices=["canonical", "raw", "raw_area"],
        default=None,
    )
    parser.add_argument("--use-wan22-encoders", type=_str2bool, default=None)
    parser.add_argument("--wan22-model-base-path", type=str, default=None)
    parser.add_argument("--wan22-fastwam-src-path", type=str, default=None)
    parser.add_argument("--wan22-skip-download", type=_str2bool, default=None)
    parser.add_argument("--wan22-text-context-length", type=int, default=None)
    parser.add_argument("--wan22-text-encode-batch-size", type=int, default=None)
    parser.add_argument("--fastwam-skip-dit-load-from-pretrain", type=_str2bool, default=None)
    parser.add_argument("--fastwam-action-dit-pretrained-path", type=str, default=None)
    parser.add_argument("--fastwam-mot-checkpoint-mixed-attn", type=_str2bool, default=None)
    parser.add_argument("--use-fastwam-heatmap-guidance", type=_str2bool, default=None)
    parser.add_argument("--fastwam-heatmap-guidance-scale", type=float, default=None)
    parser.add_argument("--fastwam-heatmap-guidance-sigma", type=float, default=None)
    parser.add_argument("--fastwam-heatmap-guidance-fov-deg", type=float, default=None)
    parser.add_argument("--fastwam-ortrack-consistency-loss-weight", type=float, default=None)
    parser.add_argument("--teacher-use-tracker-bias", type=_str2bool, default=False)
    parser.add_argument("--student-keep-tracker-supervision", type=_str2bool, default=False)
    parser.add_argument(
        "--distillation-target-mode",
        choices=["expert_offline_plus_student_sampled_action_velocity"],
        default="expert_offline_plus_student_sampled_action_velocity",
    )
    parser.add_argument(
        "--student-sampling-steps",
        type=int,
        default=8,
        help="FastWAM denoising steps used to generate final teacher action targets.",
    )
    parser.add_argument(
        "--student-sampling-seed",
        type=int,
        default=12345,
        help="Fixed initial action-noise seed for deterministic teacher targets.",
    )

    # Training config
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--max-train-steps", type=int, default=0, help="Stop after this many optimizer update steps; 0 keeps epoch-based training.")
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--wan-latent-cache-root", type=str, default="")
    parser.add_argument("--ortrack-cache-root", type=str, default="")
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

    # Tracker-free policy distillation weights.
    parser.add_argument("--sup-weight", type=float, default=1.0)
    parser.add_argument("--feat-distill-weight", type=float, default=0.0)
    parser.add_argument("--action-distill-weight", type=float, default=0.2)
    parser.add_argument("--attention-distill-weight", type=float, default=0.0)
    parser.add_argument(
        "--attention-distill-mode",
        choices=["all_queries_spatial_kl"],
        default="all_queries_spatial_kl",
    )

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
    if int(args.student_sampling_steps) <= 0:
        raise ValueError("--student-sampling-steps must be positive.")

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

    train_loader, val_loader, total_n, train_n, val_n = build_loaders(
        args,
        student_cfg,
        use_ddp=use_distributed,
        teacher_requires_gt_center=bool(teacher_cfg.use_gt_center_attention_bias),
    )
    train_sampler = getattr(train_loader, "sampler_for_epoch", None)

    if _is_main_process():
        if val_n > 0:
            print(
                f"[dataset] total={total_n}, train={train_n} "
                f"({args.scene_list} {args.trajectory_range}), val={val_n} "
                f"({args.val_scene_list or 'ratio split'} {args.val_trajectory_range or args.val_ratio})"
            )
        else:
            print(
                f"[dataset] total={total_n}, train={train_n} "
                f"({args.scene_list} {args.trajectory_range})"
            )
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
            f"[teacher cfg] target_relative_context={teacher_cfg.use_target_relative_context}, "
            f"tracker_center_context={teacher_cfg.use_tracker_center_context}, "
            f"tracker_heatmap_target={teacher_cfg.tracker_heatmap_target_mode}, "
            f"tracker_loss={teacher_cfg.use_fastwam_tracker_heatmap_loss}, "
            f"use_fastwam_mot={teacher_cfg.use_fastwam_mot}, use_wan22_encoders={teacher_cfg.use_wan22_encoders}"
        )
        print(
            f"[student cfg] target_relative_context={student_cfg.use_target_relative_context}, "
            f"tracker_center_context={student_cfg.use_tracker_center_context}, "
            f"tracker_heatmap_target={student_cfg.tracker_heatmap_target_mode}, "
            f"tracker_loss={student_cfg.use_fastwam_tracker_heatmap_loss}, "
            f"low_dim_target_input=off, fusion={student_cfg.target_token_fusion_mode}, "
            f"train_next_target_relative={student_cfg.train_next_target_relative}, rollout_head=false"
        )
        print(
            f"[distill] sup={args.sup_weight}, feat={args.feat_distill_weight}, "
            f"offline_action_video_and_sampled_action_velocity={args.action_distill_weight}, "
            f"attention={args.attention_distill_weight}, "
            f"student_sampling_steps={args.student_sampling_steps}, "
            f"student_sampling_seed={args.student_sampling_seed}"
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
    best_epoch = -1
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
        best_epoch = int(ckpt.get("best_epoch", best_epoch))
        history = list(ckpt.get("history", []))
        if _is_main_process():
            print(f"[resume] {args.resume}, start_epoch={start_epoch}, best_val={best_val:.6f}")

    run_name = args.swanlab_experiment_name or save_dir.name
    swanlab_run = _init_swanlab(args, student_cfg, run_name)
    total_pbar = None

    if use_deepspeed:
        pass
    elif use_ddp:
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
            f"teacher_target_relative_context={teacher_cfg.use_target_relative_context} | "
            f"student_target_relative_context={student_cfg.use_target_relative_context}"
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

            losses: Dict[str, torch.Tensor] = {}

            def backward_branch(branch_loss: torch.Tensor, branch_name: str) -> None:
                if not torch.isfinite(branch_loss.detach()).all():
                    raise RuntimeError(
                        f"Non-finite {branch_name} loss at epoch={epoch}, step={step}: "
                        f"{float(branch_loss.detach().float().cpu())}"
                    )
                if use_deepspeed:
                    student.backward(branch_loss)
                else:
                    scaler.scale(branch_loss).backward()

            # Branch 1: ordinary expert-action / expert-video supervision.
            with (nullcontext() if use_deepspeed else _autocast_context(device, student_cfg)):
                teacher_out = None
                if args.feat_distill_weight > 0.0:
                    with torch.no_grad():
                        teacher_out = teacher(
                            images=batch["images"],
                            text_tokens=batch["text_tokens"],
                            target_relative=batch["target_relative"],
                            prev_actions=batch["prev_actions"],
                            attention_mask=batch["attention_mask"],
                            expert_action=None,
                            valid_mask=None,
                            done=batch.get("done"),
                            instructions=batch.get("instructions"),
                            video_latents=batch.get("video_latents"),
                            guidance_confidence=batch.get("guidance_confidence"),
                            tracker_center=batch.get("tracker_center"),
                        )

                student_out = forward_student_policy_target(
                    student,
                    batch,
                    student_cfg,
                    batch["expert_action"],
                    capture_attention=False,
                )
                sup = world_model_dit_loss(
                    student_out, batch, student_cfg, valid_mask=batch.get("valid_mask")
                )
                student_belief = belief_feat(student_out)
                feat_loss = student_belief.sum() * 0.0
                if args.feat_distill_weight > 0.0:
                    if teacher_out is None:
                        raise RuntimeError("Feature distillation requested without a teacher forward.")
                    feat_loss = masked_mse_lastdim(
                        student_belief, belief_feat(teacher_out).detach(), batch.get("valid_mask")
                    )
                supervised_branch_loss = (
                    args.sup_weight * sup["total"]
                    + args.feat_distill_weight * feat_loss
                )
            backward_branch(supervised_branch_loss, "supervised")
            losses.update({f"sup_{key}": value.detach() for key, value in sup.items()})
            losses["feat_distill"] = feat_loss.detach()
            del student_out, teacher_out, student_belief, sup, feat_loss

            zero = supervised_branch_loss.detach() * 0.0
            offline_action_loss = zero
            offline_video_velocity_loss = zero
            attention_loss = zero

            # Branch 2: expert-action / expert-video offline velocity and attention KD.
            if args.action_distill_weight > 0.0 or args.attention_distill_weight > 0.0:
                with (nullcontext() if use_deepspeed else _autocast_context(device, student_cfg)):
                    expert_action = batch["expert_action"]
                    offline_flow_state = shared_flow_state(
                        _unwrap_model(student), expert_action[:, :-1], batch
                    )
                    with torch.no_grad():
                        offline_teacher_velocity_out = forward_teacher_velocity(
                            teacher, batch, expert_action, offline_flow_state,
                            capture_attention=args.attention_distill_weight > 0.0,
                        )
                    with deterministic_student_kd_forward(student):
                        offline_student_velocity_out = forward_student_policy_target(
                            student,
                            batch,
                            student_cfg,
                            expert_action,
                            capture_attention=args.attention_distill_weight > 0.0,
                            capture_flow_predictions=True,
                            flow_state=offline_flow_state,
                        )
                    offline_action_loss = action_velocity_distillation_loss(
                        offline_student_velocity_out,
                        offline_teacher_velocity_out,
                        batch.get("valid_mask"),
                        supervised_branch_loss,
                    )
                    offline_video_velocity_loss = video_velocity_distillation_loss(
                        offline_student_velocity_out,
                        offline_teacher_velocity_out,
                        supervised_branch_loss,
                    )
                    attention_loss = (
                        all_query_attention_distillation_loss(
                            offline_student_velocity_out,
                            offline_teacher_velocity_out,
                            batch.get("valid_mask"),
                            supervised_branch_loss,
                        )
                        if args.attention_distill_weight > 0.0
                        else zero
                    )
                    offline_branch_loss = (
                        args.action_distill_weight
                        * (offline_action_loss + offline_video_velocity_loss)
                        + args.attention_distill_weight * attention_loss
                    )
                backward_branch(offline_branch_loss, "offline-distill")
                del offline_student_velocity_out, offline_teacher_velocity_out, offline_flow_state

            sampled_action_loss = zero
            sampled_video_velocity_loss = zero

            # Branch 3: Student-sampled actions receive action-velocity KD only.
            if args.action_distill_weight > 0.0:
                with (nullcontext() if use_deepspeed else _autocast_context(device, student_cfg)):
                    with torch.no_grad(), deterministic_student_kd_forward(student):
                        sampled_action = sample_student_policy_actions(
                            _unwrap_model(student),
                            batch,
                            student_cfg,
                            args,
                            sampling_offset=global_step,
                        )
                    sampled_action_target = sampled_action_as_sequence_target(
                        sampled_action,
                        batch["expert_action"],
                    )
                    flow_state = shared_flow_state(
                        _unwrap_model(student), sampled_action, batch
                    )
                    with torch.no_grad():
                        sampled_teacher_velocity_out = forward_teacher_velocity(
                            teacher, batch, sampled_action_target, flow_state,
                            capture_attention=False,
                        )
                    with deterministic_student_kd_forward(student):
                        sampled_student_velocity_out = forward_student_policy_target(
                            student,
                            batch,
                            student_cfg,
                            sampled_action_target,
                            capture_attention=False,
                            capture_flow_predictions=True,
                            flow_state=flow_state,
                        )
                    sampled_action_loss = action_velocity_distillation_loss(
                        sampled_student_velocity_out,
                        sampled_teacher_velocity_out,
                        batch.get("valid_mask"),
                        supervised_branch_loss,
                    )
                    sampled_branch_loss = args.action_distill_weight * sampled_action_loss
                backward_branch(sampled_branch_loss, "sampled-action-distill")
                del (
                    sampled_student_velocity_out,
                    sampled_teacher_velocity_out,
                    sampled_action,
                    sampled_action_target,
                    flow_state,
                )

            flow_distill_loss = (
                offline_action_loss.detach()
                + offline_video_velocity_loss.detach()
                + sampled_action_loss.detach()
            )
            losses["offline_action_velocity_distill"] = offline_action_loss.detach()
            losses["offline_video_velocity_distill"] = offline_video_velocity_loss.detach()
            losses["sampled_action_velocity_distill"] = sampled_action_loss.detach()
            losses["sampled_video_velocity_distill"] = sampled_video_velocity_loss
            losses["action_distill"] = (
                offline_action_loss.detach() + sampled_action_loss.detach()
            )
            losses["video_velocity_distill"] = offline_video_velocity_loss.detach()
            losses["flow_distill"] = flow_distill_loss
            losses["attention_distill"] = attention_loss.detach()
            losses["total"] = (
                supervised_branch_loss.detach()
                + args.action_distill_weight * flow_distill_loss
                + args.attention_distill_weight * attention_loss.detach()
            )

            if use_deepspeed:
                student.step()
            else:
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
                if abs(avg.get("attention_distill", 0.0)) >= 1e-12:
                    postfix["attention"] = f"{avg['attention_distill']:.4f}"
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

        metric_source = train_avg if val_avg is None else val_avg
        metric = metric_source["total"]
        if _is_main_process():
            history.append({
                "epoch": epoch,
                "train": train_avg,
                "val": val_avg,
                "metric": metric,
                "metric_name": "total",
                "global_step": int(global_step),
            })

            should_save_last = (
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
            is_best = metric < best_val
            if is_best:
                best_val = metric
                best_epoch = epoch
            if should_save_last or (bool(args.save_best_checkpoint) and is_best):
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
                    "best_epoch": int(best_epoch),
                    "history": history,
                }
            if should_save_last:
                torch.save(ckpt, save_dir / "last.pt")

            if bool(args.save_best_checkpoint) and is_best:
                torch.save(ckpt, save_dir / "best.pt")
                print(f"[save] best.pt updated: epoch={epoch}, metric={metric:.6f}")

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
            "best_epoch": int(best_epoch),
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
