from __future__ import annotations

from typing import Dict, Optional

import torch

from .action_loss_utils import weighted_mean_action_squared_error
from .config import ModelConfig


def masked_mean(x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    if mask is None:
        return x.mean()
    mask = mask.float()
    while mask.ndim < x.ndim:
        mask = mask.unsqueeze(-1)
    return (x * mask).sum() / mask.sum().clamp(min=1.0)


def kl_normal(mean_q: torch.Tensor, std_q: torch.Tensor, mean_p: torch.Tensor, std_p: torch.Tensor) -> torch.Tensor:
    var_q = std_q.pow(2)
    var_p = std_p.pow(2)
    log_std_ratio = torch.log(std_p) - torch.log(std_q)
    kl = log_std_ratio + (var_q + (mean_q - mean_p).pow(2)) / (2 * var_p) - 0.5
    return kl.sum(dim=-1)


def action_sequence_loss(
    pred_sequence: torch.Tensor,
    expert_action: torch.Tensor,
    valid_mask: Optional[torch.Tensor],
    cfg: ModelConfig,
) -> torch.Tensor:
    if pred_sequence.ndim != 4:
        raise ValueError("pred_sequence must have shape [B, T, H, A].")
    if expert_action.ndim != 3:
        raise ValueError("expert_action must have shape [B, T, A].")
    if pred_sequence.size(0) != expert_action.size(0):
        raise ValueError("pred_sequence and expert_action batch sizes must match.")
    if pred_sequence.size(1) > expert_action.size(1):
        raise ValueError("pred_sequence cannot be longer than expert_action.")
    sequence_length = pred_sequence.size(1)
    expert_action = expert_action[:, :sequence_length]
    if valid_mask is not None:
        valid_mask = valid_mask[:, :sequence_length]

    valid = valid_mask.float() if valid_mask is not None else torch.ones_like(expert_action[..., 0])
    horizon = pred_sequence.size(2)
    terms = []
    for k in range(horizon):
        target = torch.cat(
            [expert_action[:, k:], expert_action[:, -1:].expand(-1, k, -1)],
            dim=1,
        )
        if k == 0:
            mask = valid
        else:
            mask = torch.cat([valid[:, k:], torch.zeros_like(valid[:, :k])], dim=1)
        per_t = weighted_mean_action_squared_error(pred_sequence[:, :, k], target.float(), cfg).unsqueeze(-1)
        terms.append(masked_mean(per_t, mask))
    return torch.stack(terms).mean()


def world_model_dit_loss(
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    cfg: ModelConfig,
    valid_mask: Optional[torch.Tensor] = None,
    *,
    localization_only: bool = False,
) -> Dict[str, torch.Tensor]:
    ref = outputs.get("feat", outputs.get("obs_embed"))
    if ref is None:
        ref = batch["expert_action"]
    device = ref.device
    dtype = ref.dtype

    train_kl = bool(getattr(cfg, "train_kl", True))
    train_direct_action = bool(getattr(cfg, "train_direct_action", True))
    train_next_target_relative = bool(getattr(cfg, "train_next_target_relative", False))
    losses: Dict[str, torch.Tensor] = {}

    if train_kl and outputs.get("priors") is not None and outputs.get("posts") is not None:
        priors = outputs["priors"]
        posts = outputs["posts"]
        losses["kl"] = masked_mean(
            kl_normal(posts["mean"], posts["std"], priors["mean"], priors["std"]),
            valid_mask,
        )
    else:
        losses["kl"] = torch.zeros((), device=device, dtype=dtype)

    z = torch.zeros((), device=device, dtype=dtype)
    if not train_next_target_relative:
        losses["next_target_relative"] = z
        losses["prior_next_target_relative"] = z
    else:
        if "next_target_relative" not in losses:
            target_next = batch["next_target_relative"].float()
            next_mask = valid_mask
            if outputs["next_target_relative"].shape[:2] != target_next.shape[:2]:
                out_t = outputs["next_target_relative"].size(1)
                src_t = target_next.size(1)
                idx = torch.linspace(0, src_t - 1, out_t, device=target_next.device).round().long()
                target_next = target_next[:, idx]
                next_mask = None if valid_mask is None else valid_mask[:, idx]
            losses["next_target_relative"] = masked_mean(
                (outputs["next_target_relative"] - target_next).pow(2),
                next_mask,
            )
        if outputs.get("priors") is None:
            losses["prior_next_target_relative"] = z
        elif "prior_next_target_relative" not in losses:
            losses["prior_next_target_relative"] = masked_mean(
                (outputs["prior_next_target_relative"] - batch["next_target_relative"].float()).pow(2),
                valid_mask,
            )

    expert_action = batch["expert_action"]
    if "video_flow_loss" in outputs:
        losses["video"] = outputs["video_flow_loss"]
        losses["video_x0"] = torch.zeros((), device=device, dtype=dtype)
    elif "video_diffusion_loss" in outputs:
        losses["video"] = outputs["video_diffusion_loss"]
        losses["video_x0"] = outputs.get("video_x0_loss", torch.zeros((), device=device, dtype=dtype))
    else:
        losses["video"] = torch.zeros((), device=device, dtype=dtype)
        losses["video_x0"] = torch.zeros((), device=device, dtype=dtype)

    if train_direct_action and "policy_flow_loss" in outputs:
        losses["action"] = outputs["policy_flow_loss"]
        if (
            float(getattr(cfg, "x0_action_loss_weight", 0.0)) > 0.0
            and "policy_action_sequence" in outputs
        ):
            losses["x0_action"] = action_sequence_loss(
                outputs["policy_action_sequence"],
                expert_action.float(),
                outputs.get("policy_action_valid_mask", valid_mask),
                cfg,
            )
        else:
            losses["x0_action"] = torch.zeros((), device=device, dtype=dtype)
    elif train_direct_action and "policy_diffusion_loss" in outputs:
        losses["action"] = outputs["policy_diffusion_loss"]
        if float(getattr(cfg, "x0_action_loss_weight", 0.0)) > 0.0 and "policy_action_sequence" in outputs:
            losses["x0_action"] = action_sequence_loss(
                outputs["policy_action_sequence"],
                expert_action.float(),
                valid_mask,
                cfg,
            )
        else:
            losses["x0_action"] = torch.zeros((), device=device, dtype=dtype)
    elif train_direct_action and "policy_action_sequence" in outputs:
        losses["action"] = action_sequence_loss(
            outputs["policy_action_sequence"],
            expert_action.float(),
            valid_mask,
            cfg,
        )
        losses["x0_action"] = torch.zeros((), device=device, dtype=dtype)
    elif train_direct_action and "policy_action" in outputs:
        pred = outputs["policy_action"]
        tgt = expert_action.float()
        per_t = weighted_mean_action_squared_error(pred, tgt, cfg).unsqueeze(-1)
        losses["action"] = masked_mean(per_t, valid_mask)
        losses["x0_action"] = torch.zeros((), device=device, dtype=dtype)
    else:
        losses["action"] = torch.zeros((), device=device, dtype=dtype)
        losses["x0_action"] = torch.zeros((), device=device, dtype=dtype)

    losses["fastwam_attention_heatmap"] = z
    losses["fastwam_ortrack_consistency"] = z
    losses["center_flow"] = outputs.get("center_flow_loss", z)
    losses["current_center"] = outputs.get("current_center_loss", z)
    losses["future_center"] = outputs.get("future_center_loss", z)
    losses["center_transition"] = outputs.get("center_transition_loss", z)
    losses["box_l1"] = outputs.get("box_l1_loss", z)
    losses["box_giou"] = outputs.get("box_giou_loss", z)
    losses["state_flow"] = outputs.get("state_flow_loss", z)
    losses["current_box"] = outputs.get("current_box_loss", z)
    losses["current_center_spatial"] = outputs.get("current_center_spatial_loss", z)
    losses["current_box_giou"] = outputs.get("current_box_giou_loss", z)
    losses["current_attention"] = outputs.get("current_attention_loss", z)
    losses["capture_value"] = outputs.get("capture_value_loss", z)
    for key in (
        "state_valid_ratio",
        "current_box_valid_ratio",
        "state_to_action_gate_mean",
        "current_box_action_gate_mean",
        "predicted_s0_box_error",
        "predicted_s0_center_error_pixels",
        "predicted_future_state_error",
        "capture_value_capture_loss",
        "capture_value_distance_loss",
        "capture_value_visibility_loss",
        "capture_value_ranking_loss",
        "capture_value_ranking_accuracy",
        "capture_value_target_capture",
    ):
        losses[key] = outputs.get(key, z)
    if bool(getattr(cfg, "use_fastwam_attention_heatmap_loss", False)):
        if "fastwam_attention_heatmap_loss" not in outputs:
            raise RuntimeError("FastWAM attention heatmap loss is enabled but missing from model outputs.")
        losses["fastwam_attention_heatmap"] = outputs["fastwam_attention_heatmap_loss"]
    if bool(getattr(cfg, "use_fastwam_tracker_heatmap_loss", False)):
        if "fastwam_ortrack_consistency_loss" not in outputs:
            raise RuntimeError("ORTrack consistency loss is enabled but missing from model outputs.")
        losses["fastwam_ortrack_consistency"] = outputs["fastwam_ortrack_consistency_loss"]
    if bool(getattr(cfg, "tracker_center_flow_supervision", False)) and "center_flow_loss" not in outputs:
        raise RuntimeError("Center-flow supervision is enabled but the model did not return center_flow_loss.")

    # DiT actor uses the standard diffusion denoising objective as
    # losses["action"]; an optional x0 reconstruction term keeps the sampled
    # clean trajectory aligned with expert actions.

    kl_w = float(cfg.kl_weight)

    total = torch.zeros((), device=device, dtype=dtype)
    if bool(getattr(cfg, "use_future_state_dit", False)):
        required_v4_losses = {
            "state_flow_loss",
            "current_box_loss",
            "current_center_spatial_loss",
            "current_box_giou_loss",
            "current_attention_loss",
        }
        missing_v4_losses = sorted(required_v4_losses.difference(outputs))
        if missing_v4_losses:
            raise RuntimeError(
                f"V4 Future State DiT losses are missing: {missing_v4_losses}."
            )
        localization_total = (
            float(getattr(cfg, "current_box_weight", 5.0)) * losses["current_box"]
            + float(getattr(cfg, "current_center_weight", 5.0))
            * losses["current_center_spatial"]
            + float(getattr(cfg, "current_box_giou_weight", 2.0))
            * losses["current_box_giou"]
            + float(getattr(cfg, "current_attention_weight", 1.0))
            * losses["current_attention"]
        )
        joint_total = (
            float(getattr(cfg, "fastwam_lambda_action", 1.0)) * losses["action"]
            + float(getattr(cfg, "fastwam_lambda_video", 1.0)) * losses["video"]
            + float(getattr(cfg, "x0_action_loss_weight", 0.0))
            * losses["x0_action"]
            + float(getattr(cfg, "future_state_flow_weight", 1.0)) * losses["state_flow"]
            + localization_total
        )
        # Keep all expert loss graphs connected under DDP/DeepSpeed during the
        # localization warmup, while applying gradients only to s0 localization.
        total = (
            localization_total
            + 0.0
            * (
                losses["action"]
                + losses["video"]
                + losses["x0_action"]
                + losses["state_flow"]
            )
            if localization_only
            else joint_total
        )
        losses["localization_total"] = localization_total
        losses["joint_total"] = joint_total
        losses["localization_warmup_active"] = torch.as_tensor(
            float(localization_only), device=device, dtype=dtype
        )
    elif train_kl:
        total = total + kl_w * losses["kl"]
    uses_structured_future = bool(getattr(cfg, "use_future_state_dit", False))
    if not uses_structured_future and train_next_target_relative:
        total = total + float(cfg.next_target_relative_loss_weight) * losses["next_target_relative"]
        total = total + float(cfg.prior_target_relative_loss_weight) * losses["prior_next_target_relative"]
    if not uses_structured_future and bool(getattr(cfg, "use_fastwam_attention_heatmap_loss", False)):
        total = total + float(cfg.fastwam_attention_heatmap_loss_weight) * losses["fastwam_attention_heatmap"]
    if not uses_structured_future and bool(getattr(cfg, "use_fastwam_tracker_heatmap_loss", False)):
        total = total + float(cfg.fastwam_ortrack_consistency_loss_weight) * losses["fastwam_ortrack_consistency"]
    if not uses_structured_future and bool(getattr(cfg, "use_fastwam_mot", False)):
        capture_value_only = bool(
            getattr(cfg, "use_capture_value_reranking", False)
            and getattr(cfg, "capture_value_adapter_only", False)
        )
        if capture_value_only:
            # Keep the frozen parent metrics visible without letting their
            # stochastic denoising losses determine the best Value checkpoint.
            total = total + 0.0 * (
                losses["action"] + losses["video"] + losses["x0_action"]
            )
        else:
            total = total + float(getattr(cfg, "fastwam_lambda_action", 1.0)) * losses["action"]
            total = total + float(getattr(cfg, "fastwam_lambda_video", 1.0)) * losses["video"]
            total = total + float(getattr(cfg, "x0_action_loss_weight", 0.0)) * losses["x0_action"]
            if bool(getattr(cfg, "tracker_center_flow_supervision", False)):
                total = total + float(getattr(cfg, "tracker_center_flow_loss_weight", 0.1)) * losses["center_flow"]
        if bool(getattr(cfg, "use_capture_value_reranking", False)):
            if "capture_value_loss" not in outputs:
                raise RuntimeError(
                    "Capture-value reranking is enabled but its loss is missing."
                )
            total = total + float(
                getattr(cfg, "capture_value_loss_weight", 0.2)
            ) * losses["capture_value"]
    elif not uses_structured_future and train_direct_action and "policy_action" in outputs:
        total = total + float(cfg.direct_action_loss_weight) * losses["action"]
        total = total + float(getattr(cfg, "x0_action_loss_weight", 0.0)) * losses["x0_action"]
    if total.ndim > 0:
        total = total.mean()

    losses["total"] = total
    if bool(getattr(cfg, "use_future_state_dit", False)):
        # Keep the established short metric names while exposing the explicit
        # V4 loss names requested by the experiment specification.
        losses["total_loss"] = total
        losses["action_flow_loss"] = losses["action"]
        losses["video_flow_loss"] = losses["video"]
        losses["state_flow_loss"] = losses["state_flow"]
        losses["current_box_loss"] = losses["current_box"]
    return losses


@torch.no_grad()
def summarize_losses(losses: Dict[str, torch.Tensor]) -> Dict[str, float]:
    out = {}
    for k, v in losses.items():
        vv = v.detach()
        if vv.ndim > 0:
            vv = vv.mean()
        out[k] = float(vv.cpu())
    return out
