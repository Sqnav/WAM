from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F

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

    # DiT actor uses the standard diffusion denoising objective as
    # losses["action"]; an optional x0 reconstruction term keeps the sampled
    # clean trajectory aligned with expert actions.

    kl_w = float(cfg.kl_weight)

    total = torch.zeros((), device=device, dtype=dtype)
    if train_kl:
        total = total + kl_w * losses["kl"]
    if train_next_target_relative:
        total = total + float(cfg.next_target_relative_loss_weight) * losses["next_target_relative"]
        total = total + float(cfg.prior_target_relative_loss_weight) * losses["prior_next_target_relative"]
    if bool(getattr(cfg, "use_fastwam_mot", False)):
        total = total + float(getattr(cfg, "fastwam_lambda_action", 1.0)) * losses["action"]
        total = total + float(getattr(cfg, "fastwam_lambda_video", 1.0)) * losses["video"]
    elif train_direct_action and "policy_action" in outputs:
        total = total + float(cfg.direct_action_loss_weight) * losses["action"]
        total = total + float(getattr(cfg, "x0_action_loss_weight", 0.0)) * losses["x0_action"]
    if total.ndim > 0:
        total = total.mean()

    if bool(getattr(cfg, "use_target_belief_tracker", False)) and "target_belief_sequence" in outputs:
        pred_belief = outputs["target_belief_sequence"].float().clamp_min(1e-8)
        gt_belief = outputs["target_belief_gt"].float().clamp_min(1e-8)
        if pred_belief.shape != gt_belief.shape:
            raise ValueError("target_belief_sequence and target_belief_gt must have the same shape.")
        belief_mask = valid_mask
        if belief_mask is not None and belief_mask.shape[:2] != pred_belief.shape[:2]:
            src_t = belief_mask.size(1)
            out_t = pred_belief.size(1)
            idx = torch.linspace(0, src_t - 1, out_t, device=belief_mask.device).round().long()
            belief_mask = belief_mask[:, idx]
        per_t = F.kl_div(pred_belief.log(), gt_belief, reduction="none").sum(dim=-1, keepdim=True)
        losses["target_belief"] = masked_mean(per_t, belief_mask)
        total = total + float(getattr(cfg, "target_belief_loss_weight", 0.1)) * losses["target_belief"]
    else:
        losses["target_belief"] = torch.zeros((), device=device, dtype=dtype)

    losses["total"] = total
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
