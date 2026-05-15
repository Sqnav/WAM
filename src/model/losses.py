from __future__ import annotations

from typing import Dict, Optional

import torch

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


def _prediction_losses(
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    cfg: ModelConfig,
    valid_mask: Optional[torch.Tensor],
    output_prefix: str = "",
    loss_prefix: str = "",
) -> Dict[str, torch.Tensor]:
    losses: Dict[str, torch.Tensor] = {}

    reward_target = batch["reward"].float()
    losses[f"{loss_prefix}reward"] = masked_mean((outputs[f"{output_prefix}reward"] - reward_target).pow(2), valid_mask)
    return losses


def _weighted_aux_total(losses: Dict[str, torch.Tensor], cfg: ModelConfig, prefix: str = "") -> torch.Tensor:
    return cfg.reward_weight * losses[f"{prefix}reward"]


def world_model_dit_loss(
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    cfg: ModelConfig,
    valid_mask: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    priors = outputs["priors"]
    posts = outputs["posts"]

    losses: Dict[str, torch.Tensor] = {}
    losses["kl"] = masked_mean(kl_normal(posts["mean"], posts["std"], priors["mean"], priors["std"]), valid_mask)

    losses.update(_prediction_losses(outputs, batch, cfg, valid_mask, output_prefix="", loss_prefix=""))
    losses.update(_prediction_losses(outputs, batch, cfg, valid_mask, output_prefix="prior_", loss_prefix="prior_"))

    if "action_loss" not in outputs:
        raise KeyError("outputs must contain action_loss. Pass expert_action to model.forward during training/evaluation.")
    action_loss = outputs["action_loss"]
    if action_loss.ndim > 0:
        action_loss = action_loss.mean()
    losses["action"] = action_loss

    posterior_aux = _weighted_aux_total(losses, cfg, prefix="")
    prior_aux = _weighted_aux_total(losses, cfg, prefix="prior_")

    total = (
        cfg.kl_weight * losses["kl"]
        + posterior_aux
        + cfg.prior_loss_weight * prior_aux
        + cfg.action_loss_weight * losses["action"]
    )

    if total.ndim > 0:
        total = total.mean()

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
