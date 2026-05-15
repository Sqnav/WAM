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
    losses[f"{loss_prefix}reward"] = masked_mean(
        (outputs[f"{output_prefix}reward"] - reward_target).pow(2), valid_mask
    )
    return losses


def _weighted_aux_total(losses: Dict[str, torch.Tensor], cfg: ModelConfig, prefix: str = "") -> torch.Tensor:
    return cfg.reward_weight * losses[f"{prefix}reward"]


def _get_kl_weight(
    cfg: ModelConfig,
    global_step: Optional[int] = None,
) -> float:
    final_kl_weight = float(getattr(cfg, "kl_weight", 1.0))
    warmup_steps = int(getattr(cfg, "kl_warmup_steps", 0))
    warmup_start = float(getattr(cfg, "kl_warmup_start", 0.0))

    if global_step is None or warmup_steps <= 0:
        return final_kl_weight

    progress = min(max(float(global_step) / float(warmup_steps), 0.0), 1.0)
    scale = warmup_start + (1.0 - warmup_start) * progress
    return final_kl_weight * scale


def world_model_dit_loss(
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    cfg: ModelConfig,
    valid_mask: Optional[torch.Tensor] = None,
    global_step: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    priors = outputs["priors"]
    posts = outputs["posts"]
    device = posts["mean"].device
    dtype = posts["mean"].dtype

    train_kl = bool(getattr(cfg, "train_kl", True))
    train_reward_aux = bool(getattr(cfg, "train_reward_aux", True))
    train_privileged_recon = bool(getattr(cfg, "train_privileged_recon", True))
    train_direct_action = bool(getattr(cfg, "train_direct_action", True))

    losses: Dict[str, torch.Tensor] = {}

    if train_kl:
        losses["kl"] = masked_mean(
            kl_normal(posts["mean"], posts["std"], priors["mean"], priors["std"]),
            valid_mask,
        )
    else:
        losses["kl"] = torch.zeros((), device=device, dtype=dtype)

    if train_reward_aux:
        losses.update(
            _prediction_losses(outputs, batch, cfg, valid_mask, output_prefix="", loss_prefix="")
        )
        losses.update(
            _prediction_losses(
                outputs, batch, cfg, valid_mask, output_prefix="prior_", loss_prefix="prior_"
            )
        )
    else:
        z = torch.zeros((), device=device, dtype=dtype)
        losses["reward"] = z
        losses["prior_reward"] = z

    expert_action = batch["expert_action"]

    if train_privileged_recon and "privileged_recon" in outputs:
        priv_tgt = batch["privileged"].float()
        pr = outputs["privileged_recon"]
        losses["privileged_recon"] = masked_mean(
            (pr - priv_tgt).pow(2).mean(dim=-1, keepdim=True),
            valid_mask,
        )
    else:
        losses["privileged_recon"] = torch.zeros((), device=device, dtype=dtype)

    if train_direct_action and "policy_action" in outputs:
        pred = outputs["policy_action"]
        tgt = expert_action.float()
        per_t = weighted_mean_action_squared_error(pred, tgt, cfg).unsqueeze(-1)
        losses["action_direct"] = masked_mean(per_t, valid_mask)
    else:
        losses["action_direct"] = torch.zeros((), device=device, dtype=dtype)

    losses["action_diffusion_noise"] = torch.zeros((), device=device, dtype=dtype)
    losses["action_x0"] = torch.zeros((), device=device, dtype=dtype)

    # 教师训练：DiT 时不再把噪声预测 / x0 辅助项计入 total，仅对采样动作做 BC（见 policy_action）。
    losses["action"] = losses["action_direct"]

    kl_weight = _get_kl_weight(cfg, global_step=global_step)
    losses["kl_weight"] = losses["kl"].new_tensor(kl_weight)

    total = torch.zeros((), device=device, dtype=dtype)
    if train_kl:
        total = total + kl_weight * losses["kl"]
    if train_reward_aux:
        total = total + _weighted_aux_total(losses, cfg, prefix="")
        total = total + cfg.prior_loss_weight * _weighted_aux_total(losses, cfg, prefix="prior_")
    if train_privileged_recon and "privileged_recon" in outputs:
        total = total + float(cfg.privileged_recon_loss_weight) * losses["privileged_recon"]
    if train_direct_action and "policy_action" in outputs:
        total = total + float(cfg.direct_action_loss_weight) * losses["action_direct"]

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
