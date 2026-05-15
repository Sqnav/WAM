"""Shared action reconstruction / noise MSE with per-dimension weights (yaw emphasis)."""

from __future__ import annotations

import torch

from .config import ModelConfig


def weighted_mean_action_squared_error(pred: torch.Tensor, tgt: torch.Tensor, cfg: ModelConfig) -> torch.Tensor:
    """Mean squared error over the last dimension.

    For ``action_dim == 4`` (vx, vy, vz, yaw_norm), the yaw channel index 3 is weighted by
    ``cfg.action_yaw_loss_weight``; other dims weight 1. Otherwise uniform mean.
    """
    err = (pred.float() - tgt.float()).pow(2)
    dim = err.shape[-1]
    adim = int(getattr(cfg, "action_dim", dim))
    yaw_w = float(getattr(cfg, "action_yaw_loss_weight", 1.0))
    if dim == 4 and adim == 4 and abs(yaw_w - 1.0) > 1e-8:
        w = err.new_tensor([1.0, 1.0, 1.0, yaw_w])
        return (err * w).sum(dim=-1) / w.sum().clamp(min=1e-8)
    return err.mean(dim=-1)
