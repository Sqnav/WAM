from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


CAPTURE_ACTION_PRIOR_VERSION = "capture_action_prior_v1"


def featurize_capture_action_context(
    previous_history: torch.Tensor,
    previous_valid: torch.Tensor,
    current_box: torch.Tensor,
    current_confidence: Optional[torch.Tensor],
    previous_action: Optional[torch.Tensor],
) -> torch.Tensor:
    """Build a fixed-size feature from K-1 past boxes and current b0.

    Box rows use normalized ``[cx, cy, w, h, confidence]``.  The feature keeps
    absolute image geometry, validity, first differences, and the last
    executed normalized action.  This function is shared by standalone prior
    training and online candidate scoring to prevent train/eval drift.
    """
    if previous_history.ndim != 3 or previous_history.size(-1) != 5:
        raise ValueError("previous_history must be [B,K-1,5].")
    batch, previous_length, _ = previous_history.shape
    if previous_valid.shape != (batch, previous_length):
        raise ValueError("previous_valid must be [B,K-1].")
    if current_box.shape != (batch, 4):
        raise ValueError("current_box must be [B,4].")

    dtype = current_box.dtype
    device = current_box.device
    history = previous_history.to(device=device, dtype=dtype)
    valid = previous_valid.to(device=device, dtype=torch.bool)
    current_box = current_box.to(device=device, dtype=dtype)
    if current_confidence is None:
        confidence = torch.ones(batch, 1, device=device, dtype=dtype)
    else:
        confidence = current_confidence.to(device=device, dtype=dtype).reshape(batch, -1)[:, :1]
    current = torch.cat([current_box, confidence], dim=-1)[:, None]
    sequence = torch.cat([history, current], dim=1)
    sequence_valid = torch.cat(
        [valid, torch.ones(batch, 1, device=device, dtype=torch.bool)], dim=1
    )

    centers = 2.0 * (sequence[..., :2] - 0.5)
    log_sizes = torch.log(sequence[..., 2:4].clamp_min(1.0e-4))
    conf = sequence[..., 4:5].clamp(0.0, 1.0)
    valid_float = sequence_valid[..., None].to(dtype=dtype)
    states = torch.cat([centers, log_sizes, conf, valid_float], dim=-1)
    states = states * valid_float

    delta_valid = sequence_valid[:, 1:] & sequence_valid[:, :-1]
    deltas = states[:, 1:, :4] - states[:, :-1, :4]
    deltas = deltas * delta_valid[..., None].to(dtype=dtype)

    if previous_action is None:
        action = torch.zeros(batch, 4, device=device, dtype=dtype)
    else:
        action = previous_action.to(device=device, dtype=dtype).reshape(batch, 4)
    return torch.cat(
        [states.flatten(1), deltas.flatten(1), action.clamp(-1.0, 1.0)], dim=-1
    )


class CaptureActionPrior(nn.Module):
    """Predict the normalized expert a0 and calibrated per-axis uncertainty."""

    def __init__(
        self,
        history_length: int = 8,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if history_length < 2:
            raise ValueError("history_length must be at least two.")
        self.history_length = int(history_length)
        input_dim = self.history_length * 6 + (self.history_length - 1) * 4 + 4
        self.register_buffer("feature_mean", torch.zeros(input_dim))
        self.register_buffer("feature_std", torch.ones(input_dim))
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.output = nn.Linear(hidden_dim, 8)

    def set_feature_statistics(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        if mean.shape != self.feature_mean.shape or std.shape != self.feature_std.shape:
            raise ValueError("CaptureActionPrior feature-statistic shape mismatch.")
        self.feature_mean.copy_(mean)
        self.feature_std.copy_(std.clamp_min(1.0e-4))

    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = features.to(dtype=self.feature_mean.dtype)
        x = (x - self.feature_mean) / self.feature_std
        output = self.output(self.backbone(x))
        mean = output[..., :4].tanh()
        # A bounded standard deviation makes candidate scores numerically stable.
        std = 0.05 + 0.95 * torch.sigmoid(output[..., 4:])
        return {"mean": mean, "std": std}


def capture_action_prior_loss(
    prediction: Dict[str, torch.Tensor],
    target: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    mean = prediction["mean"]
    std = prediction["std"]
    target = target.to(device=mean.device, dtype=mean.dtype)
    residual = (mean - target) / std
    nll = 0.5 * residual.square() + std.log()
    mean_loss = F.smooth_l1_loss(mean, target, beta=0.1)
    return {"loss": nll.mean() + 0.25 * mean_loss, "mean_loss": mean_loss}


def score_candidates_with_action_prior(
    candidates: torch.Tensor,
    prediction: Dict[str, torch.Tensor],
    *,
    dimension_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if candidates.ndim != 4 or candidates.size(-1) != 4:
        raise ValueError("candidates must be [B,N,H,4].")
    mean = prediction["mean"][:, None]
    std = prediction["std"][:, None].clamp_min(0.05)
    error = (candidates[:, :, 0].float() - mean) / std
    if dimension_weights is None:
        dimension_weights = error.new_tensor([0.5, 1.0, 1.0, 1.5])
    else:
        dimension_weights = dimension_weights.to(device=error.device, dtype=error.dtype)
    return -(error.square() * dimension_weights).sum(dim=-1) / dimension_weights.sum()


@dataclass(frozen=True)
class CaptureActionPriorMetadata:
    version: str
    history_length: int
    hidden_dim: int


def load_capture_action_prior(
    checkpoint_path: str,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> CaptureActionPrior:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    metadata = checkpoint.get("metadata", {})
    if metadata.get("version") != CAPTURE_ACTION_PRIOR_VERSION:
        raise ValueError(
            f"Unsupported CaptureActionPrior checkpoint version: {metadata.get('version')!r}."
        )
    # Constructing Linear layers consumes the global CPU RNG. Keep policy
    # sampling bitwise comparable with the parent model when this frozen side
    # module is enabled.
    with torch.random.fork_rng(devices=[]):
        model = CaptureActionPrior(
            history_length=int(metadata["history_length"]),
            hidden_dim=int(metadata["hidden_dim"]),
            dropout=0.0,
        )
    model.load_state_dict(checkpoint["model"])
    model.eval().requires_grad_(False)
    return model.to(device=device, dtype=dtype)
