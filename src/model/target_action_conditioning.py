from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn as nn


TARGET_HISTORY_RAW_DIM = 5
TARGET_HISTORY_FEATURE_DIM = 10


def augment_target_box_history(
    history: torch.Tensor,
    history_valid: torch.Tensor,
    *,
    partial_history_probability: float = 0.0,
    center_jitter_std: float = 0.0,
    log_size_jitter_std: float = 0.0,
    confidence_dropout_probability: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply online-like corruption to previous normalized cxcywh/confidence states."""
    squeeze = history.ndim == 2
    value = history.unsqueeze(0) if squeeze else history
    valid = history_valid.unsqueeze(0) if squeeze else history_valid
    if (
        value.ndim != 3
        or value.size(-1) != TARGET_HISTORY_RAW_DIM
        or valid.shape != value.shape[:2]
    ):
        raise ValueError("Expected target history [B,K,5] and validity [B,K].")
    value = value.float().clone()
    valid = valid.bool().clone()
    batch_size, history_length = valid.shape

    partial_probability = min(max(float(partial_history_probability), 0.0), 1.0)
    if partial_probability > 0.0 and history_length > 0:
        use_partial = torch.rand(batch_size, device=value.device) < partial_probability
        keep = torch.randint(0, history_length + 1, (batch_size,), device=value.device)
        positions = torch.arange(history_length, device=value.device)[None]
        partial_valid = positions >= (history_length - keep[:, None])
        valid &= torch.where(use_partial[:, None], partial_valid, True)

    center_std = max(float(center_jitter_std), 0.0)
    if center_std > 0.0:
        increments = torch.randn_like(value[..., :2]) * center_std
        center_noise = increments.cumsum(dim=1) / math.sqrt(max(float(history_length), 1.0))
        value[..., :2] = (value[..., :2] + center_noise).clamp(0.0, 1.0)

    size_std = max(float(log_size_jitter_std), 0.0)
    if size_std > 0.0:
        log_size = value[..., 2:4].clamp_min(1.0e-6).log()
        value[..., 2:4] = (log_size + torch.randn_like(log_size) * size_std).exp().clamp(0.0, 1.0)

    dropout_probability = min(
        max(float(confidence_dropout_probability), 0.0), 1.0
    )
    if dropout_probability > 0.0:
        drop = torch.rand_like(value[..., 4]) < dropout_probability
        value[..., 4] = value[..., 4].masked_fill(drop & valid, 0.0)

    value = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
    value[..., :4] = value[..., :4].clamp(0.0, 1.0)
    value[..., 4] = value[..., 4].clamp(0.0, 1.0)
    value = value * valid.to(value.dtype)[..., None]
    if squeeze:
        return value[0], valid[0]
    return value, valid


def make_online_target_box_history(
    previous_boxes: list[torch.Tensor],
    previous_confidences: list[float],
    *,
    previous_length: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Right-align previous online Tracker states; current b0 is appended in-model."""
    previous_length = max(int(previous_length), 0)
    boxes = [torch.as_tensor(box, dtype=torch.float32).reshape(4) for box in previous_boxes]
    confidences = [float(value) for value in previous_confidences]
    if len(boxes) != len(confidences):
        raise ValueError("Online target boxes and confidences must have equal length.")
    boxes = boxes[-previous_length:]
    confidences = confidences[-previous_length:]
    real_count = len(boxes)
    pad = previous_length - real_count
    rows = [torch.zeros(TARGET_HISTORY_RAW_DIM, dtype=torch.float32) for _ in range(pad)]
    rows.extend(
        torch.cat(
            [box.clamp(0.0, 1.0), torch.tensor([confidence]).clamp(0.0, 1.0)]
        )
        for box, confidence in zip(boxes, confidences)
    )
    history = (
        torch.stack(rows)
        if rows
        else torch.empty(0, TARGET_HISTORY_RAW_DIM, dtype=torch.float32)
    )
    valid = torch.tensor([False] * pad + [True] * real_count, dtype=torch.bool)
    return history.unsqueeze(0).to(device), valid.unsqueeze(0).to(device)


class HistoricalTargetMemory(nn.Module):
    """Encode target motion relative to the observed current b0."""

    def __init__(
        self,
        *,
        action_hidden_dim: int,
        history_length: int = 8,
        horizon: int = 8,
        memory_dim: int = 256,
        num_layers: int = 2,
        num_heads: int = 8,
    ) -> None:
        super().__init__()
        self.action_hidden_dim = int(action_hidden_dim)
        self.history_length = int(history_length)
        self.previous_length = self.history_length - 1
        self.horizon = int(horizon)
        self.memory_dim = int(memory_dim)
        if self.history_length < 2 or self.horizon <= 0 or self.memory_dim <= 0:
            raise ValueError("Target history dimensions must be positive.")
        if self.memory_dim % int(num_heads) != 0:
            raise ValueError("Target history memory_dim must be divisible by num_heads.")

        self.state_encoder = nn.Sequential(
            nn.Linear(TARGET_HISTORY_FEATURE_DIM, self.memory_dim),
            nn.SiLU(),
            nn.Linear(self.memory_dim, self.memory_dim),
            nn.LayerNorm(self.memory_dim),
        )
        self.history_position = nn.Parameter(
            torch.zeros(1, self.history_length, self.memory_dim)
        )
        layer = nn.TransformerEncoderLayer(
            d_model=self.memory_dim,
            nhead=int(num_heads),
            dim_feedforward=self.memory_dim * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(
            layer, num_layers=int(num_layers), norm=nn.LayerNorm(self.memory_dim)
        )
        self.horizon_query = nn.Parameter(
            torch.zeros(1, self.horizon, self.memory_dim)
        )
        self.horizon_attention = nn.MultiheadAttention(
            self.memory_dim, int(num_heads), dropout=0.0, batch_first=True
        )
        self.previous_action_encoder = nn.Sequential(
            nn.Linear(4, self.memory_dim),
            nn.SiLU(),
            nn.Linear(self.memory_dim, self.memory_dim),
            nn.LayerNorm(self.memory_dim),
        )
        self.output = nn.Sequential(
            nn.Linear(self.memory_dim, self.action_hidden_dim),
            nn.LayerNorm(self.action_hidden_dim),
        )
        nn.init.normal_(self.history_position, std=0.02)
        nn.init.normal_(self.horizon_query, std=0.02)

    def _features(
        self,
        previous_history: torch.Tensor,
        previous_valid: torch.Tensor,
        current_box: torch.Tensor,
        current_confidence: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if previous_history.ndim != 3 or previous_history.shape[1:] != (
            self.previous_length,
            TARGET_HISTORY_RAW_DIM,
        ):
            raise ValueError(
                "Previous target history must be "
                f"[B,{self.previous_length},{TARGET_HISTORY_RAW_DIM}]."
            )
        batch_size = previous_history.size(0)
        if previous_valid.shape != (batch_size, self.previous_length):
            raise ValueError("Previous target history validity shape is invalid.")
        if current_box.shape != (batch_size, 4):
            raise ValueError("Current target box must be normalized cxcywh [B,4].")
        if current_confidence is None:
            confidence = torch.ones(
                batch_size, 1, device=current_box.device, dtype=current_box.dtype
            )
        else:
            confidence = current_confidence.reshape(batch_size, -1)[:, :1].to(
                device=current_box.device, dtype=current_box.dtype
            )
        current = torch.cat(
            [current_box.detach().clamp(0.0, 1.0), confidence.detach().clamp(0.0, 1.0)],
            dim=-1,
        )
        raw = torch.cat(
            [
                previous_history.to(device=current.device, dtype=current.dtype),
                current[:, None],
            ],
            dim=1,
        )
        valid = torch.cat(
            [
                previous_valid.to(device=current.device, dtype=torch.bool),
                torch.ones(batch_size, 1, device=current.device, dtype=torch.bool),
            ],
            dim=1,
        )
        box = raw[..., :4].clamp(0.0, 1.0)
        current_center = current[:, None, :2]
        current_log_size = current[:, None, 2:4].clamp_min(1.0e-6).log()
        # Absolute b0 is already supplied by CurrentBoxActionConditioner.  The
        # history branch should describe motion around that anchor instead of
        # learning a second, potentially conflicting copy of b0.
        state = torch.cat(
            [
                box[..., :2] - current_center,
                box[..., 2:4].clamp_min(1.0e-6).log() - current_log_size,
            ],
            dim=-1,
        )
        delta = torch.zeros_like(state)
        delta[:, 1:] = state[:, 1:] - state[:, :-1]
        delta_valid = torch.zeros_like(valid)
        delta_valid[:, 1:] = valid[:, 1:] & valid[:, :-1]
        delta = delta * delta_valid.to(delta.dtype)[..., None]
        relative_time = torch.linspace(
            -1.0, 0.0, self.history_length, device=raw.device, dtype=raw.dtype
        ).view(1, self.history_length, 1).expand(batch_size, -1, -1)
        features = torch.cat([state, delta, raw[..., 4:5], relative_time], dim=-1)
        features = torch.nan_to_num(features) * valid.to(raw.dtype)[..., None]
        return features, valid

    def forward(
        self,
        previous_history: torch.Tensor,
        previous_valid: torch.Tensor,
        current_box: torch.Tensor,
        current_confidence: Optional[torch.Tensor] = None,
        previous_action: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        features, valid = self._features(
            previous_history, previous_valid, current_box, current_confidence
        )
        encoder_weight = self.state_encoder[0].weight
        features = features.to(
            device=encoder_weight.device, dtype=encoder_weight.dtype
        )
        valid = valid.to(device=encoder_weight.device)
        encoded = self.state_encoder(features) + self.history_position.to(features)
        encoded = self.temporal_encoder(encoded, src_key_padding_mask=~valid)
        query = self.horizon_query.to(encoded).expand(encoded.size(0), -1, -1)
        if previous_action is None:
            action = torch.zeros(
                encoded.size(0), 4, device=encoded.device, dtype=encoded.dtype
            )
        else:
            if previous_action.shape != (encoded.size(0), 4):
                raise ValueError("Previous executed action must have shape [B,4].")
            action = previous_action.to(device=encoded.device, dtype=encoded.dtype)
        query = query + self.previous_action_encoder(action)[:, None]
        horizons, _ = self.horizon_attention(
            query, encoded, encoded, key_padding_mask=~valid, need_weights=False
        )
        return self.output(horizons)


class _ZeroResidualAdapter(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.hidden_norm = nn.LayerNorm(hidden_dim)
        self.condition_norm = nn.LayerNorm(hidden_dim)
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        nn.init.zeros_(self.fusion[-1].weight)
        nn.init.zeros_(self.fusion[-1].bias)

    def forward(self, hidden: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        if hidden.shape != condition.shape:
            raise ValueError(
                f"Action hidden and target condition must match, got {hidden.shape} and {condition.shape}."
            )
        output_dtype = hidden.dtype
        parameter = self.fusion[0].weight
        hidden = hidden.to(device=parameter.device, dtype=parameter.dtype)
        condition = condition.to(device=parameter.device, dtype=parameter.dtype)
        delta = self.fusion(
            torch.cat([self.hidden_norm(hidden), self.condition_norm(condition)], dim=-1)
        )
        return delta.to(dtype=output_dtype)


class TargetActionConditioning(nn.Module):
    """Historical target conditions aligned with Action horizon tokens."""

    def __init__(
        self,
        *,
        action_hidden_dim: int,
        history_length: int,
        horizon: int,
        memory_dim: int,
        memory_layers: int,
        memory_heads: int,
        layers: tuple[int, ...],
    ) -> None:
        super().__init__()
        self.horizon = int(horizon)
        self.layers = tuple(sorted({int(layer) for layer in layers}))
        if not self.layers or min(self.layers) < 0:
            raise ValueError("Historical target injection layers must be non-negative.")
        self.history_memory = HistoricalTargetMemory(
            action_hidden_dim=action_hidden_dim,
            history_length=history_length,
            horizon=horizon,
            memory_dim=memory_dim,
            num_layers=memory_layers,
            num_heads=memory_heads,
        )
        self.history_adapter = _ZeroResidualAdapter(action_hidden_dim)
        self.next_center_delta_head = nn.Sequential(
            nn.LayerNorm(action_hidden_dim),
            nn.Linear(action_hidden_dim, action_hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(action_hidden_dim // 2, 2),
        )

    @property
    def history_length(self) -> int:
        return self.history_memory.history_length

    def build_conditions(
        self,
        *,
        previous_history: torch.Tensor,
        previous_valid: torch.Tensor,
        current_box: torch.Tensor,
        current_confidence: Optional[torch.Tensor],
        previous_action: Optional[torch.Tensor],
    ) -> Dict[str, Optional[torch.Tensor]]:
        history = self.history_memory(
            previous_history,
            previous_valid,
            current_box,
            current_confidence,
            previous_action,
        )
        return {
            "history": history,
            "next_center_delta": self.next_center_delta_head(history[:, 0]),
        }

    def enabled_at(self, layer_idx: int) -> bool:
        return int(layer_idx) in self.layers

    def residual(
        self,
        action_hidden: torch.Tensor,
        conditions: Dict[str, Optional[torch.Tensor]],
    ) -> torch.Tensor:
        history = conditions.get("history")
        if history is None:
            raise RuntimeError("Historical target condition is missing.")
        return self.history_adapter(action_hidden, history)
