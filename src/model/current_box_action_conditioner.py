from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn


class CurrentBoxActionConditioner(nn.Module):
    """Broadcast one observed b0 box into selected Action DiT layers."""

    def __init__(
        self,
        *,
        hidden_dim: int = 1024,
        layers: Iterable[int] = (18, 23, 26, 29),
        gate_init: float = 0.0,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.layers = tuple(sorted({int(layer) for layer in layers}))
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")
        if not self.layers or self.layers[0] < 0:
            raise ValueError("At least one non-negative Action layer is required.")

        self.box_encoder = nn.Sequential(
            nn.Linear(4, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
        )
        self.action_norms = nn.ModuleDict(
            {str(layer): nn.LayerNorm(self.hidden_dim) for layer in self.layers}
        )
        self.fusion_layers = nn.ModuleDict(
            {
                str(layer): nn.Sequential(
                    nn.Linear(self.hidden_dim * 2, self.hidden_dim),
                    nn.SiLU(),
                    nn.Linear(self.hidden_dim, self.hidden_dim),
                )
                for layer in self.layers
            }
        )
        self.gates = nn.ParameterDict(
            {
                str(layer): nn.Parameter(
                    torch.tensor(float(gate_init), dtype=torch.float32),
                    requires_grad=False,
                )
                for layer in self.layers
            }
        )

    def enabled_at(self, layer_idx: int) -> bool:
        return int(layer_idx) in self.layers

    def encode_box(self, current_box: torch.Tensor) -> torch.Tensor:
        if current_box.ndim != 2 or current_box.size(-1) != 4:
            raise ValueError(
                f"current_box must be normalized cxcywh [B,4], got {tuple(current_box.shape)}."
            )
        if not torch.isfinite(current_box).all():
            raise FloatingPointError("current_box contains NaN/Inf.")
        # The pretrained Tracker is frozen for this baseline. Detaching here
        # makes that architectural boundary explicit.
        return self.box_encoder(current_box.detach().clamp(0.0, 1.0))

    def delta(
        self,
        layer_idx: int,
        action_hidden: torch.Tensor,
        box_feature: torch.Tensor,
    ) -> torch.Tensor:
        key = str(int(layer_idx))
        if key not in self.fusion_layers:
            return torch.zeros_like(action_hidden)
        if action_hidden.ndim != 3 or action_hidden.size(-1) != self.hidden_dim:
            raise ValueError(
                "action_hidden must be "
                f"[B,T,{self.hidden_dim}], got {tuple(action_hidden.shape)}."
            )
        if box_feature.shape != (action_hidden.size(0), self.hidden_dim):
            raise ValueError(
                f"box_feature must be [B,{self.hidden_dim}], got {tuple(box_feature.shape)}."
            )
        broadcast_box = box_feature[:, None].expand(-1, action_hidden.size(1), -1)
        fused = self.fusion_layers[key](
            torch.cat([self.action_norms[key](action_hidden), broadcast_box], dim=-1)
        )
        return fused

    def gate_mean(self) -> torch.Tensor:
        # The residual is always applied at full strength. Keep legacy gate
        # parameters only so checkpoints produced by the gated model still load.
        first_gate = next(iter(self.gates.values()))
        return torch.ones((), device=first_gate.device, dtype=torch.float32)
