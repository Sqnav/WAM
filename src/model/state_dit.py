from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn


class FutureStateDiT(nn.Module):
    """DiT expert for eight s0-relative future target states."""

    def __init__(
        self,
        *,
        state_dim: int,
        hidden_dim: int,
        ffn_dim: int,
        freq_dim: int,
        num_heads: int,
        attn_head_dim: int,
        num_layers: int,
        horizon: int,
        eps: float,
        use_gradient_checkpointing: bool,
    ) -> None:
        super().__init__()
        from fastwam.models.wan22.wan_video_dit import DiTBlock, precompute_freqs_cis

        self.state_dim = int(state_dim)
        self.hidden_dim = int(hidden_dim)
        self.ffn_dim = int(ffn_dim)
        self.freq_dim = int(freq_dim)
        self.num_heads = int(num_heads)
        self.attn_head_dim = int(attn_head_dim)
        self.horizon = int(horizon)
        self.use_gradient_checkpointing = bool(use_gradient_checkpointing)
        self.state_encoder = nn.Sequential(
            nn.Linear(self.state_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.current_condition_norm = nn.LayerNorm(self.hidden_dim)
        self.state_modality_embedding = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        self.horizon_embedding = nn.Parameter(torch.zeros(1, self.horizon + 1, self.hidden_dim))
        self.time_embedding = nn.Sequential(
            nn.Linear(self.freq_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.time_projection = nn.Sequential(
            nn.SiLU(), nn.Linear(self.hidden_dim, self.hidden_dim * 6)
        )
        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    hidden_dim=self.hidden_dim,
                    attn_head_dim=self.attn_head_dim,
                    num_heads=self.num_heads,
                    ffn_dim=self.ffn_dim,
                    eps=float(eps),
                )
                for _ in range(int(num_layers))
            ]
        )
        self.head = nn.Linear(self.hidden_dim, self.state_dim)
        self.freqs = precompute_freqs_cis(self.attn_head_dim, end=1024)
        nn.init.trunc_normal_(self.state_modality_embedding, std=0.02)
        nn.init.trunc_normal_(self.horizon_embedding, std=0.02)

    def pre_dit(
        self,
        noisy_future_states: torch.Tensor,
        current_condition: torch.Tensor,
        timestep: torch.Tensor,
        tracker_memory: torch.Tensor,
        state_valid_mask: torch.Tensor,
    ) -> Dict[str, Any]:
        from fastwam.models.wan22.wan_video_dit import sinusoidal_embedding_1d

        if noisy_future_states.ndim != 3 or tuple(noisy_future_states.shape[1:]) != (
            self.horizon,
            self.state_dim,
        ):
            raise ValueError(
                f"noisy_future_states must be [B,{self.horizon},{self.state_dim}], "
                f"got {tuple(noisy_future_states.shape)}."
            )
        batch = noisy_future_states.size(0)
        if current_condition.shape != (batch, self.hidden_dim):
            raise ValueError(
                f"current_condition must be [B,{self.hidden_dim}], got {tuple(current_condition.shape)}."
            )
        if tracker_memory.ndim != 3 or tracker_memory.shape[:2] != (batch, 320):
            raise ValueError(f"tracker_memory must be [B,320,D], got {tuple(tracker_memory.shape)}.")
        if tracker_memory.size(-1) != self.hidden_dim:
            raise ValueError("tracker_memory hidden dimension must match StateDiT.")
        if state_valid_mask.shape != (batch, self.horizon + 1):
            raise ValueError(
                f"state_valid_mask must be [B,{self.horizon + 1}], got {tuple(state_valid_mask.shape)}."
            )
        if timestep.ndim != 1 or timestep.size(0) not in {1, batch}:
            raise ValueError("state timestep must be [1] or [B].")
        if timestep.size(0) == 1 and batch > 1:
            if self.training:
                raise ValueError("Training state timestep must match batch size.")
            timestep = timestep.expand(batch)

        dtype, device = self.state_modality_embedding.dtype, self.state_modality_embedding.device
        valid = state_valid_mask.to(device=device, dtype=torch.bool)
        current = self.current_condition_norm(
            current_condition.to(device=device, dtype=dtype)
        ).unsqueeze(1)
        future = self.state_encoder(noisy_future_states.to(device=device, dtype=dtype))
        tokens = torch.cat([current, future], dim=1)
        tokens = tokens + self.horizon_embedding + self.state_modality_embedding
        tokens = tokens * valid.unsqueeze(-1).to(tokens.dtype)

        t = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, timestep.to(device=device, dtype=dtype))
        )
        t_mod = self.time_projection(t).unflatten(1, (6, self.hidden_dim))
        seq_len = self.horizon + 1
        freqs = self.freqs[:seq_len].view(seq_len, 1, -1).to(device=device)
        memory = tracker_memory.to(device=device, dtype=dtype)
        memory_mask = torch.ones(batch, seq_len, memory.size(1), device=device, dtype=torch.bool)
        return {
            "tokens": tokens,
            "freqs": freqs,
            "t": t,
            "t_mod": t_mod,
            "context": memory,
            "context_mask": memory_mask,
            "state_valid_mask": valid,
            "meta": {"batch_size": batch, "seq_len": seq_len},
        }

    def post_dit(self, tokens: torch.Tensor, pre_state: Dict[str, Any]) -> torch.Tensor:
        del pre_state
        if tokens.ndim != 3 or tokens.size(1) != self.horizon + 1:
            raise ValueError(
                f"StateDiT tokens must be [B,{self.horizon + 1},D], got {tuple(tokens.shape)}."
            )
        flow = self.head(tokens[:, 1:])
        if not torch.isfinite(flow).all():
            raise FloatingPointError("StateDiT produced NaN/Inf state flow.")
        return flow
