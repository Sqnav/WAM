from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from .config import ModelConfig


class CrossAttentionFusion(nn.Module):
    """Fuse visual tokens with text tokens and privileged teacher information.

    The previous low-dimensional state input has been removed. Visual tokens are
    queries; text tokens and privileged token are key/value context.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.image_proj = nn.Linear(cfg.image_encoder_dim, cfg.fusion_dim)
        self.text_proj = nn.Linear(cfg.text_width, cfg.fusion_dim)
        self.priv_proj = nn.Linear(cfg.fusion_dim, cfg.fusion_dim)

        self.query_norm = nn.LayerNorm(cfg.fusion_dim)
        self.context_norm = nn.LayerNorm(cfg.fusion_dim)
        self.cross_attn = nn.MultiheadAttention(
            cfg.fusion_dim,
            cfg.fusion_heads,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.ffn_norm = nn.LayerNorm(cfg.fusion_dim)
        self.ffn = nn.Sequential(
            nn.Linear(cfg.fusion_dim, cfg.fusion_dim * cfg.fusion_ffn_mult),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.fusion_dim * cfg.fusion_ffn_mult, cfg.fusion_dim),
            nn.Dropout(cfg.dropout),
        )
        self.out_norm = nn.LayerNorm(cfg.fusion_dim)

    def forward(
        self,
        image_tokens: torch.Tensor,
        text_tokens: torch.Tensor,
        privileged_token: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if image_tokens.ndim != 3 or text_tokens.ndim != 3:
            raise ValueError("image_tokens and text_tokens must have shape [B, N, C].")
        if privileged_token.ndim != 2:
            raise ValueError("privileged_token must have shape [B, C].")

        queries = self.image_proj(image_tokens)
        txt = self.text_proj(text_tokens)
        priv = self.priv_proj(privileged_token).unsqueeze(1)
        context = torch.cat([txt, priv], dim=1)

        context_norm = self.context_norm(context)
        attn_out, _ = self.cross_attn(
            self.query_norm(queries),
            context_norm,
            context_norm,
            need_weights=False,
        )
        fused_tokens = queries + attn_out
        fused_tokens = fused_tokens + self.ffn(self.ffn_norm(fused_tokens))
        fused_tokens = self.out_norm(fused_tokens)
        fused_embed = fused_tokens[:, 0]
        return fused_embed, fused_tokens
