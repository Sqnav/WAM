from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn

from .config import ModelConfig


class CrossAttentionFusion(nn.Module):
    """Fuse visual tokens with text tokens and a null target token.

    The previous low-dimensional state input has been removed. Visual tokens are
    queries; text tokens and the null target token are key/value context.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.image_proj = nn.Linear(cfg.image_encoder_dim, cfg.fusion_dim)
        self.text_proj = nn.Linear(cfg.text_width, cfg.fusion_dim)
        self.target_token_proj = nn.Linear(cfg.fusion_dim, cfg.fusion_dim)
        self.target_token_fusion_mode = str(getattr(cfg, "target_token_fusion_mode", "attention")).strip().lower()
        if self.target_token_fusion_mode not in {"attention", "concat"}:
            raise ValueError("cfg.target_token_fusion_mode must be 'attention' or 'concat'.")
        self.concat_proj = nn.Sequential(
            nn.LayerNorm(cfg.fusion_dim * 2),
            nn.Linear(cfg.fusion_dim * 2, cfg.fusion_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.fusion_dim, cfg.fusion_dim),
        )

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
        self.pool_norm = nn.LayerNorm(cfg.fusion_dim)
        self.pool_query = nn.Parameter(torch.zeros(1, 1, cfg.fusion_dim))
        self.target_bias_embed = nn.Parameter(torch.zeros(1, 1, cfg.fusion_dim))
        nn.init.trunc_normal_(self.pool_query, std=0.02)
        nn.init.trunc_normal_(self.target_bias_embed, std=0.02)

        if self.target_token_fusion_mode != "concat":
            for p in self.concat_proj.parameters():
                p.requires_grad_(False)
        if not bool(getattr(cfg, "use_patch_attention_pool", True)):
            self.pool_query.requires_grad_(False)
            for p in self.pool_norm.parameters():
                p.requires_grad_(False)
        if not (
            bool(getattr(cfg, "use_target_visual_guidance", False))
            and bool(getattr(cfg, "use_attention_heatmap", True))
        ):
            self.target_bias_embed.requires_grad_(False)

    def forward(
        self,
        image_tokens: torch.Tensor,
        text_tokens: torch.Tensor,
        target_token: torch.Tensor,
        target_patch_bias: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if image_tokens.ndim != 3 or text_tokens.ndim != 3:
            raise ValueError("image_tokens and text_tokens must have shape [B, N, C].")
        if target_token.ndim != 2:
            raise ValueError("target_token must have shape [B, C].")

        image_tokens = image_tokens.to(dtype=self.image_proj.weight.dtype)
        text_tokens = text_tokens.to(dtype=self.text_proj.weight.dtype)
        target_token = target_token.to(dtype=self.target_token_proj.weight.dtype)
        queries = self.image_proj(image_tokens)
        if target_patch_bias is not None:
            if target_patch_bias.ndim != 2:
                raise ValueError("target_patch_bias must have shape [B, N].")
            if target_patch_bias.shape != queries.shape[:2]:
                raise ValueError(
                    f"target_patch_bias shape {tuple(target_patch_bias.shape)} must match "
                    f"image token shape {tuple(queries.shape[:2])}."
                )
            target_bias = target_patch_bias.to(device=queries.device, dtype=queries.dtype).clamp_min(0.0)
            queries = queries + target_bias.unsqueeze(-1) * self.target_bias_embed.to(queries.dtype)
        else:
            target_bias = None
        txt = self.text_proj(text_tokens)
        target_ctx = self.target_token_proj(target_token).unsqueeze(1)
        if self.target_token_fusion_mode == "attention":
            context = torch.cat([txt, target_ctx], dim=1)
        else:
            context = txt

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
        if bool(getattr(self.cfg, "use_patch_attention_pool", True)):
            pool_tokens = self.pool_norm(fused_tokens)
            pool_query = self.pool_query.to(device=pool_tokens.device, dtype=pool_tokens.dtype)
            pool_logits = (pool_tokens * pool_query).sum(dim=-1) / math.sqrt(pool_tokens.size(-1))
            if target_bias is not None:
                pool_logits = pool_logits + float(getattr(self.cfg, "heatmap_attention_bias_strength", 2.0)) * target_bias
            pool_weights = torch.softmax(pool_logits, dim=-1)
            fused_embed = torch.sum(fused_tokens * pool_weights.unsqueeze(-1), dim=1)
        else:
            fused_embed = fused_tokens[:, 0]
        if self.target_token_fusion_mode == "concat":
            fused_embed = fused_embed + self.concat_proj(torch.cat([fused_embed, target_ctx.squeeze(1)], dim=-1))
        return fused_embed, fused_tokens
