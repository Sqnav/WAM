from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig
from .encoders import MLP


class ScalarHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int = 1, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = MLP(in_dim, hidden_dim, out_dim, num_layers=3, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TeacherPredictionHeads(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        feat_dim = cfg.feature_dim
        hidden = cfg.head_hidden_dim
        self.reward = ScalarHead(feat_dim, hidden, out_dim=1, dropout=cfg.dropout)

    def forward(self, feat: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {
            "reward": self.reward(feat),
        }


class SinusoidalTimestepEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(0, half, device=t.device, dtype=torch.float32) / max(half - 1, 1)
        )
        args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class AdaLNDiTBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.Dropout(dropout),
        )
        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 6 * hidden_dim),
        )

    def _modulate(self, x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.modulation(cond).chunk(6, dim=-1)
        h = self._modulate(self.norm1(x), shift_msa, scale_msa)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + gate_msa.unsqueeze(1) * attn_out
        h2 = self._modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(h2)
        return x


class DiTActionHead(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.action_dim = cfg.action_dim
        self.hidden_dim = cfg.action_dit_hidden_dim
        self.num_steps = cfg.action_diffusion_steps

        self.scalar_embed = nn.Linear(1, self.hidden_dim)
        self.action_token_embed = nn.Parameter(torch.zeros(1, cfg.action_dim, self.hidden_dim))
        self.cond_proj = nn.Sequential(
            nn.Linear(cfg.feature_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.time_embed = nn.Sequential(
            SinusoidalTimestepEmbedding(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.blocks = nn.ModuleList(
            [AdaLNDiTBlock(self.hidden_dim, cfg.action_dit_heads, cfg.dropout) for _ in range(cfg.action_dit_depth)]
        )
        self.final_norm = nn.LayerNorm(self.hidden_dim)
        self.out_proj = nn.Linear(self.hidden_dim, 1)

        betas = torch.linspace(1e-4, 2e-2, self.num_steps, dtype=torch.float32)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)
        self.register_buffer("sqrt_alpha_bars", torch.sqrt(alpha_bars))
        self.register_buffer("sqrt_one_minus_alpha_bars", torch.sqrt(1.0 - alpha_bars))
        nn.init.trunc_normal_(self.action_token_embed, std=0.02)

    def _expand_time(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if t.ndim == 0:
            t = t.unsqueeze(0)
        if t.ndim == 1 and t.shape[0] != x.shape[0]:
            t = t.expand(x.shape[0])
        return t

    def forward(self, feat: torch.Tensor, noisy_action: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        original_shape = noisy_action.shape
        if noisy_action.ndim == 3:
            batch = noisy_action.shape[0] * noisy_action.shape[1]
            noisy_action = noisy_action.reshape(batch, noisy_action.shape[-1])
            feat = feat.reshape(batch, feat.shape[-1])
        elif noisy_action.ndim != 2:
            raise ValueError("noisy_action must have shape [B, A] or [B, T, A].")

        t = self._expand_time(t, noisy_action)
        cond = self.cond_proj(feat) + self.time_embed(t)
        x = self.scalar_embed(noisy_action.unsqueeze(-1)) + self.action_token_embed
        for blk in self.blocks:
            x = blk(x, cond)
        pred_noise = self.out_proj(self.final_norm(x)).squeeze(-1)
        return pred_noise.view(*original_shape)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        sqrt_ab = self.sqrt_alpha_bars[t].view(-1, *([1] * (x0.ndim - 1)))
        sqrt_omab = self.sqrt_one_minus_alpha_bars[t].view(-1, *([1] * (x0.ndim - 1)))
        return sqrt_ab * x0 + sqrt_omab * noise

    def predict_x0(self, xt: torch.Tensor, t: torch.Tensor, pred_noise: torch.Tensor) -> torch.Tensor:
        sqrt_ab = self.sqrt_alpha_bars[t].view(-1, *([1] * (xt.ndim - 1)))
        sqrt_omab = self.sqrt_one_minus_alpha_bars[t].view(-1, *([1] * (xt.ndim - 1)))
        x0 = (xt - sqrt_omab * pred_noise) / torch.clamp(sqrt_ab, min=1e-6)
        return x0.clamp(-1.0, 1.0)

    def diffusion_loss(
        self,
        feat: torch.Tensor,
        expert_action: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        device = expert_action.device
        if expert_action.ndim == 3:
            batch = expert_action.shape[0] * expert_action.shape[1]
            flat_action = expert_action.reshape(batch, expert_action.shape[-1])
            flat_feat = feat.reshape(batch, feat.shape[-1])
            flat_mask = None if valid_mask is None else valid_mask.reshape(batch)
        elif expert_action.ndim == 2:
            flat_action = expert_action
            flat_feat = feat
            flat_mask = valid_mask
        else:
            raise ValueError("expert_action must have shape [B, A] or [B, T, A].")

        t = torch.randint(0, self.num_steps, (flat_action.shape[0],), device=device)
        noise = torch.randn_like(flat_action)
        xt = self.q_sample(flat_action, t, noise)
        pred_noise = self.forward(flat_feat, xt, t)
        per_item = F.mse_loss(pred_noise, noise, reduction="none").mean(dim=-1)

        if flat_mask is not None:
            flat_mask = flat_mask.float()
            loss = (per_item * flat_mask).sum() / flat_mask.sum().clamp(min=1.0)
        else:
            loss = per_item.mean()

        pred_x0 = self.predict_x0(xt, t, pred_noise)
        if expert_action.ndim == 3:
            pred_x0 = pred_x0.view_as(expert_action)
            pred_noise = pred_noise.view_as(expert_action)
        return {"loss": loss, "pred_action": pred_x0, "pred_noise": pred_noise}

    @torch.no_grad()
    def sample(self, feat: torch.Tensor, num_steps: Optional[int] = None, deterministic: bool = True) -> torch.Tensor:
        if feat.ndim == 3:
            flat_feat = feat.reshape(-1, feat.shape[-1])
            original_shape = feat.shape[:-1]
        elif feat.ndim == 2:
            flat_feat = feat
            original_shape = feat.shape[:-1]
        else:
            raise ValueError("feat must have shape [B, D] or [B, T, D].")

        if deterministic:
            x = torch.zeros(flat_feat.shape[0], self.action_dim, device=flat_feat.device)
        else:
            x = torch.randn(flat_feat.shape[0], self.action_dim, device=flat_feat.device)

        steps = self.num_steps if num_steps is None else min(num_steps, self.num_steps)
        start_t = self.num_steps - 1
        stride = max(self.num_steps // steps, 1)
        time_indices = list(range(start_t, -1, -stride))
        if time_indices[-1] != 0:
            time_indices.append(0)

        for t_value in time_indices:
            t = torch.full((x.shape[0],), t_value, device=x.device, dtype=torch.long)
            pred_noise = self.forward(flat_feat, x, t)
            x0 = self.predict_x0(x, t, pred_noise)
            if t_value == 0:
                x = x0
                continue
            alpha = self.alphas[t].view(-1, 1)
            alpha_bar = self.alpha_bars[t].view(-1, 1)
            beta = self.betas[t].view(-1, 1)
            mean = (1.0 / torch.sqrt(alpha)) * (x - beta / torch.sqrt(1.0 - alpha_bar) * pred_noise)
            if deterministic:
                x = mean
            else:
                noise = torch.randn_like(x)
                x = mean + torch.sqrt(beta) * noise

        x = x.clamp(-1.0, 1.0)
        return x.view(*original_shape, self.action_dim)
