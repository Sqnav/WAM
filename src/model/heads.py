from __future__ import annotations

import math
import os
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as gradient_checkpoint

from data.visual_guidance import make_attention_heatmap, project_body_to_image
from .action_loss_utils import weighted_mean_action_squared_error
from .config import ModelConfig
from .encoders import MLP, _ensure_fastwam_path, _torch_dtype_from_name


def build_center_gaussian(
    center_xy: torch.Tensor,
    grid_height: int,
    grid_width: int,
    sigma: float,
    visible: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Build an unnormalized Gaussian on visual-token cell centers."""
    if center_xy.ndim != 2 or center_xy.size(-1) != 2:
        raise ValueError(f"center_xy must have shape [B,2], got {tuple(center_xy.shape)}.")
    if grid_height <= 0 or grid_width <= 0 or sigma <= 0.0:
        raise ValueError("grid dimensions and center Gaussian sigma must be positive.")
    center = center_xy.float()
    valid = torch.isfinite(center).all(dim=-1)
    valid = valid & (center >= 0.0).all(dim=-1) & (center <= 1.0).all(dim=-1)
    if visible is not None:
        valid = valid & visible.reshape(center.size(0), -1)[:, 0].to(device=center.device).bool()
    # NaN * 0 is still NaN. Replace invalid coordinates before evaluating the
    # Gaussian, then mask the resulting finite map to exact zero.
    safe_center = torch.where(valid[:, None], center, torch.zeros_like(center))
    gy = torch.arange(grid_height, device=center.device, dtype=center.dtype) + 0.5
    gx = torch.arange(grid_width, device=center.device, dtype=center.dtype) + 0.5
    yy, xx = torch.meshgrid(gy, gx, indexing="ij")
    dx = xx.unsqueeze(0) - safe_center[:, 0, None, None] * float(grid_width)
    dy = yy.unsqueeze(0) - safe_center[:, 1, None, None] * float(grid_height)
    gaussian = torch.exp(-(dx.square() + dy.square()) / (2.0 * float(sigma) ** 2))
    return gaussian.flatten(1) * valid[:, None].to(gaussian.dtype)


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
        self.next_target_relative = ScalarHead(feat_dim, hidden, out_dim=cfg.target_relative_dim, dropout=cfg.dropout)

    def forward(self, feat: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {
            "next_target_relative": self.next_target_relative(feat),
        }


class DirectActionHead(nn.Module):
    """MLP mapping RSSM feature to normalized expert action in [-1, 1]."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.net = MLP(
            cfg.feature_dim,
            cfg.head_hidden_dim,
            cfg.action_dim,
            num_layers=3,
            dropout=cfg.dropout,
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net(feat))


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
        if t.is_floating_point():
            emb = emb.to(dtype=t.dtype)
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
        self.action_horizon = max(int(getattr(cfg, "action_sequence_horizon", 1)), 1)
        self.output_dim = self.action_dim * self.action_horizon
        self.hidden_dim = cfg.action_dit_hidden_dim
        self.num_steps = cfg.action_diffusion_steps

        self.scalar_embed = nn.Linear(1, self.hidden_dim)
        self.action_token_embed = nn.Parameter(torch.zeros(1, self.output_dim, self.hidden_dim))
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
        if noisy_action.ndim == 4:
            batch = noisy_action.shape[0] * noisy_action.shape[1]
            noisy_action = noisy_action.reshape(batch, self.output_dim)
            feat = feat.reshape(batch, feat.shape[-1])
        elif noisy_action.ndim == 3:
            batch = noisy_action.shape[0] * noisy_action.shape[1]
            noisy_action = noisy_action.reshape(batch, noisy_action.shape[-1])
            feat = feat.reshape(batch, feat.shape[-1])
        elif noisy_action.ndim != 2:
            raise ValueError("noisy_action must have shape [B, A], [B, T, A], or [B, T, H, A].")
        if noisy_action.shape[-1] != self.output_dim:
            raise ValueError(
                f"DiT noisy_action last dimension must be H*action_dim={self.output_dim}, "
                f"got {noisy_action.shape[-1]}."
            )

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
        original_action_shape = expert_action.shape
        if expert_action.ndim == 3 and expert_action.shape[-1] == self.action_dim:
            seq_targets = []
            for k in range(self.action_horizon):
                seq_targets.append(
                    torch.cat(
                        [expert_action[:, k:], expert_action[:, -1:].expand(-1, k, -1)],
                        dim=1,
                    )
                )
            expert_action = torch.stack(seq_targets, dim=2)

        if expert_action.ndim == 4:
            batch = expert_action.shape[0] * expert_action.shape[1]
            flat_action = expert_action.reshape(batch, self.output_dim)
            flat_feat = feat.reshape(batch, feat.shape[-1])
            flat_mask = None if valid_mask is None else valid_mask.reshape(batch)
        elif expert_action.ndim == 3:
            batch = expert_action.shape[0] * expert_action.shape[1]
            flat_action = expert_action.reshape(batch, expert_action.shape[-1])
            flat_feat = feat.reshape(batch, feat.shape[-1])
            flat_mask = None if valid_mask is None else valid_mask.reshape(batch)
        elif expert_action.ndim == 2:
            if expert_action.shape[-1] == self.action_dim:
                expert_action = expert_action.unsqueeze(1).expand(-1, self.action_horizon, -1)
                flat_action = expert_action.reshape(expert_action.shape[0], self.output_dim)
            else:
                flat_action = expert_action
            flat_feat = feat
            flat_mask = valid_mask
        else:
            raise ValueError("expert_action must have shape [B, A], [B, T, A], or [B, T, H, A].")
        if flat_action.shape[-1] != self.output_dim:
            raise ValueError(
                f"DiT expert_action last dimension must be H*action_dim={self.output_dim}, "
                f"got {flat_action.shape[-1]}."
            )

        t = torch.randint(0, self.num_steps, (flat_action.shape[0],), device=device)
        noise = torch.randn_like(flat_action)
        xt = self.q_sample(flat_action, t, noise)
        pred_noise = self.forward(flat_feat, xt, t)
        per_item = weighted_mean_action_squared_error(pred_noise, noise, self.cfg)

        if flat_mask is not None:
            flat_mask = flat_mask.float()
            loss = (per_item * flat_mask).sum() / flat_mask.sum().clamp(min=1.0)
        else:
            loss = per_item.mean()

        pred_x0 = self.predict_x0(xt, t, pred_noise)
        if len(original_action_shape) == 3 and original_action_shape[-1] == self.action_dim:
            pred_x0 = pred_x0.view(*original_action_shape[:2], self.action_horizon, self.action_dim)
            pred_noise = pred_noise.view(*original_action_shape[:2], self.action_horizon, self.action_dim)
        elif expert_action.ndim in (3, 4):
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

        # Diffusion sampling starts from Gaussian noise. ``deterministic`` only
        # controls whether additional reverse-process noise is injected.
        x = torch.randn(flat_feat.shape[0], self.output_dim, device=flat_feat.device)

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
        return x.view(*original_shape, self.action_horizon, self.action_dim)


class FlowMatchScheduler:
    """Continuous flow-matching scheduler matching FastWAM's Wan scheduler."""

    def __init__(self, num_train_timesteps: int = 1000, shift: float = 5.0, eps: float = 1e-10) -> None:
        self.num_train_timesteps = int(num_train_timesteps)
        self.shift = float(shift)
        self.eps = float(eps)
        self._y_min, self._weight_norm_const = self._precompute_training_weight_stats()

    @staticmethod
    def _phi(u: torch.Tensor, shift: float) -> torch.Tensor:
        return shift * u / (1.0 + (shift - 1.0) * u)

    def _precompute_training_weight_stats(self) -> tuple[float, float]:
        steps = self.num_train_timesteps
        u_grid = torch.linspace(1.0, 0.0, steps + 1, dtype=torch.float64)[:-1]
        t_grid = self._phi(u_grid, self.shift) * float(steps)
        y_grid = torch.exp(-2.0 * ((t_grid - (steps / 2.0)) / steps) ** 2)
        y_min = float(y_grid.min().item())
        y_shifted = y_grid - y_min
        return y_min, float(y_shifted.mean().item())

    def sample_training_t(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        u = torch.rand((batch_size,), device=device, dtype=torch.float32)
        sigma = self._phi(u, self.shift)
        return (sigma * float(self.num_train_timesteps)).to(dtype=dtype)

    def add_noise(self, sample: torch.Tensor, noise: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        sigma = (timestep / float(self.num_train_timesteps)).to(sample.device, dtype=sample.dtype)
        sigma = sigma.view(-1, *([1] * (sample.ndim - 1)))
        return (1.0 - sigma) * sample + sigma * noise

    @staticmethod
    def training_target(sample: torch.Tensor, noise: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        del timestep
        return noise - sample

    def predict_x0(
        self,
        noisy_sample: torch.Tensor,
        model_output: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        sigma = (timestep / float(self.num_train_timesteps)).to(
            noisy_sample.device, dtype=noisy_sample.dtype
        )
        sigma = sigma.view(-1, *([1] * (noisy_sample.ndim - 1)))
        return noisy_sample - sigma * model_output.to(noisy_sample)

    def training_weight(self, timestep: torch.Tensor) -> torch.Tensor:
        t = timestep.to(dtype=torch.float32)
        steps = float(self.num_train_timesteps)
        y = torch.exp(-2.0 * ((t - (steps / 2.0)) / steps) ** 2)
        return (y - self._y_min) / (self._weight_norm_const + self.eps)

    def build_inference_schedule(
        self,
        num_inference_steps: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, Dict[str, Optional[torch.Tensor]]]:
        u_steps = torch.linspace(1.0, 0.0, int(num_inference_steps) + 1, device=device, dtype=torch.float32)
        sigma_steps = self._phi(u_steps, self.shift)
        timesteps = sigma_steps[:-1] * float(self.num_train_timesteps)
        deltas = sigma_steps[1:] - sigma_steps[:-1]
        return timesteps.to(dtype=dtype), deltas.to(dtype=dtype)

    @staticmethod
    def step(model_output: torch.Tensor, delta: torch.Tensor, sample: torch.Tensor) -> torch.Tensor:
        delta = delta.to(sample.device, dtype=sample.dtype)
        delta = delta.view(-1, *([1] * (sample.ndim - 1))) if delta.ndim > 0 else delta
        return sample + model_output * delta


class FastWAMExpertBlock(nn.Module):
    """Small DiT block with FastWAM-style pre/post split for MoT mixing."""

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.q = nn.Linear(hidden_dim, hidden_dim)
        self.k = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, hidden_dim)
        self.o = nn.Linear(hidden_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.Dropout(dropout),
        )
        self.num_heads = num_heads
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, 6 * hidden_dim))

    @staticmethod
    def _modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)

    def attention_io(self, x: torch.Tensor, cond: torch.Tensor):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.modulation(cond).chunk(6, dim=-1)
        h = self._modulate(self.norm1(x), shift_msa, scale_msa)
        return self.q(h), self.k(h), self.v(h), x, gate_msa, shift_mlp, scale_mlp, gate_mlp

    def post_attention(
        self,
        residual: torch.Tensor,
        attn_out: torch.Tensor,
        gate_msa: torch.Tensor,
        shift_mlp: torch.Tensor,
        scale_mlp: torch.Tensor,
        gate_mlp: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor | tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        x = residual + gate_msa.unsqueeze(1) * self.o(attn_out)
        if context is not None:
            context = context.reshape(context.size(0), -1, context.size(-1))
            cross_out, _ = self.cross_attn(self.norm3(x), context, context, need_weights=False)
            x = x + cross_out
        h = self._modulate(self.norm2(x), shift_mlp, scale_mlp)
        return x + gate_mlp.unsqueeze(1) * self.ffn(h)


class FastWAMVideoExpert(nn.Module):
    """Video DiT expert over fused visual-language patch tokens."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.hidden_dim = cfg.fastwam_hidden_dim
        self.num_heads = cfg.fastwam_heads
        self.attn_head_dim = self.hidden_dim // self.num_heads
        self.token_in = nn.Linear(cfg.fusion_dim, self.hidden_dim)
        self.token_out = nn.Linear(self.hidden_dim, cfg.fusion_dim)
        self.text_embedding = nn.Sequential(
            nn.Linear(cfg.fusion_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.time_embedding = nn.Sequential(
            SinusoidalTimestepEmbedding(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.blocks = nn.ModuleList(
            [FastWAMExpertBlock(self.hidden_dim, self.num_heads, cfg.dropout) for _ in range(cfg.fastwam_layers)]
        )

    def pre_dit(self, video_tokens: torch.Tensor, timestep: torch.Tensor, context: torch.Tensor) -> Dict[str, torch.Tensor]:
        if video_tokens.ndim != 4:
            raise ValueError("video_tokens must have shape [B, T, N, D].")
        b, t, n, d = video_tokens.shape
        x = self.token_in(video_tokens.reshape(b, t * n, d))
        time = self.time_embedding(timestep)
        context_emb = self.text_embedding(context)
        return {"tokens": x, "t_mod": time, "context": context_emb, "tokens_per_frame": torch.tensor(n, device=x.device)}

    def post_dit(self, tokens: torch.Tensor, pre_state: Dict[str, torch.Tensor], seq_len: int) -> torch.Tensor:
        b = tokens.size(0)
        n = int(pre_state["tokens_per_frame"].item())
        x = self.token_out(tokens).reshape(b, seq_len, n, -1)
        return x

    @staticmethod
    def build_video_to_video_mask(seq_len: int, tokens_per_frame: int, mode: str, device: torch.device) -> torch.Tensor:
        total = seq_len * tokens_per_frame
        if mode != "first_frame_causal":
            return torch.ones(total, total, dtype=torch.bool, device=device)
        frame_ids = torch.arange(seq_len, device=device).repeat_interleave(tokens_per_frame)
        query = frame_ids[:, None]
        key = frame_ids[None, :]
        return (key == 0) | (key <= query)


class FastWAMActionExpert(nn.Module):
    """Action expert DiT following FastWAM's pre_dit/post_dit interface."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.action_dim = cfg.action_dim
        self.hidden_dim = cfg.fastwam_hidden_dim
        self.num_heads = cfg.fastwam_heads
        self.attn_head_dim = self.hidden_dim // self.num_heads
        self.action_encoder = nn.Linear(cfg.action_dim, self.hidden_dim)
        self.head = nn.Linear(self.hidden_dim, cfg.action_dim)
        self.text_embedding = nn.Sequential(
            nn.Linear(cfg.fusion_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.time_embedding = nn.Sequential(
            SinusoidalTimestepEmbedding(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.blocks = nn.ModuleList(
            [FastWAMExpertBlock(self.hidden_dim, self.num_heads, cfg.dropout) for _ in range(cfg.fastwam_layers)]
        )

    def pre_dit(self, action_tokens: torch.Tensor, timestep: torch.Tensor, context: torch.Tensor) -> Dict[str, torch.Tensor]:
        if action_tokens.ndim != 3:
            raise ValueError("action_tokens must have shape [B, T, action_dim].")
        return {
            "tokens": self.action_encoder(action_tokens),
            "t_mod": self.time_embedding(timestep),
            "context": self.text_embedding(context),
        }

    def post_dit(self, tokens: torch.Tensor, pre_state: Dict[str, torch.Tensor]) -> torch.Tensor:
        del pre_state
        return self.head(tokens)


class FastWAMMoT(nn.Module):
    """Mixture-of-Transformers shared attention for video/action experts."""

    def __init__(self, video: FastWAMVideoExpert, action: FastWAMActionExpert, cfg: ModelConfig) -> None:
        super().__init__()
        self.video = video
        self.action = action
        self.num_heads = cfg.fastwam_heads
        self.layers = len(video.blocks)
        if len(action.blocks) != self.layers:
            raise ValueError("Video and action experts must have same layer count.")

    def _mixed_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        b, q_len, d = q.shape
        kv_len = k.size(1)
        if v.size(1) != kv_len:
            raise ValueError("k and v sequence lengths must match.")
        if mask.shape != (q_len, kv_len):
            raise ValueError(f"attention mask must have shape [{q_len}, {kv_len}], got {tuple(mask.shape)}.")
        h = self.num_heads
        q = q.view(b, q_len, h, d // h).transpose(1, 2)
        k = k.view(b, kv_len, h, d // h).transpose(1, 2)
        v = v.view(b, kv_len, h, d // h).transpose(1, 2)
        attn_mask = mask.to(device=q.device).unsqueeze(0).unsqueeze(0)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        return out.transpose(1, 2).reshape(b, q_len, d)

    def prefill_video_cache(
        self,
        video_tokens: torch.Tensor,
        video_pre: Dict[str, torch.Tensor],
        video_attention_mask: torch.Tensor,
    ) -> list[Dict[str, torch.Tensor]]:
        """Run the video branch once and cache per-layer K/V for action denoising."""
        if video_tokens.ndim != 3:
            raise ValueError("video_tokens must have shape [B, Sv, D].")
        if video_attention_mask.ndim != 2:
            raise ValueError("video_attention_mask must have shape [Sv, Sv].")
        if video_attention_mask.shape[0] != video_tokens.size(1) or video_attention_mask.shape[1] != video_tokens.size(1):
            raise ValueError("video_attention_mask sequence length must match video_tokens.")

        x = video_tokens
        kv_cache: list[Dict[str, torch.Tensor]] = []
        for i in range(self.layers):
            block = self.video.blocks[i]
            q, k, v, residual, gate_msa, shift_mlp, scale_mlp, gate_mlp = block.attention_io(
                x,
                video_pre["t_mod"],
            )
            mixed = self._mixed_attention(q, k, v, video_attention_mask)
            x = block.post_attention(
                residual,
                mixed,
                gate_msa,
                shift_mlp,
                scale_mlp,
                gate_mlp,
                video_pre.get("context"),
            )
            kv_cache.append({"k": k, "v": v})
        return kv_cache

    def forward_action_with_video_cache(
        self,
        action_tokens: torch.Tensor,
        action_pre: Dict[str, torch.Tensor],
        video_kv_cache: list[Dict[str, torch.Tensor]],
        attention_mask: torch.Tensor,
        video_seq_len: int,
    ) -> torch.Tensor:
        """Run only the action branch while attending to cached video K/V."""
        if action_tokens.ndim != 3:
            raise ValueError("action_tokens must have shape [B, Sa, D].")
        if len(video_kv_cache) != self.layers:
            raise ValueError(f"video_kv_cache must contain {self.layers} layers, got {len(video_kv_cache)}.")
        if attention_mask.ndim != 2:
            raise ValueError("attention_mask must have shape [Sv+Sa, Sv+Sa].")

        action_seq_len = action_tokens.size(1)
        total_seq_len = int(video_seq_len) + action_seq_len
        if attention_mask.shape != (total_seq_len, total_seq_len):
            raise ValueError(
                f"attention_mask shape must be [{total_seq_len}, {total_seq_len}], got {tuple(attention_mask.shape)}."
            )
        action_attention_mask = attention_mask[video_seq_len:total_seq_len, :total_seq_len]

        x = action_tokens
        for i in range(self.layers):
            block = self.action.blocks[i]
            q_action, k_action, v_action, residual, gate_msa, shift_mlp, scale_mlp, gate_mlp = block.attention_io(
                x,
                action_pre["t_mod"],
            )
            layer_cache = video_kv_cache[i]
            if "k" not in layer_cache or "v" not in layer_cache:
                raise ValueError(f"video_kv_cache[{i}] must contain k and v.")
            k_video = layer_cache["k"]
            v_video = layer_cache["v"]
            if k_video.size(1) != video_seq_len or v_video.size(1) != video_seq_len:
                raise ValueError(f"video_kv_cache[{i}] sequence length must be {video_seq_len}.")
            k = torch.cat([k_video, k_action], dim=1)
            v = torch.cat([v_video, v_action], dim=1)
            mixed = self._mixed_attention(q_action, k, v, action_attention_mask)
            x = block.post_attention(
                residual,
                mixed,
                gate_msa,
                shift_mlp,
                scale_mlp,
                gate_mlp,
                action_pre.get("context"),
            )
        return x

    @staticmethod
    def build_attention_mask(video_len: int, action_len: int, tokens_per_frame: int, seq_len: int, device: torch.device) -> torch.Tensor:
        total = video_len + action_len
        mask = torch.zeros(total, total, dtype=torch.bool, device=device)
        mask[:video_len, :video_len] = FastWAMVideoExpert.build_video_to_video_mask(
            seq_len=seq_len,
            tokens_per_frame=tokens_per_frame,
            mode="first_frame_causal",
            device=device,
        )
        mask[video_len:, video_len:] = True
        mask[video_len:, :tokens_per_frame] = True
        return mask

    def forward(
        self,
        video_tokens: torch.Tensor,
        action_tokens: torch.Tensor,
        video_pre: Dict[str, torch.Tensor],
        action_pre: Dict[str, torch.Tensor],
        attention_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        xv = video_tokens
        xa = action_tokens
        for i in range(self.layers):
            vb = self.video.blocks[i]
            ab = self.action.blocks[i]
            v_io = vb.attention_io(xv, video_pre["t_mod"])
            a_io = ab.attention_io(xa, action_pre["t_mod"])
            q = torch.cat([v_io[0], a_io[0]], dim=1)
            k = torch.cat([v_io[1], a_io[1]], dim=1)
            v = torch.cat([v_io[2], a_io[2]], dim=1)
            mixed = self._mixed_attention(q, k, v, attention_mask)
            mv, ma = mixed[:, : xv.size(1)], mixed[:, xv.size(1):]
            xv = vb.post_attention(v_io[3], mv, v_io[4], v_io[5], v_io[6], v_io[7], video_pre.get("context"))
            xa = ab.post_attention(a_io[3], ma, a_io[4], a_io[5], a_io[6], a_io[7], action_pre.get("context"))
        return {"video": xv, "action": xa}


class FastWAMHead(nn.Module):
    """Official FastWAM WanVideoDiT + ActionDiT + MoT head.

    The project keeps its UAV action normalization outside this head, but the
    diffusion/video path follows the official FastWAM latent-space design.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.torch_dtype = _torch_dtype_from_name(getattr(cfg, "wan22_torch_dtype", "bfloat16"))
        self.use_fasterwam_dot = bool(getattr(cfg, "use_fasterwam_dot", False))
        self.video_expert = self._load_official_video_expert(cfg)
        if self.use_fasterwam_dot:
            tracker_mot_integration = str(
                getattr(cfg, "tracker_mot_integration", "none")
            ).strip().lower()
            allows_current_box_tracker = bool(
                getattr(cfg, "use_current_box_action_conditioning", False)
            ) and tracker_mot_integration == "mot_tracker_finetune_local_feature"
            incompatible = {
                "tracker_mot_integration": tracker_mot_integration != "none"
                and not allows_current_box_tracker,
                "use_future_state_dit": bool(getattr(cfg, "use_future_state_dit", False)),
                "tracker_future_target_alignment": bool(
                    getattr(cfg, "tracker_future_target_alignment", False)
                ),
                "use_fastwam_attention_heatmap_loss": bool(
                    getattr(cfg, "use_fastwam_attention_heatmap_loss", False)
                ),
                "use_fastwam_tracker_heatmap_loss": bool(
                    getattr(cfg, "use_fastwam_tracker_heatmap_loss", False)
                ),
                "use_fastwam_attention_bias": bool(
                    getattr(cfg, "use_fastwam_attention_bias", False)
                ),
                "use_gt_center_attention_bias": bool(
                    getattr(cfg, "use_gt_center_attention_bias", False)
                ),
                "use_fastwam_heatmap_guidance": bool(
                    getattr(cfg, "use_fastwam_heatmap_guidance", False)
                ),
                "use_target_relative_context": bool(
                    getattr(cfg, "use_target_relative_context", False)
                ),
                "use_tracker_center_context": bool(
                    getattr(cfg, "use_tracker_center_context", False)
                ),
                "tracker_center_flow_supervision": bool(
                    getattr(cfg, "tracker_center_flow_supervision", False)
                ),
                "train_next_target_relative": bool(
                    getattr(cfg, "train_next_target_relative", False)
                ),
                "x0_action_loss_weight": float(
                    getattr(cfg, "x0_action_loss_weight", 0.0)
                )
                != 0.0,
            }
            enabled = [name for name, value in incompatible.items() if value]
            if enabled:
                raise ValueError(
                    "Faster-WAM DoT baseline cannot be combined with: "
                    + ", ".join(enabled)
                )
            _ensure_fastwam_path(cfg)
            from .fasterwam_dot import FasterWAMActionHead

            self.action_expert = FasterWAMActionHead(
                action_dim=int(cfg.action_dim),
                hidden_dim=1024,
                ffn_dim=4096,
                freq_dim=256,
                num_heads=int(self.video_expert.num_heads),
                attn_head_dim=int(self.video_expert.attn_head_dim),
                eps=1.0e-6,
                use_gradient_checkpointing=bool(
                    getattr(cfg, "fastwam_mot_checkpoint_mixed_attn", True)
                ),
            ).to(dtype=self.torch_dtype)
        else:
            self.action_expert = self._load_official_action_expert(cfg)
        self.state_expert = None
        if bool(getattr(cfg, "use_future_state_dit", False)):
            from .state_dit import FutureStateDiT

            if int(getattr(cfg, "tracker_state_action_alignment_version", 1)) != 4:
                raise ValueError("Future State DiT checkpoints require alignment version 4.")
            if int(getattr(cfg, "future_state_horizon", 8)) != int(
                getattr(cfg, "action_sequence_horizon", 8)
            ):
                raise ValueError("Future State and Action horizons must match.")
            if int(getattr(cfg, "future_state_hidden_dim", 1024)) != int(
                self.action_expert.hidden_dim
            ):
                raise ValueError("Future State hidden dimension must match ActionDiT hidden dimension.")
            self.state_expert = FutureStateDiT(
                state_dim=int(getattr(cfg, "future_state_dim", 4)),
                hidden_dim=int(getattr(cfg, "future_state_hidden_dim", 1024)),
                ffn_dim=int(getattr(cfg, "future_state_ffn_dim", 4096)),
                freq_dim=256,
                num_heads=int(self.action_expert.num_heads),
                attn_head_dim=int(self.action_expert.attn_head_dim),
                num_layers=int(getattr(cfg, "future_state_num_layers", 30)),
                horizon=int(getattr(cfg, "future_state_horizon", 8)),
                eps=1.0e-6,
                use_gradient_checkpointing=bool(
                    getattr(cfg, "fastwam_mot_checkpoint_mixed_attn", True)
                ),
            ).to(dtype=self.torch_dtype)
        self.tracker_integration = str(
            getattr(cfg, "tracker_mot_integration", "none")
        ).strip().lower()
        self.tracker_condition_mode = str(
            getattr(cfg, "tracker_condition_mode", "center_features")
        ).strip().lower()
        self.tracker_spatial_cross_attention = bool(
            getattr(cfg, "tracker_spatial_cross_attention", True)
        )
        valid_tracker_integrations = {
            "none",
            "frozen_deit_tracker_fusion",
            "frozen_deit_tracker_local_feature",
            "mot_tracker_finetune_local_feature",
        }
        if self.tracker_integration not in valid_tracker_integrations:
            raise ValueError(
                "Unsupported tracker_mot_integration; expected one of "
                f"{sorted(valid_tracker_integrations)}."
            )
        if int(self.action_expert.num_heads) != int(self.video_expert.num_heads):
            raise ValueError("Official ActionDiT num_heads must match WanVideoDiT for MoT.")
        if int(self.action_expert.attn_head_dim) != int(self.video_expert.attn_head_dim):
            raise ValueError("Official ActionDiT attn_head_dim must match WanVideoDiT for MoT.")
        self.dot = None
        self.mot = None
        if self.use_fasterwam_dot:
            from .fasterwam_dot import FasterWAMDoT

            self.dot = FasterWAMDoT(
                video_num_layers=len(self.video_expert.blocks),
                num_action_layers=len(self.action_expert.blocks),
                num_heads=int(self.video_expert.num_heads),
                attn_head_dim=int(self.video_expert.attn_head_dim),
                use_gradient_checkpointing=bool(
                    getattr(cfg, "fastwam_mot_checkpoint_mixed_attn", True)
                ),
            ).to(dtype=self.torch_dtype)
        else:
            if int(len(self.action_expert.blocks)) != int(len(self.video_expert.blocks)):
                raise ValueError("Official ActionDiT num_layers must match WanVideoDiT.")
            _ensure_fastwam_path(cfg)
            from fastwam.models.wan22.mot import MoT

            mixtures: Dict[str, nn.Module] = {
                "video": self.video_expert,
                "action": self.action_expert,
            }
            if self.state_expert is not None:
                mixtures["state"] = self.state_expert
            self.mot = MoT(
                mixtures=mixtures,
                mot_checkpoint_mixed_attn=bool(
                    getattr(cfg, "fastwam_mot_checkpoint_mixed_attn", True)
                ),
            )
        tracker_dim = int(getattr(cfg, "tracker_feature_dim", 192))
        self.tracker_fusion = None
        if self.tracker_integration == "frozen_deit_tracker_fusion":
            from .tracker_fusion import FrozenTrackerConditionFusion

            self.tracker_fusion = FrozenTrackerConditionFusion(
                tracker_dim=tracker_dim,
                action_dim=int(self.action_expert.hidden_dim),
                num_heads=int(self.action_expert.num_heads),
                head_dim=int(self.action_expert.attn_head_dim),
                num_layers=len(self.action_expert.blocks),
                start_layer=int(getattr(cfg, "tracker_fusion_start_layer", 18)),
                condition_mode=self.tracker_condition_mode,
                response_grid_size=int(getattr(cfg, "tracker_response_grid_size", 7)),
                gate_init=float(getattr(cfg, "tracker_fusion_gate_init", 0.0)),
            ).to(dtype=self.torch_dtype)
        elif self.tracker_integration in {"frozen_deit_tracker_local_feature", "mot_tracker_finetune_local_feature"}:
            from .tracker_fusion import LocalFeatureTrackerConditionFusion

            tracker_grid_size = int(getattr(cfg, "tracker_feature_grid_size", 16))
            if tracker_grid_size != 16:
                raise ValueError(
                    "local Tracker feature integration requires "
                    "tracker_feature_grid_size=16."
                )
            self.tracker_fusion = LocalFeatureTrackerConditionFusion(
                tracker_dim=tracker_dim,
                action_dim=int(self.action_expert.hidden_dim),
                num_heads=int(self.action_expert.num_heads),
                head_dim=int(self.action_expert.attn_head_dim),
                num_layers=len(self.action_expert.blocks),
                start_layer=int(getattr(cfg, "tracker_fusion_start_layer", 18)),
                grid_size=tracker_grid_size,
                use_local_position_embedding=bool(
                    getattr(cfg, "tracker_use_local_position_embedding", False)
                ),
                include_box_token=bool(getattr(cfg, "tracker_include_box_token", True)),
                gate_init=float(getattr(cfg, "tracker_fusion_gate_init", 0.0)),
                detach_tracker_inputs=(self.tracker_integration != "mot_tracker_finetune_local_feature"),
                enable_cross_attention=self.tracker_spatial_cross_attention,
            ).to(dtype=self.torch_dtype)
        self.future_target_readout = None
        if bool(getattr(cfg, "tracker_future_target_alignment", False)):
            if int(getattr(cfg, "tracker_state_action_alignment_version", 1)) != 3:
                raise ValueError(
                    "Future Target box checkpoints must use state-action alignment version 3 "
                    "(states s0..sH, independent s0-relative box offsets, actions a0..a(H-1))."
                )
            if self.tracker_integration != "mot_tracker_finetune_local_feature":
                raise ValueError("Box-free future target alignment requires the jointly trained local-feature Tracker.")
            if bool(getattr(cfg, "tracker_include_box_token", True)):
                raise ValueError("Box-free future target alignment requires tracker_include_box_token=false.")
            from .tracker_fusion import BoxFreeFutureTargetReadout

            self.future_target_readout = BoxFreeFutureTargetReadout(
                tracker_dim=tracker_dim,
                action_dim=int(self.action_expert.hidden_dim),
                video_dim=int(self.video_expert.hidden_dim),
                num_layers=len(self.action_expert.blocks),
                start_layer=int(getattr(cfg, "tracker_future_target_start_layer", 18)),
                action_horizon=int(getattr(cfg, "action_sequence_horizon", 8)),
            ).to(dtype=self.torch_dtype)
        self.future_state_conditioner = None
        self.state_to_action_gates = None
        if self.state_expert is not None:
            if bool(getattr(cfg, "tracker_future_target_alignment", False)):
                raise ValueError("V4 StateDiT cannot be combined with the old Future Target Readout.")
            if self.tracker_integration != "mot_tracker_finetune_local_feature":
                raise ValueError("Future State DiT requires the jointly trained local-feature Tracker.")
            if bool(getattr(cfg, "tracker_include_box_token", True)):
                raise ValueError("Future State DiT requires tracker_include_box_token=false.")
            from .tracker_fusion import FutureStateConditioner

            self.future_state_conditioner = FutureStateConditioner(
                tracker_dim=tracker_dim,
                state_hidden_dim=int(getattr(cfg, "future_state_hidden_dim", 1024)),
            ).to(dtype=self.torch_dtype)
            self.state_to_action_gates = nn.Parameter(
                torch.zeros(len(self.action_expert.blocks), dtype=self.torch_dtype)
            )
        self.current_target_localizer = None
        self.current_box_action_conditioner = None
        self.target_action_conditioning = None
        self.capture_value_head = None
        self.capture_action_prior = None
        self.capture_value_reranking_enabled = bool(
            getattr(cfg, "use_capture_value_reranking", False)
        )
        self.capture_value_score_mode = str(
            getattr(cfg, "capture_value_score_mode", "learned")
        ).strip().lower()
        if bool(getattr(cfg, "use_current_box_action_conditioning", False)):
            if self.state_expert is not None or self.future_target_readout is not None:
                raise ValueError(
                    "Current-box Action conditioning cannot be combined with future-state heads."
                )
            if self.tracker_integration != "mot_tracker_finetune_local_feature":
                raise ValueError(
                    "Current-box Action conditioning requires the complete pretrained Tracker."
                )
            if not bool(getattr(cfg, "tracker_include_box_token", True)):
                raise ValueError(
                    "Current-box Action conditioning requires tracker_include_box_token=true."
                )
            if self.tracker_spatial_cross_attention:
                raise ValueError(
                    "The Tracker is b0-only; set tracker_spatial_cross_attention=false."
                )
            if bool(getattr(cfg, "use_tracker_memory", True)):
                raise ValueError(
                    "Current-box Action conditioning requires use_tracker_memory=false."
                )
            hidden_dim = int(getattr(cfg, "current_box_action_hidden_dim", 1024))
            if hidden_dim != int(self.action_expert.hidden_dim):
                raise ValueError(
                    "Current-box conditioner hidden dimension must match ActionDiT."
                )
            from .current_box_action_conditioner import CurrentBoxActionConditioner

            self.tracker_fusion = None
            self.current_box_action_conditioner = CurrentBoxActionConditioner(
                hidden_dim=hidden_dim,
                layers=getattr(cfg, "current_box_action_layers", (18, 23, 26, 29)),
                gate_init=float(getattr(cfg, "current_box_action_gate_init", 0.0)),
            ).to(dtype=self.torch_dtype)
            if bool(getattr(cfg, "freeze_current_box_action_conditioner", False)):
                for parameter in self.current_box_action_conditioner.parameters():
                    parameter.requires_grad_(False)
            if max(self.current_box_action_conditioner.layers) >= len(
                self.action_expert.blocks
            ):
                raise ValueError(
                    "Current-box Action conditioning layer exceeds ActionDiT depth."
                )
        use_history_memory = bool(
            getattr(cfg, "use_historical_target_memory", False)
        )
        if use_history_memory:
            if self.current_box_action_conditioner is None:
                raise ValueError("Historical target conditioning requires Current Box Action.")
            if int(getattr(cfg, "target_history_length", 8)) < 2:
                raise ValueError("Historical Target Memory requires at least two states.")
            from .target_action_conditioning import TargetActionConditioning

            self.target_action_conditioning = TargetActionConditioning(
                action_hidden_dim=int(self.action_expert.hidden_dim),
                history_length=int(getattr(cfg, "target_history_length", 8)),
                horizon=int(getattr(cfg, "action_sequence_horizon", 8)),
                memory_dim=int(getattr(cfg, "target_history_hidden_dim", 256)),
                memory_layers=int(getattr(cfg, "target_history_num_layers", 2)),
                memory_heads=int(getattr(cfg, "target_history_num_heads", 8)),
            ).to(dtype=self.torch_dtype)
        if self.capture_value_reranking_enabled:
            if self.current_box_action_conditioner is None:
                raise ValueError(
                    "Capture-value reranking requires Current Box Action conditioning."
                )
            candidate_count = int(
                getattr(cfg, "capture_value_candidate_count", 4)
            )
            if candidate_count < 2:
                raise ValueError("Capture-value reranking requires at least 2 candidates.")
            if self.capture_value_score_mode not in {
                "learned",
                "geometric",
                "action_prior",
            }:
                raise ValueError(
                    "capture_value_score_mode must be learned, geometric, or action_prior."
                )
            if self.capture_value_score_mode == "learned":
                if self.dot is None:
                    raise ValueError(
                        "Learned Capture-Value scoring requires FasterWAM DoT video context; "
                        "FastWAM supports action_prior or geometric scoring."
                    )
                from .capture_value_reranker import CaptureValueHead

                self.capture_value_head = CaptureValueHead(
                    video_context_dim=2 * int(self.dot.attention_dim),
                    target_context_dim=int(self.action_expert.hidden_dim),
                    action_dim=int(cfg.action_dim),
                    horizon=int(getattr(cfg, "action_sequence_horizon", 8)),
                    hidden_dim=int(getattr(cfg, "capture_value_hidden_dim", 256)),
                    num_layers=int(getattr(cfg, "capture_value_num_layers", 2)),
                    num_heads=int(getattr(cfg, "capture_value_num_heads", 8)),
                    distance_score_weight=float(
                        getattr(cfg, "capture_value_distance_score_weight", 1.0)
                    ),
                    visibility_score_weight=float(
                        getattr(cfg, "capture_value_visibility_score_weight", 0.25)
                    ),
                ).to(dtype=self.torch_dtype)
            elif self.capture_value_score_mode == "action_prior":
                checkpoint_path = str(
                    getattr(cfg, "capture_action_prior_checkpoint", "")
                ).strip()
                if not checkpoint_path:
                    raise ValueError(
                        "action_prior scoring requires capture_action_prior_checkpoint."
                    )
                from .capture_action_prior import load_capture_action_prior

                self.capture_action_prior = load_capture_action_prior(
                    checkpoint_path,
                    device=torch.device("cpu"),
                    dtype=self.torch_dtype,
                )
        if self.tracker_fusion is not None and self.tracker_spatial_cross_attention:
            for block in self.action_expert.blocks[: self.tracker_fusion.start_layer]:
                for parameter in block.parameters():
                    parameter.requires_grad_(False)
        self.video_scheduler = FlowMatchScheduler(cfg.fastwam_video_train_timesteps, cfg.fastwam_video_shift)
        self.action_scheduler = FlowMatchScheduler(cfg.fastwam_action_train_timesteps, cfg.fastwam_action_shift)
        self.state_scheduler = (
            FlowMatchScheduler(
                cfg.fastwam_action_train_timesteps, cfg.fastwam_action_shift
            )
            if self.state_expert is not None
            else None
        )
        object.__setattr__(self, "_compiled_action_sampler", None)
        self._action_compile_failed = False

    def _video_dit_config(self, cfg: ModelConfig) -> Dict[str, Any]:
        return {
            "has_image_input": False,
            "patch_size": (1, 2, 2),
            "in_dim": 48,
            "hidden_dim": 3072,
            "ffn_dim": 14336,
            "freq_dim": 256,
            "text_dim": int(getattr(cfg, "text_width", 4096)),
            "out_dim": 48,
            "num_heads": 24,
            "attn_head_dim": 128,
            "num_layers": 30,
            "eps": 1.0e-6,
            "seperated_timestep": True,
            "require_clip_embedding": False,
            "require_vae_embedding": False,
            "fuse_vae_embedding_in_latents": True,
            "use_gradient_checkpointing": bool(getattr(cfg, "fastwam_mot_checkpoint_mixed_attn", True)),
            "video_attention_mask_mode": "first_frame_causal",
            "action_conditioned": False,
            "action_dim": int(cfg.action_dim),
            "action_group_causal_mask_mode": "group_diagonal",
        }

    def _action_dit_config(self, cfg: ModelConfig) -> Dict[str, Any]:
        return {
            "action_dim": int(cfg.action_dim),
            "hidden_dim": 1024,
            "ffn_dim": 4096,
            "num_heads": 24,
            "attn_head_dim": 128,
            "num_layers": 30,
            "text_dim": int(getattr(cfg, "text_width", 4096)),
            "freq_dim": 256,
            "eps": 1.0e-6,
            "use_gradient_checkpointing": bool(getattr(cfg, "fastwam_mot_checkpoint_mixed_attn", True)),
        }

    def _load_official_video_expert(self, cfg: ModelConfig) -> nn.Module:
        _ensure_fastwam_path(cfg)
        from fastwam.models.wan22.helpers.loader import _load_registered_model, _resolve_configs
        from fastwam.models.wan22.wan_video_dit import WanVideoDiT

        dit_config = self._video_dit_config(cfg)
        device_s = "cpu"
        if bool(getattr(cfg, "fastwam_skip_dit_load_from_pretrain", False)):
            return WanVideoDiT(**dit_config).to(device=device_s, dtype=self.torch_dtype)

        old_base = os.environ.get("DIFFSYNTH_MODEL_BASE_PATH")
        old_skip = os.environ.get("DIFFSYNTH_SKIP_DOWNLOAD")
        os.environ["DIFFSYNTH_MODEL_BASE_PATH"] = str(cfg.wan22_model_base_path)
        os.environ["DIFFSYNTH_SKIP_DOWNLOAD"] = "true" if cfg.wan22_skip_download else "false"
        try:
            dit_model_config, _, _, _ = _resolve_configs(
                model_id=cfg.wan22_model_id,
                tokenizer_model_id=cfg.wan22_tokenizer_model_id,
                redirect_common_files=bool(cfg.wan22_redirect_common_files),
            )
            dit_model_config.skip_download = bool(cfg.wan22_skip_download)
            dit_model_config.download_if_necessary()
            if isinstance(dit_model_config.path, list) and len(dit_model_config.path) == 0:
                raise FileNotFoundError(
                    "Official WanVideoDiT weights were not found locally. "
                    f"Expected diffusion_pytorch_model*.safetensors under "
                    f"{os.path.join(str(cfg.wan22_model_base_path), str(cfg.wan22_model_id))}. "
                    "Set WAN22_SKIP_DOWNLOAD=false to download them, or set "
                    "FASTWAM_SKIP_DIT_LOAD_FROM_PRETRAIN=true to use a randomly initialized WanVideoDiT."
                )
            return _load_registered_model(
                dit_model_config.path,
                "wan_video_dit",
                torch_dtype=self.torch_dtype,
                device=device_s,
                model_kwargs_override=dit_config,
            )
        finally:
            if old_base is None:
                os.environ.pop("DIFFSYNTH_MODEL_BASE_PATH", None)
            else:
                os.environ["DIFFSYNTH_MODEL_BASE_PATH"] = old_base
            if old_skip is None:
                os.environ.pop("DIFFSYNTH_SKIP_DOWNLOAD", None)
            else:
                os.environ["DIFFSYNTH_SKIP_DOWNLOAD"] = old_skip

    def _load_official_action_expert(self, cfg: ModelConfig) -> nn.Module:
        _ensure_fastwam_path(cfg)
        from fastwam.models.wan22.action_dit import ActionDiT

        path = str(getattr(cfg, "fastwam_action_dit_pretrained_path", "") or "")
        return ActionDiT.from_pretrained(
            action_dit_config=self._action_dit_config(cfg),
            action_dit_pretrained_path=path or None,
            skip_dit_load_from_pretrain=bool(getattr(cfg, "fastwam_skip_dit_load_from_pretrain", False)),
            device="cpu",
            torch_dtype=self.torch_dtype,
        )

    def _make_training_action_tokens(
        self,
        expert_action: torch.Tensor,
        valid_mask: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        if expert_action.ndim != 3:
            raise ValueError("expert_action must have shape [B, T, A].")
        if expert_action.size(1) <= 1:
            raise ValueError("FastWAM training requires at least 2 action timesteps.")
        action = expert_action[:, :-1]
        action_mask = None
        if valid_mask is not None:
            if valid_mask.shape[:2] != expert_action.shape[:2]:
                raise ValueError("valid_mask must have shape [B, T] matching expert_action.")
            action_mask = valid_mask[:, :-1]
            if valid_mask.size(1) > 1:
                action_mask = action_mask * valid_mask[:, 1:]
        return action, action_mask

    def _prepare_tracker_features(
        self,
        tracker_features: Optional[torch.Tensor],
        tracker_confidence: Optional[torch.Tensor],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        if self.tracker_integration == "none" or (
            self.tracker_integration == "frozen_deit_tracker_fusion"
            and "features" not in self.tracker_condition_mode
        ):
            return None
        if tracker_features is None:
            raise RuntimeError(
                f"tracker_mot_integration={self.tracker_integration!r} requires Tracker features."
            )
        expected_tokens = max(int(getattr(self.cfg, "tracker_feature_grid_size", 7)), 1) ** 2
        expected_dim = int(getattr(self.cfg, "tracker_feature_dim", 192))
        if tracker_features.ndim != 3 or tuple(tracker_features.shape[1:]) != (
            expected_tokens,
            expected_dim,
        ):
            raise ValueError(
                f"Expected Tracker features [B,{expected_tokens},{expected_dim}], "
                f"got {tuple(tracker_features.shape)}."
            )
        features = tracker_features.to(device=device, dtype=dtype)
        return features * float(getattr(self.cfg, "tracker_feature_token_scale", 1.0))

    def _make_tracker_condition(
        self,
        tracker_features: Optional[torch.Tensor],
        tracker_center: Optional[torch.Tensor],
        tracker_bbox: Optional[torch.Tensor],
        tracker_response: Optional[torch.Tensor],
        tracker_search_geometry: Optional[torch.Tensor],
        tracker_image_size: Optional[torch.Tensor],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        if self.current_box_action_conditioner is not None and bool(
            getattr(self.cfg, "tracker_include_box_token", True)
        ):
            return None
        if self.tracker_integration == "none":
            return None
        if (
            self.tracker_integration == "frozen_deit_tracker_fusion"
            and self.tracker_condition_mode == "none"
        ):
            return None
        prepared_features = self._prepare_tracker_features(
            tracker_features,
            None,
            device=device,
            dtype=dtype,
        )
        assert self.tracker_fusion is not None
        if self.tracker_integration in {"frozen_deit_tracker_local_feature", "mot_tracker_finetune_local_feature"}:
            if bool(getattr(self.cfg, "tracker_include_box_token", True)) and tracker_bbox is None:
                raise RuntimeError("Tracker local-feature fusion requires tracker_bbox.")
            if tracker_search_geometry is None:
                raise RuntimeError(
                    "Tracker local-feature fusion requires tracker_search_geometry."
                )
            if tracker_image_size is None:
                raise RuntimeError("Tracker local-feature fusion requires tracker_image_size.")
            return self.tracker_fusion.make_condition(
                tracker_features=prepared_features,
                tracker_bbox=tracker_bbox,
                tracker_search_geometry=tracker_search_geometry,
                tracker_image_size=tracker_image_size,
            )
        return self.tracker_fusion.make_condition(
            tracker_features=prepared_features,
            tracker_center=tracker_center,
            tracker_bbox=tracker_bbox,
            tracker_response=tracker_response,
        )

    def _make_future_target_state(
        self,
        tracker_template_features: Optional[torch.Tensor],
        tracker_condition: Optional[torch.Tensor],
        tracker_search_geometry: Optional[torch.Tensor],
        tracker_image_size: Optional[torch.Tensor],
    ) -> Optional[dict[str, torch.Tensor]]:
        if self.future_target_readout is None:
            return None
        if (
            tracker_template_features is None or tracker_condition is None
            or tracker_search_geometry is None or tracker_image_size is None
        ):
            raise RuntimeError("Box-free future target alignment requires current Template features, Search geometry, and image size.")
        if tracker_condition.size(1) != 256:
            raise ValueError("Box-free future target alignment requires exactly 256 spatial Tracker tokens.")
        assert self.tracker_fusion is not None
        full_xy = self.tracker_fusion.full_image_coordinates(
            tracker_search_geometry.to(device=tracker_condition.device),
            tracker_image_size.to(device=tracker_condition.device),
        ).to(dtype=tracker_condition.dtype)
        return self.future_target_readout.make_target_state(
            tracker_template_features.to(device=tracker_condition.device, dtype=tracker_condition.dtype),
            tracker_condition,
            full_xy,
        )

    def _make_future_state_condition(
        self,
        tracker_template_features: Optional[torch.Tensor],
        tracker_condition: Optional[torch.Tensor],
        tracker_search_geometry: Optional[torch.Tensor],
        tracker_image_size: Optional[torch.Tensor],
    ) -> Optional[dict[str, torch.Tensor]]:
        if self.future_state_conditioner is None:
            return None
        if (
            tracker_template_features is None
            or tracker_condition is None
            or tracker_search_geometry is None
            or tracker_image_size is None
        ):
            raise RuntimeError(
                "Future State DiT requires Template/Search features, Search geometry, and image size."
            )
        if tracker_condition.size(1) != 256:
            raise ValueError("Future State DiT requires exactly 256 spatial Search tokens.")
        assert self.tracker_fusion is not None
        full_xy = self.tracker_fusion.full_image_coordinates(
            tracker_search_geometry.to(device=tracker_condition.device),
            tracker_image_size.to(device=tracker_condition.device),
        ).to(dtype=tracker_condition.dtype)
        return self.future_state_conditioner(
            tracker_template_features.to(
                device=tracker_condition.device, dtype=tracker_condition.dtype
            ),
            tracker_condition,
            full_xy,
        )

    def _make_current_target_localization(
        self,
        tracker_template_features: Optional[torch.Tensor],
        tracker_condition: Optional[torch.Tensor],
        tracker_search_geometry: Optional[torch.Tensor],
        tracker_image_size: Optional[torch.Tensor],
        tracker_bbox: Optional[torch.Tensor] = None,
    ) -> Optional[dict[str, torch.Tensor]]:
        if self.current_box_action_conditioner is not None and bool(
            getattr(self.cfg, "tracker_include_box_token", True)
        ):
            if tracker_bbox is None:
                raise RuntimeError(
                    "The pretrained Tracker head did not provide its current bbox."
                )
            if tracker_bbox.ndim != 2 or tracker_bbox.size(-1) != 4:
                raise ValueError("Tracker-head current bbox must be [B,4] normalized cxcywh.")
            return {"current_box": tracker_bbox.float().clamp(0.0, 1.0)}
        if self.current_target_localizer is None:
            return None
        if (
            tracker_template_features is None
            or tracker_condition is None
            or tracker_search_geometry is None
            or tracker_image_size is None
        ):
            raise RuntimeError(
                "Current target localization requires Template/Search features, Search geometry, and image size."
            )
        if tracker_condition.size(1) != 256:
            raise ValueError(
                "The b0 Target Query requires exactly 256 Search spatial tokens."
            )
        assert self.tracker_fusion is not None
        full_xy = self.tracker_fusion.full_image_coordinates(
            tracker_search_geometry.to(device=tracker_condition.device),
            tracker_image_size.to(device=tracker_condition.device),
        ).to(dtype=tracker_condition.dtype)
        return self.current_target_localizer(
            tracker_template_features.to(
                device=tracker_condition.device, dtype=tracker_condition.dtype
            ),
            tracker_condition,
            full_xy,
        )

    @staticmethod
    def _state_validity(
        target_valid: torch.Tensor,
        sequence_valid: Optional[torch.Tensor],
        horizon: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        valid = target_valid
        if valid.ndim == 3:
            valid = valid.squeeze(-1)
        if valid.ndim != 2 or valid.size(1) < horizon + 1:
            raise ValueError(f"Target validity must cover {horizon + 1} states.")
        valid = valid[:, : horizon + 1].bool()
        if sequence_valid is not None:
            if sequence_valid.ndim != 2 or sequence_valid.size(1) < horizon + 1:
                raise ValueError(f"Sequence validity must cover {horizon + 1} states.")
            valid = valid & sequence_valid[:, : horizon + 1].bool()
        future_valid = valid[:, 1:] & valid[:, :1]
        state_valid = torch.cat([valid[:, :1], future_valid], dim=1)
        return state_valid, future_valid

    def _forward_three_expert_layers(
        self,
        *,
        video_tokens: Optional[torch.Tensor],
        video_pre: Dict[str, Any],
        action_pre: Dict[str, Any],
        state_pre: Dict[str, Any],
        base_attention_mask: torch.Tensor,
        action_valid_mask: Optional[torch.Tensor],
        video_kv_cache: Optional[list[dict[str, torch.Tensor]]] = None,
    ) -> dict[str, torch.Tensor]:
        """Directional Video/Action/State MoT with zero-gated State-to-Action residuals."""
        if self.state_expert is None or self.state_to_action_gates is None:
            raise RuntimeError("Three-expert execution requires Future State DiT.")
        xa = action_pre["tokens"]
        xs = state_pre["tokens"]
        xv = video_tokens
        batch = xa.size(0)
        video_len = int(
            video_pre["tokens"].size(1) if video_kv_cache is not None else xv.size(1)
        )
        action_len = int(xa.size(1))
        state_len = int(xs.size(1))
        if base_attention_mask.shape != (video_len + action_len, video_len + action_len):
            raise ValueError("Base Video/Action attention mask has the wrong shape.")
        directional_base_mask = base_attention_mask
        if video_kv_cache is None:
            directional_base_mask = base_attention_mask.clone()
            directional_base_mask[:video_len, video_len:] = True
        state_valid = state_pre["state_valid_mask"].to(device=xa.device, dtype=torch.bool)
        if state_valid.shape != (batch, state_len):
            raise ValueError("State valid mask shape does not match StateDiT tokens.")
        if action_valid_mask is None:
            action_valid = torch.ones(batch, action_len, device=xa.device, dtype=torch.bool)
        else:
            action_valid = action_valid_mask[:, :action_len].to(device=xa.device, dtype=torch.bool)

        action_context = {"context": action_pre["context"], "mask": action_pre["context_mask"]}
        video_context = {"context": video_pre["context"], "mask": video_pre["context_mask"]}
        state_context = {"context": state_pre["context"], "mask": state_pre["context_mask"]}
        for layer_idx in range(int(self.mot.num_layers)):
            action_block = self.action_expert.blocks[layer_idx]
            state_block = self.state_expert.blocks[layer_idx]
            a_io = self.mot._build_expert_attention_io(
                self.action_expert, action_block, xa, action_pre["freqs"], action_pre["t_mod"]
            )
            s_io = self.mot._build_expert_attention_io(
                self.state_expert, state_block, xs, state_pre["freqs"], state_pre["t_mod"]
            )
            if video_kv_cache is None:
                assert xv is not None
                video_block = self.video_expert.blocks[layer_idx]
                v_io = self.mot._build_expert_attention_io(
                    self.video_expert, video_block, xv, video_pre["freqs"], video_pre["t_mod"]
                )
                k_video, v_video = v_io[1], v_io[2]
                q_va = torch.cat([v_io[0], a_io[0]], dim=1)
            else:
                v_io = None
                k_video = video_kv_cache[layer_idx]["k"]
                v_video = video_kv_cache[layer_idx]["v"]
                q_va = a_io[0]

            k_va = torch.cat([k_video, a_io[1]], dim=1)
            v_va = torch.cat([v_video, a_io[2]], dim=1)
            va_key_valid = torch.cat(
                [
                    torch.ones(batch, video_len, device=xa.device, dtype=torch.bool),
                    action_valid,
                ],
                dim=1,
            )
            if video_kv_cache is None:
                base_query_valid = torch.cat(
                    [
                        torch.ones(batch, video_len, device=xa.device, dtype=torch.bool),
                        action_valid,
                    ],
                    dim=1,
                )
                base_mask = (
                    directional_base_mask[None, None]
                    & va_key_valid[:, None, None, :]
                    & base_query_valid[:, None, :, None]
                )
            else:
                action_rows = directional_base_mask[video_len:, :]
                base_mask = (
                    action_rows[None, None]
                    & va_key_valid[:, None, None, :]
                    & action_valid[:, None, :, None]
                )
            mixed_va = self.mot._mixed_attention(q_va, k_va, v_va, base_mask)

            state_key_mask = (
                state_valid[:, None, None, :]
                & action_valid[:, None, :, None]
            )
            mixed_state_to_action = self.mot._mixed_attention(
                a_io[0], s_io[1], s_io[2], state_key_mask
            )

            state_k = torch.cat([k_va, s_io[1]], dim=1)
            state_v = torch.cat([v_va, s_io[2]], dim=1)
            state_base_row = directional_base_mask[video_len : video_len + 1, :].expand(
                state_len, -1
            )
            state_direction_mask = torch.cat(
                [state_base_row, torch.ones(state_len, state_len, device=xa.device, dtype=torch.bool)],
                dim=1,
            )
            state_key_valid = torch.cat([va_key_valid, state_valid], dim=1)
            state_mask = (
                state_direction_mask[None, None]
                & state_key_valid[:, None, None, :]
                & state_valid[:, None, :, None]
            )
            mixed_state = self.mot._mixed_attention(
                s_io[0], state_k, state_v, state_mask
            )

            if video_kv_cache is None:
                assert v_io is not None and xv is not None
                xv = self.mot._apply_post_with_optional_checkpoint(
                    self.video_expert.blocks[layer_idx],
                    v_io[3], v_io[4], v_io[5], v_io[6], v_io[7], v_io[8],
                    mixed_va[:, :video_len], video_context,
                )
                mixed_action = mixed_va[:, video_len:]
            else:
                mixed_action = mixed_va
            xa = self.mot._apply_post_with_optional_checkpoint(
                action_block,
                a_io[3], a_io[4], a_io[5], a_io[6], a_io[7], a_io[8],
                mixed_action, action_context,
            )
            state_delta = action_block.self_attn.o(mixed_state_to_action)
            xa = xa + torch.tanh(self.state_to_action_gates[layer_idx]) * state_delta
            xs = self.mot._apply_post_with_optional_checkpoint(
                state_block,
                s_io[3], s_io[4], s_io[5], s_io[6], s_io[7], s_io[8],
                mixed_state, state_context,
            )
            if not torch.isfinite(xa).all() or not torch.isfinite(xs).all():
                raise FloatingPointError("Three-expert MoT produced NaN/Inf hidden states.")
        result = {"action": xa, "state": xs}
        if xv is not None:
            result["video"] = xv
        return result

    @staticmethod
    def _video_token_coordinates(video_pre: Dict[str, Any], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        frames, height, width = (int(value) for value in video_pre["meta"]["grid_size"])
        y = (torch.arange(height, device=device, dtype=dtype) + 0.5) / float(height)
        x = (torch.arange(width, device=device, dtype=dtype) + 0.5) / float(width)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        return torch.stack([xx, yy], dim=-1).reshape(1, height * width, 2).repeat(1, frames, 1)

    def _forward_training_with_tracker_fusion(
        self,
        video_pre: Dict[str, Any],
        action_pre: Dict[str, Any],
        tracker_condition: torch.Tensor,
        attention_mask: torch.Tensor,
        tracker_template_features: Optional[torch.Tensor] = None,
        tracker_search_geometry: Optional[torch.Tensor] = None,
        tracker_image_size: Optional[torch.Tensor] = None,
        action_timestep: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        xv = video_pre["tokens"]
        xa = action_pre["tokens"]
        video_len = int(xv.size(1))
        video_context = {"context": video_pre["context"], "mask": video_pre["context_mask"]}
        action_context = {"context": action_pre["context"], "mask": action_pre["context_mask"]}
        future_target_state = self._make_future_target_state(
            tracker_template_features, tracker_condition, tracker_search_geometry, tracker_image_size
        )
        # Online policy inference only has the current RGB frame. Future Target
        # Readout must therefore read the current-frame visual memory only,
        # even though video flow training carries future latent frames.
        current_video_tokens = int(video_pre["meta"]["tokens_per_frame"])
        final_future_target_tokens = None
        for layer_idx in range(int(self.mot.num_layers)):
            video_block = self.video_expert.blocks[layer_idx]
            action_block = self.action_expert.blocks[layer_idx]
            v_io = self.mot._build_expert_attention_io(
                self.video_expert, video_block, xv, video_pre["freqs"], video_pre["t_mod"]
            )
            a_io = self.mot._build_expert_attention_io(
                self.action_expert, action_block, xa, action_pre["freqs"], action_pre["t_mod"]
            )
            q = torch.cat([v_io[0], a_io[0]], dim=1)
            k = torch.cat([v_io[1], a_io[1]], dim=1)
            v = torch.cat([v_io[2], a_io[2]], dim=1)
            mixed = self.mot._mixed_attention(q, k, v, attention_mask)
            assert self.tracker_fusion is not None
            tracker_delta = (
                self.tracker_fusion.delta(layer_idx, xa, tracker_condition)
                if self.tracker_spatial_cross_attention
                else torch.zeros_like(xa)
            )
            xv = self.mot._apply_post_with_optional_checkpoint(
                video_block, v_io[3], v_io[4], v_io[5], v_io[6], v_io[7], v_io[8],
                mixed[:, :video_len], video_context,
            )
            future_delta = torch.zeros_like(xa)
            if future_target_state is not None:
                assert self.future_target_readout is not None and action_timestep is not None
                if layer_idx == int(self.mot.num_layers) - 1:
                    future_delta, final_future_target_tokens = self.future_target_readout.delta(
                        layer_idx,
                        xa,
                        xv[:, :current_video_tokens],
                        future_target_state,
                        action_timestep,
                        return_tokens=True,
                    )
                else:
                    future_delta = self.future_target_readout.delta(
                        layer_idx,
                        xa,
                        xv[:, :current_video_tokens],
                        future_target_state,
                        action_timestep,
                    )
            xa = self._apply_action_post_with_tracker_delta(
                action_block, a_io[3], a_io[4], a_io[5], a_io[6], a_io[7], a_io[8],
                mixed[:, video_len:], tracker_delta, future_delta, action_context,
            )
        result = {"video": xv, "action": xa}
        if final_future_target_tokens is not None:
            result["final_future_target_tokens"] = final_future_target_tokens
        if future_target_state is not None:
            result["current_soft_center"] = future_target_state["soft_center"]
            result["current_target_size"] = future_target_state["current_size"]
        return result

    @staticmethod
    def _mot_attention_context_without_ffn(
        block: nn.Module,
        attention_io: tuple[torch.Tensor, ...],
        mixed: torch.Tensor,
        context_payload: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Apply the original attention/cross-attention, leaving FFN for later."""
        attention_output = block.self_attn.o(mixed)
        hidden = block.gate(attention_io[3], attention_io[4], attention_output)
        context = context_payload.get("context")
        if context is not None:
            context_mask = context_payload.get("mask")
            if context_mask is not None and context_mask.dim() == 3:
                context_mask = context_mask.unsqueeze(1)
            hidden = hidden + block.cross_attn(
                block.norm3(hidden), context, ctx_mask=context_mask
            )
        return hidden

    @staticmethod
    def _mot_ffn(
        block: nn.Module,
        hidden: torch.Tensor,
        attention_io: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        from fastwam.models.wan22.wan_video_dit import modulate

        mlp_input = modulate(block.norm2(hidden), attention_io[5], attention_io[6])
        return block.gate(hidden, attention_io[7], block.ffn(mlp_input))

    def _forward_video_action_with_current_box(
        self,
        video_pre: Dict[str, Any],
        action_pre: Dict[str, Any],
        attention_mask: torch.Tensor,
        current_target: Dict[str, torch.Tensor],
        target_conditions: Optional[Dict[str, Optional[torch.Tensor]]] = None,
    ) -> Dict[str, Any]:
        """Run Video/Action MoT with the observed b0 residual."""
        if self.current_box_action_conditioner is None:
            raise RuntimeError("Current-box Action conditioning is not enabled.")
        if self.mot is None:
            raise RuntimeError("FastWAM MoT is not initialized.")

        video_hidden = video_pre["tokens"]
        action_hidden = action_pre["tokens"]
        video_length = int(video_hidden.size(1))
        box_feature = self.current_box_action_conditioner.encode_box(
            current_target["current_box"].to(
                device=action_hidden.device, dtype=action_hidden.dtype
            )
        )
        video_context = {
            "context": video_pre["context"],
            "mask": video_pre["context_mask"],
        }
        action_context = {
            "context": action_pre["context"],
            "mask": action_pre["context_mask"],
        }

        for layer_idx in range(int(self.mot.num_layers)):
            video_block = self.video_expert.blocks[layer_idx]
            action_block = self.action_expert.blocks[layer_idx]
            video_io = self.mot._build_expert_attention_io(
                self.video_expert,
                video_block,
                video_hidden,
                video_pre["freqs"],
                video_pre["t_mod"],
            )
            action_io = self.mot._build_expert_attention_io(
                self.action_expert,
                action_block,
                action_hidden,
                action_pre["freqs"],
                action_pre["t_mod"],
            )
            mixed = self.mot._mixed_attention(
                torch.cat([video_io[0], action_io[0]], dim=1),
                torch.cat([video_io[1], action_io[1]], dim=1),
                torch.cat([video_io[2], action_io[2]], dim=1),
                attention_mask,
            )
            video_hidden = self.mot._apply_post_with_optional_checkpoint(
                block=video_block,
                residual_x=video_io[3],
                gate_msa=video_io[4],
                shift_mlp=video_io[5],
                scale_mlp=video_io[6],
                gate_mlp=video_io[7],
                use_gradient_checkpointing=video_io[8],
                mixed_slice=mixed[:, :video_length],
                context_payload=video_context,
            )
            action_mixed = mixed[:, video_length:]
            if self.current_box_action_conditioner.enabled_at(layer_idx):
                action_mid = self._mot_attention_context_without_ffn(
                    action_block, action_io, action_mixed, action_context
                )
                action_mid = action_mid + self.current_box_action_conditioner.delta(
                    layer_idx, action_mid, box_feature
                )
                if self.target_action_conditioning is not None:
                    if target_conditions is None:
                        raise RuntimeError(
                            "Historical Target Memory conditions were not constructed."
                        )
                    action_mid = action_mid + self.target_action_conditioning.residual(
                        action_mid, target_conditions
                    )
                action_hidden = self._mot_ffn(action_block, action_mid, action_io)
            else:
                action_hidden = self.mot._apply_post_with_optional_checkpoint(
                    block=action_block,
                    residual_x=action_io[3],
                    gate_msa=action_io[4],
                    shift_mlp=action_io[5],
                    scale_mlp=action_io[6],
                    gate_mlp=action_io[7],
                    use_gradient_checkpointing=action_io[8],
                    mixed_slice=action_mixed,
                    context_payload=action_context,
                )
            if not torch.isfinite(video_hidden).all() or not torch.isfinite(
                action_hidden
            ).all():
                raise FloatingPointError(
                    "Video/Action/current-box MoT produced NaN/Inf hidden states."
                )

        return {
            "video": video_hidden,
            "action": action_hidden,
            "current_target": current_target,
        }

    def _apply_action_post_with_tracker_delta(
        self,
        block: nn.Module,
        residual_x: torch.Tensor,
        gate_msa: torch.Tensor,
        shift_mlp: torch.Tensor,
        scale_mlp: torch.Tensor,
        gate_mlp: torch.Tensor,
        use_gradient_checkpointing: bool,
        mixed_slice: torch.Tensor,
        tracker_delta: torch.Tensor,
        future_delta: Optional[torch.Tensor],
        context_payload: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Apply Tracker and Future-Target residuals in the Action MoT block.

        Tracker conditioning is part of the self-attention update. The Future
        Target residual is applied immediately after that original MoT residual
        and before the unchanged text cross-attention and FFN stages.
        """
        from fastwam.models.wan22.wan_video_dit import modulate

        def _post(
            mixed: torch.Tensor,
            residual: torch.Tensor,
            msa_gate: torch.Tensor,
            mlp_shift: torch.Tensor,
            mlp_scale: torch.Tensor,
            mlp_gate: torch.Tensor,
            tracker_residual: torch.Tensor,
            future_residual: torch.Tensor,
        ) -> torch.Tensor:
            attention_output = block.self_attn.o(mixed) + tracker_residual
            x = block.gate(residual, msa_gate, attention_output)
            x = x + future_residual
            context = context_payload.get("context")
            if context is not None:
                context_mask = context_payload.get("mask")
                if context_mask is not None and context_mask.dim() == 3:
                    context_mask = context_mask.unsqueeze(1)
                x = x + block.cross_attn(block.norm3(x), context, ctx_mask=context_mask)
            mlp_input = modulate(block.norm2(x), mlp_shift, mlp_scale)
            return block.gate(x, mlp_gate, block.ffn(mlp_input))

        args = (
            mixed_slice, residual_x, gate_msa, shift_mlp, scale_mlp, gate_mlp,
            tracker_delta, torch.zeros_like(tracker_delta) if future_delta is None else future_delta,
        )
        if use_gradient_checkpointing and self.training:
            return torch.utils.checkpoint.checkpoint(_post, *args, use_reentrant=False)
        return _post(*args)

    def _prepare_guidance_distribution(
        self,
        heatmap: torch.Tensor,
        confidence: Optional[torch.Tensor],
        grid_hw: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if heatmap is None:
            raise RuntimeError("ORTrack-guided FastWAM requires cached or online heatmaps.")
        if heatmap.ndim == 3:
            heatmap = heatmap.unsqueeze(1)
        if heatmap.ndim != 4:
            raise ValueError("guidance_heatmap must have shape [B,H,W] or [B,T,H,W].")
        first = heatmap[:, :1].float()
        if str(getattr(self.cfg, "tracker_heatmap_target_mode", "canonical")) == "raw_area":
            first = F.interpolate(first, size=grid_hw, mode="area").squeeze(1)
        else:
            first = F.interpolate(
                first, size=grid_hw, mode="bilinear", align_corners=False
            ).squeeze(1)
        dist = first.flatten(1).clamp_min(1.0e-8)
        dist = dist / dist.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)
        if confidence is None:
            conf = torch.ones(dist.size(0), device=dist.device, dtype=dist.dtype)
        else:
            conf = confidence.float().reshape(confidence.size(0), -1)[:, 0].to(dist.device).clamp(0.0, 1.0)
        return dist, conf

    def _biased_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: torch.Tensor,
        first_frame_distribution: torch.Tensor,
        confidence: torch.Tensor,
        first_frame_tokens: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, query_len, hidden = q.shape
        key_len = k.size(1)
        heads = int(self.mot.num_heads)
        head_dim = hidden // max(heads, 1)
        qh = q.reshape(bsz, query_len, heads, head_dim).transpose(1, 2).float()
        kh = k.reshape(bsz, key_len, heads, head_dim).transpose(1, 2).float()
        vh = v.reshape(bsz, key_len, heads, head_dim).transpose(1, 2).float()
        scores = torch.matmul(qh, kh.transpose(-2, -1)) / math.sqrt(max(head_dim, 1))
        mask = attention_mask.to(device=scores.device, dtype=torch.bool).view(1, 1, query_len, key_len)
        raw_scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        raw_attention = torch.softmax(raw_scores, dim=-1)
        log_prior = first_frame_distribution[:, :first_frame_tokens].clamp_min(1.0e-8).log()
        log_prior = log_prior - log_prior.amax(dim=-1, keepdim=True)
        scale = float(getattr(self.cfg, "fastwam_heatmap_guidance_scale", 1.0))
        scores[..., :first_frame_tokens] += (
            scale * confidence[:, None, None, None] * log_prior[:, None, None, :]
        )
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        attention = torch.softmax(scores, dim=-1)
        mixed = torch.matmul(attention, vh).transpose(1, 2).reshape(bsz, query_len, hidden)
        return (
            mixed.to(dtype=q.dtype),
            raw_attention[..., :first_frame_tokens],
            attention[..., :first_frame_tokens],
        )

    def _gt_center_biased_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: torch.Tensor,
        center_gaussian: torch.Tensor,
        first_frame_tokens: int,
        valid_action_queries: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, query_len, hidden = q.shape
        key_len = int(k.size(1))
        heads = int(self.mot.num_heads)
        head_dim = hidden // heads
        qh = q.reshape(bsz, query_len, heads, head_dim).transpose(1, 2).float()
        kh = k.reshape(bsz, key_len, heads, head_dim).transpose(1, 2).float()
        vh = v.reshape(bsz, key_len, heads, head_dim).transpose(1, 2).float()
        raw_logits = torch.matmul(qh, kh.transpose(-2, -1)) / math.sqrt(head_dim)
        mask = attention_mask.to(device=raw_logits.device, dtype=torch.bool).view(
            1, 1, query_len, key_len
        )
        masked_raw_logits = raw_logits.masked_fill(~mask, torch.finfo(raw_logits.dtype).min)
        raw_attention = torch.softmax(masked_raw_logits, dim=-1)

        spatial = center_gaussian[:, :first_frame_tokens].to(raw_logits)
        if bool(getattr(self.cfg, "gt_center_attention_zero_mean", True)):
            spatial = spatial - spatial.mean(dim=-1, keepdim=True)
        ratio = float(getattr(self.cfg, "gt_center_guided_head_ratio", 0.5))
        guided_heads = max(1, min(heads, int(round(heads * ratio))))
        head_mask = (torch.arange(heads, device=q.device) < guided_heads).to(raw_logits.dtype)
        bias = float(getattr(self.cfg, "gt_center_attention_beta", 2.0)) * spatial[:, None, None, :]
        bias = bias * head_mask[None, :, None, None]
        if valid_action_queries is not None:
            query_valid = valid_action_queries.to(device=q.device, dtype=raw_logits.dtype)
            if query_valid.ndim == 3 and query_valid.size(-1) == 1:
                query_valid = query_valid.squeeze(-1)
            if query_valid.shape != (bsz, query_len):
                raise ValueError(
                    f"valid action query mask must be {(bsz, query_len)}, got {tuple(query_valid.shape)}."
                )
            bias = bias * query_valid[:, None, :, None]
        effective_logits = raw_logits.clone()
        effective_logits[..., :first_frame_tokens] += bias
        masked_effective_logits = effective_logits.masked_fill(
            ~mask, torch.finfo(effective_logits.dtype).min
        )
        effective_attention = torch.softmax(masked_effective_logits, dim=-1)
        mixed = torch.matmul(effective_attention, vh).transpose(1, 2).reshape(
            bsz, query_len, hidden
        )
        return (
            mixed.to(dtype=q.dtype),
            raw_logits[..., :first_frame_tokens],
            effective_logits[..., :first_frame_tokens],
            raw_attention[..., :first_frame_tokens],
            effective_attention[..., :first_frame_tokens],
        )

    def _forward_training_mot_with_attention(
        self,
        video_pre: Dict[str, torch.Tensor],
        action_pre: Dict[str, torch.Tensor],
        attention_mask: torch.Tensor,
        guidance_heatmap: Optional[torch.Tensor] = None,
        guidance_confidence: Optional[torch.Tensor] = None,
        gt_center_xy: Optional[torch.Tensor] = None,
        gt_center_visible: Optional[torch.Tensor] = None,
        valid_action_queries: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        xv = video_pre["tokens"]
        xa = action_pre["tokens"]
        video_seq_len = int(xv.size(1))
        action_seq_len = int(xa.size(1))
        total_seq_len = video_seq_len + action_seq_len
        action_attention_mask = attention_mask[video_seq_len:total_seq_len, :total_seq_len]
        video_context = {"context": video_pre["context"], "mask": video_pre["context_mask"]}
        action_context = {"context": action_pre["context"], "mask": action_pre["context_mask"]}
        last_action_attention = None
        last_guided_action_attention = None
        last_raw_action_attention_logits = None
        last_effective_action_attention_logits = None
        grid_hw = tuple(int(v) for v in video_pre["meta"]["grid_size"][-2:])
        guidance = None
        if bool(getattr(self.cfg, "use_fastwam_attention_bias", False)):
            guidance = self._prepare_guidance_distribution(guidance_heatmap, guidance_confidence, grid_hw)
        center_gaussian = None
        if bool(getattr(self.cfg, "use_gt_center_attention_bias", False)):
            if gt_center_xy is None:
                raise RuntimeError("GT center is required for the privileged center-bias teacher.")
            center_gaussian = build_center_gaussian(
                gt_center_xy,
                grid_hw[0],
                grid_hw[1],
                float(getattr(self.cfg, "gt_center_attention_sigma", 1.0)),
                visible=gt_center_visible,
            )
        guided_start = max(
            0,
            int(self.mot.num_layers) - max(int(getattr(self.cfg, "gt_center_guided_layers", 3)), 0),
        )

        for layer_idx in range(self.mot.num_layers):
            video_block = self.video_expert.blocks[layer_idx]
            action_block = self.action_expert.blocks[layer_idx]
            v_io = self.mot._build_expert_attention_io(
                expert=self.video_expert,
                block=video_block,
                x=xv,
                freqs=video_pre["freqs"],
                t_mod=video_pre["t_mod"],
            )
            a_io = self.mot._build_expert_attention_io(
                expert=self.action_expert,
                block=action_block,
                x=xa,
                freqs=action_pre["freqs"],
                t_mod=action_pre["t_mod"],
            )
            q_cat = torch.cat([v_io[0], a_io[0]], dim=1)
            k_cat = torch.cat([v_io[1], a_io[1]], dim=1)
            v_cat = torch.cat([v_io[2], a_io[2]], dim=1)

            if center_gaussian is not None and layer_idx >= guided_start:
                (
                    mixed_action,
                    raw_logits,
                    effective_logits,
                    raw_attention,
                    biased_attention,
                ) = self._gt_center_biased_attention(
                    a_io[0], k_cat, v_cat, action_attention_mask, center_gaussian,
                    int(video_pre["meta"]["tokens_per_frame"]), valid_action_queries,
                )
                mixed_full = self.mot._mixed_attention(
                    q_cat=q_cat, k_cat=k_cat, v_cat=v_cat, attention_mask=attention_mask
                )
                mixed_video = mixed_full[:, :video_seq_len]
                if layer_idx == self.mot.num_layers - 1:
                    last_action_attention = raw_attention
                    last_guided_action_attention = biased_attention
                    last_raw_action_attention_logits = raw_logits
                    last_effective_action_attention_logits = effective_logits
            elif layer_idx == self.mot.num_layers - 1 and guidance is not None:
                mixed_action, raw_attention, biased_attention = self._biased_attention(
                    a_io[0], k_cat, v_cat, action_attention_mask, guidance[0], guidance[1],
                    int(video_pre["meta"]["tokens_per_frame"]),
                )
                mixed_full = self.mot._mixed_attention(
                    q_cat=q_cat, k_cat=k_cat, v_cat=v_cat, attention_mask=attention_mask
                )
                mixed_video = mixed_full[:, :video_seq_len]
                last_action_attention = raw_attention
                last_guided_action_attention = biased_attention
            elif layer_idx == self.mot.num_layers - 1:
                q_action = a_io[0]
                bsz, query_len, hidden = q_action.shape
                num_heads = int(self.mot.num_heads)
                head_dim = hidden // max(num_heads, 1)
                qh = q_action.reshape(bsz, query_len, num_heads, head_dim).transpose(1, 2).float()
                kh = k_cat.reshape(bsz, total_seq_len, num_heads, head_dim).transpose(1, 2).float()
                scores = torch.matmul(qh, kh.transpose(-2, -1)) / math.sqrt(max(head_dim, 1))
                mask = action_attention_mask.to(device=scores.device, dtype=torch.bool).view(
                    1, 1, query_len, total_seq_len
                )
                scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
                last_action_attention = torch.softmax(scores, dim=-1)[..., :video_seq_len]
                mixed = self.mot._mixed_attention(
                    q_cat=q_cat, k_cat=k_cat, v_cat=v_cat, attention_mask=attention_mask
                )
                mixed_video = mixed[:, :video_seq_len]
                mixed_action = mixed[:, video_seq_len:]
            else:
                mixed = self.mot._mixed_attention(
                    q_cat=q_cat, k_cat=k_cat, v_cat=v_cat, attention_mask=attention_mask
                )
                mixed_video = mixed[:, :video_seq_len]
                mixed_action = mixed[:, video_seq_len:]
            xv = self.mot._apply_post_with_optional_checkpoint(
                block=video_block,
                residual_x=v_io[3],
                gate_msa=v_io[4],
                shift_mlp=v_io[5],
                scale_mlp=v_io[6],
                gate_mlp=v_io[7],
                use_gradient_checkpointing=v_io[8],
                mixed_slice=mixed_video,
                context_payload=video_context,
            )
            xa = self.mot._apply_post_with_optional_checkpoint(
                block=action_block,
                residual_x=a_io[3],
                gate_msa=a_io[4],
                shift_mlp=a_io[5],
                scale_mlp=a_io[6],
                gate_mlp=a_io[7],
                use_gradient_checkpointing=a_io[8],
                mixed_slice=mixed_action,
                context_payload=action_context,
            )

        if last_action_attention is None:
            raise RuntimeError("Failed to capture training last-layer action attention.")
        return {
            "video": xv,
            "action": xa,
            "last_action_attention": last_action_attention,
            "last_guided_action_attention": last_guided_action_attention,
            "last_raw_action_attention_logits": last_raw_action_attention_logits,
            "last_effective_action_attention_logits": last_effective_action_attention_logits,
            "center_gaussian": center_gaussian,
        }

    def _attention_heatmap_loss(
        self,
        last_action_attention: torch.Tensor,
        video_pre: Dict[str, torch.Tensor],
        target_relative: torch.Tensor,
        action_valid_mask: Optional[torch.Tensor],
        guidance_heatmap: Optional[torch.Tensor] = None,
        guidance_confidence: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if target_relative is None and guidance_heatmap is None:
            raise RuntimeError("FastWAM attention heatmap loss requires target_relative.")
        if guidance_heatmap is None and (target_relative.ndim != 3 or target_relative.size(-1) < 3):
            raise ValueError("target_relative must have shape [B, T, D>=3].")
        if last_action_attention.ndim != 4:
            raise ValueError("last_action_attention must have shape [B, heads, action_queries, video_tokens].")

        bsz = int(last_action_attention.size(0))
        _, grid_h, grid_w = (int(v) for v in video_pre["meta"]["grid_size"])
        tokens_per_frame = int(video_pre["meta"]["tokens_per_frame"])
        first_frame_tokens = min(tokens_per_frame, int(last_action_attention.size(-1)))
        if first_frame_tokens != grid_h * grid_w:
            raise ValueError(
                "First-frame token count must match video token grid for attention heatmap loss; "
                f"got first_frame_tokens={first_frame_tokens}, grid={grid_h}x{grid_w}."
            )

        if guidance_heatmap is not None:
            gt_flat, confidence = self._prepare_guidance_distribution(
                guidance_heatmap,
                guidance_confidence,
                (grid_h, grid_w),
            )
            visible = torch.ones(bsz, device=last_action_attention.device, dtype=torch.bool)
        else:
            target_first = target_relative[:, :1, :3].to(device=last_action_attention.device, dtype=torch.float32)
            _, visible = project_body_to_image(
                target_first[:, 0],
                (grid_h, grid_w),
                fov_deg=float(getattr(self.cfg, "fastwam_attention_heatmap_fov_deg", 90.0)),
                camera_offset_body=getattr(self.cfg, "fastwam_attention_heatmap_camera_offset_body", None),
            )
            gt_heatmap = make_attention_heatmap(
                target_first,
                (grid_h, grid_w),
                fov_deg=float(getattr(self.cfg, "fastwam_attention_heatmap_fov_deg", 90.0)),
                sigma=float(getattr(self.cfg, "fastwam_attention_heatmap_sigma", 0.08)),
                camera_offset_body=getattr(self.cfg, "fastwam_attention_heatmap_camera_offset_body", None),
            )
            gt_flat = gt_heatmap.reshape(bsz, first_frame_tokens).clamp_min(1.0e-12)
            visible = visible.to(device=last_action_attention.device, dtype=torch.bool)
        query_mode = str(getattr(self.cfg, "tracker_attention_query_mode", "all_queries"))
        if query_mode not in {"query0", "all_queries"}:
            raise ValueError(f"Unsupported tracker_attention_query_mode={query_mode!r}.")
        attention = last_action_attention[:, :, :, :first_frame_tokens].float()
        if query_mode == "query0":
            attention = attention[:, :, :1]
        attn = attention.mean(dim=1).clamp_min(1.0e-8)
        attn = attn / attn.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)
        gt_dist = gt_flat[:, None, :].clamp_min(1.0e-8)
        gt_dist = gt_dist / gt_dist.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)
        kl = (gt_dist * (gt_dist.log() - attn.log())).sum(dim=-1)

        valid = visible[:, None].expand_as(kl)
        if action_valid_mask is not None:
            query_valid = action_valid_mask.to(device=last_action_attention.device, dtype=torch.bool)
            if query_valid.ndim == 3 and query_valid.size(-1) == 1:
                query_valid = query_valid.squeeze(-1)
            if query_valid.ndim != 2 or query_valid.size(0) != kl.size(0) or query_valid.size(1) < kl.size(1):
                raise ValueError(
                    "action_valid_mask must cover every supervised action query; "
                    f"got {tuple(query_valid.shape)} for KL shape {tuple(kl.shape)}."
                )
            valid = valid & query_valid[:, : kl.size(1)]
        if not valid.any():
            return last_action_attention.sum() * 0.0
        return kl[valid].mean()

    def _ortrack_consistency_loss(
        self,
        last_action_attention: torch.Tensor,
        video_pre: Dict[str, torch.Tensor],
        guidance_heatmap: Optional[torch.Tensor],
        guidance_confidence: Optional[torch.Tensor],
        action_valid_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if guidance_heatmap is None:
            return last_action_attention.sum() * 0.0
        _, grid_h, grid_w = (int(v) for v in video_pre["meta"]["grid_size"])
        track_dist, confidence = self._prepare_guidance_distribution(
            guidance_heatmap, guidance_confidence, (grid_h, grid_w)
        )
        tokens = min(grid_h * grid_w, last_action_attention.size(-1))
        query_mode = str(getattr(self.cfg, "tracker_attention_query_mode", "all_queries"))
        if query_mode not in {"query0", "all_queries"}:
            raise ValueError(f"Unsupported tracker_attention_query_mode={query_mode!r}.")
        attention = last_action_attention[:, :, :, :tokens].float()
        if query_mode == "query0":
            attention = attention[:, :, :1]

        # Preserve each action query, average all heads, then normalize each
        # query independently over the first-frame visual token grid.
        pred = attention.mean(dim=1).clamp_min(1.0e-8)
        pred = pred / pred.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)
        target = track_dist[:, None, :tokens].clamp_min(1.0e-8)
        kl = (target * (target.log() - pred.log())).sum(dim=-1)
        valid = torch.ones_like(kl, dtype=torch.bool)
        if action_valid_mask is not None:
            query_valid = action_valid_mask.to(device=kl.device, dtype=torch.bool)
            if query_valid.ndim != 2 or query_valid.size(0) != kl.size(0) or query_valid.size(1) < kl.size(1):
                raise ValueError(
                    "action_valid_mask must cover every supervised action query; "
                    f"got {tuple(query_valid.shape)} for KL shape {tuple(kl.shape)}."
                )
            query_valid = query_valid[:, : kl.size(1)]
            valid = valid & query_valid
        if not valid.any():
            return last_action_attention.sum() * 0.0
        return kl[valid].mean()

    def training_loss(
        self,
        video_latents: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        expert_action: torch.Tensor,
        valid_mask: Optional[torch.Tensor],
        target_relative: Optional[torch.Tensor] = None,
        guidance_heatmap: Optional[torch.Tensor] = None,
        guidance_confidence: Optional[torch.Tensor] = None,
        gt_center_xy: Optional[torch.Tensor] = None,
        tracker_features: Optional[torch.Tensor] = None,
        tracker_template_features: Optional[torch.Tensor] = None,
        tracker_confidence: Optional[torch.Tensor] = None,
        tracker_center: Optional[torch.Tensor] = None,
        tracker_bbox: Optional[torch.Tensor] = None,
        tracker_response: Optional[torch.Tensor] = None,
        tracker_search_geometry: Optional[torch.Tensor] = None,
        tracker_image_size: Optional[torch.Tensor] = None,
        target_box_history: Optional[torch.Tensor] = None,
        target_box_history_valid: Optional[torch.Tensor] = None,
        target_centers: Optional[torch.Tensor] = None,
        target_boxes: Optional[torch.Tensor] = None,
        target_center_valid: Optional[torch.Tensor] = None,
        capture_attention: bool = False,
        return_flow_predictions: bool = False,
        noise_video_override: Optional[torch.Tensor] = None,
        t_video_override: Optional[torch.Tensor] = None,
        noise_action_override: Optional[torch.Tensor] = None,
        t_action_override: Optional[torch.Tensor] = None,
        action_context: Optional[torch.Tensor] = None,
        action_context_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if video_latents.ndim != 5:
            raise ValueError("video_latents must have shape [B, C, T_lat, H_lat, W_lat].")
        b = video_latents.size(0)
        action_tokens_clean, action_valid_mask = self._make_training_action_tokens(expert_action, valid_mask)
        action_tokens_clean = action_tokens_clean.to(device=video_latents.device, dtype=video_latents.dtype)
        action_token_len = int(action_tokens_clean.size(1))
        if self.target_action_conditioning is not None:
            previous_length = self.target_action_conditioning.history_length - 1
            expected_history = (b, previous_length, 5)
            if target_box_history is None or target_box_history.shape != expected_history:
                raise ValueError(
                    "Historical Target Memory requires previous Tracker states "
                    f"{expected_history}, got "
                    f"{None if target_box_history is None else tuple(target_box_history.shape)}."
                )
            if (
                target_box_history_valid is None
                or target_box_history_valid.shape != (b, previous_length)
            ):
                raise ValueError(
                    "Historical Target Memory validity must match previous history."
                )
        latent_transitions = max(int(video_latents.size(2)) - 1, 1)
        if action_token_len % latent_transitions != 0:
            raise ValueError(
                "FastWAM action token length must be divisible by Wan latent transitions; "
                f"got action_len={action_token_len}, latent_frames={int(video_latents.size(2))}."
            )

        noise_video = (
            torch.randn_like(video_latents)
            if noise_video_override is None
            else noise_video_override.to(device=video_latents.device, dtype=video_latents.dtype)
        )
        if noise_video.shape != video_latents.shape:
            raise ValueError("noise_video_override must match video_latents shape.")
        t_video = (
            self.video_scheduler.sample_training_t(b, video_latents.device, video_latents.dtype)
            if t_video_override is None
            else t_video_override.to(device=video_latents.device, dtype=video_latents.dtype)
        )
        if t_video.shape != (b,):
            raise ValueError("t_video_override must have shape [B].")
        noisy_video = self.video_scheduler.add_noise(video_latents, noise_video, t_video)
        target_video = self.video_scheduler.training_target(video_latents, noise_video, t_video)
        first_frame_latents = video_latents[:, :, 0:1].clone()
        noisy_video[:, :, 0:1] = first_frame_latents

        noise_action = (
            torch.randn_like(action_tokens_clean)
            if noise_action_override is None
            else noise_action_override.to(device=action_tokens_clean.device, dtype=action_tokens_clean.dtype)
        )
        if noise_action.shape != action_tokens_clean.shape:
            raise ValueError("noise_action_override must match clean action-token shape.")
        t_action = (
            self.action_scheduler.sample_training_t(b, action_tokens_clean.device, action_tokens_clean.dtype)
            if t_action_override is None
            else t_action_override.to(device=action_tokens_clean.device, dtype=action_tokens_clean.dtype)
        )
        if t_action.shape != (b,):
            raise ValueError("t_action_override must have shape [B].")
        noisy_action = self.action_scheduler.add_noise(action_tokens_clean, noise_action, t_action)
        target_action = self.action_scheduler.training_target(action_tokens_clean, noise_action, t_action)

        state_valid_mask = None
        future_state_valid_mask = None
        clean_future_states = None
        noisy_future_states = None
        target_state_flow = None
        if self.state_expert is not None:
            horizon = int(getattr(self.cfg, "future_state_horizon", 8))
            if target_boxes is None or target_center_valid is None:
                raise RuntimeError("Future State DiT training requires target boxes and validity.")
            if target_boxes.ndim != 3 or tuple(target_boxes.shape[1:]) != (horizon + 1, 4):
                raise ValueError(
                    f"Future State targets must be [B,{horizon + 1},4], got {tuple(target_boxes.shape)}."
                )
            assert self.future_state_conditioner is not None
            clean_future_states = self.future_state_conditioner.relative_states(target_boxes)
            if clean_future_states.shape != (b, horizon, 4):
                raise ValueError("Relative Future State target shape is invalid.")
            state_valid_mask, future_state_valid_mask = self._state_validity(
                target_center_valid, valid_mask, horizon
            )
            state_noise = torch.randn_like(clean_future_states).to(
                device=video_latents.device, dtype=video_latents.dtype
            )
            clean_future_states = clean_future_states.to(
                device=video_latents.device, dtype=video_latents.dtype
            )
            assert self.state_scheduler is not None
            noisy_future_states = self.state_scheduler.add_noise(
                clean_future_states, state_noise, t_action
            )
            target_state_flow = self.state_scheduler.training_target(
                clean_future_states, state_noise, t_action
            )
            if not torch.isfinite(noisy_future_states).all() or not torch.isfinite(target_state_flow).all():
                raise FloatingPointError("Future State flow construction produced NaN/Inf.")

        context = context.to(device=video_latents.device, dtype=video_latents.dtype)
        context_mask = context_mask.to(device=video_latents.device, dtype=torch.bool)
        if action_context is None:
            if action_context_mask is not None:
                raise ValueError("action_context_mask requires action_context.")
            action_context = context
            action_context_mask = context_mask
        elif action_context_mask is None:
            raise ValueError("action_context requires action_context_mask.")
        else:
            action_context = action_context.to(
                device=video_latents.device, dtype=video_latents.dtype
            )
            action_context_mask = action_context_mask.to(
                device=video_latents.device, dtype=torch.bool
            )
        video_pre = self.video_expert.pre_dit(
            x=noisy_video,
            timestep=t_video,
            context=context,
            context_mask=context_mask,
            action=action_tokens_clean,
            fuse_vae_embedding_in_latents=True,
        )
        action_pre = self.action_expert.pre_dit(
            action_tokens=noisy_action,
            timestep=t_action,
            context=action_context,
            context_mask=action_context_mask,
        )
        tracker_condition = self._make_tracker_condition(
            tracker_features,
            tracker_center,
            tracker_bbox,
            tracker_response,
            tracker_search_geometry,
            tracker_image_size,
            device=video_latents.device,
            dtype=video_latents.dtype,
        )
        future_state_condition = self._make_future_state_condition(
            tracker_template_features,
            tracker_condition,
            tracker_search_geometry,
            tracker_image_size,
        )
        current_target = self._make_current_target_localization(
            tracker_template_features,
            tracker_condition,
            tracker_search_geometry,
            tracker_image_size,
            tracker_bbox,
        )
        state_pre = None
        if self.state_expert is not None:
            if future_state_condition is None or noisy_future_states is None or state_valid_mask is None:
                raise RuntimeError("Future State DiT inputs were not constructed.")
            state_pre = self.state_expert.pre_dit(
                noisy_future_states=noisy_future_states,
                current_condition=future_state_condition["current_condition"],
                timestep=t_action,
                tracker_memory=future_state_condition["tracker_memory"],
                state_valid_mask=state_valid_mask,
            )
            if state_pre["tokens"].shape != (
                b,
                int(getattr(self.cfg, "future_state_horizon", 8)) + 1,
                int(getattr(self.cfg, "future_state_hidden_dim", 1024)),
            ):
                raise ValueError("StateDiT hidden token shape assertion failed.")
        mask = self._build_mot_attention_mask(
            video_seq_len=video_pre["tokens"].size(1),
            action_seq_len=action_pre["tokens"].size(1),
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_latents.device,
            tracker_seq_len=0,
        )
        if (
            str(getattr(self.cfg, "fastwam_heatmap_source", "none")) == "gt"
            and bool(getattr(self.cfg, "use_fastwam_attention_bias", False))
        ):
            if target_relative is None:
                raise RuntimeError("GT attention bias requires target_relative during training.")
            grid_hw = tuple(int(v) for v in video_pre["meta"]["grid_size"][-2:])
            guidance_heatmap = make_attention_heatmap(
                target_relative[:, 0, :3].float(),
                grid_hw,
                fov_deg=float(getattr(self.cfg, "fastwam_heatmap_guidance_fov_deg", 90.0)),
                sigma=float(getattr(self.cfg, "fastwam_heatmap_guidance_sigma", 0.08)),
                camera_offset_body=getattr(
                    self.cfg, "fastwam_heatmap_guidance_camera_offset_body", None
                ),
            )
            guidance_confidence = torch.ones(
                (video_latents.size(0), 1), device=video_latents.device, dtype=torch.float32
            )
        use_attention_heatmap_loss = bool(
            getattr(self.cfg, "use_fastwam_attention_heatmap_loss", False)
        )
        use_tracker_heatmap_loss = bool(
            getattr(self.cfg, "use_fastwam_tracker_heatmap_loss", False)
        )
        use_attention_bias = bool(getattr(self.cfg, "use_fastwam_attention_bias", False))
        use_gt_center_bias = bool(getattr(self.cfg, "use_gt_center_attention_bias", False))
        if use_gt_center_bias and gt_center_xy is None:
            raise RuntimeError("GT center is required for the privileged center-bias teacher.")
        if self.tracker_integration in {
            "frozen_deit_tracker_fusion",
            "frozen_deit_tracker_local_feature",
        } and (
            use_attention_heatmap_loss
            or use_tracker_heatmap_loss
            or use_attention_bias
            or use_gt_center_bias
            or capture_attention
        ):
            raise RuntimeError(
                "Tracker MoT feature experiments currently isolate the fusion mechanism; "
                "attention heatmap/bias/capture must be disabled during training."
            )
        fused_video_cache = None
        capture_box_feature = None
        target_conditions = None
        if self.target_action_conditioning is not None:
            if current_target is None:
                raise RuntimeError("Historical Target Memory did not construct current b0.")
            assert target_box_history is not None
            assert target_box_history_valid is not None
            target_conditions = self.target_action_conditioning.build_conditions(
                previous_history=target_box_history,
                previous_valid=target_box_history_valid,
                current_box=current_target["current_box"],
                current_confidence=tracker_confidence,
            )
        if self.dot is not None:
            if capture_attention:
                raise ValueError("Faster-WAM DoT does not expose legacy MoT attention maps.")
            video_len = int(video_pre["tokens"].size(1))
            if self.target_action_conditioning is not None and current_target is None:
                raise RuntimeError("Target-conditioned FasterWAM did not construct current b0.")
            video_hidden, fused_video_cache = self.dot.prefill_video(
                video_expert=self.video_expert,
                video_pre=video_pre,
                video_attention_mask=mask[:video_len, :video_len],
                tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            )
            condition_residual = None
            if self.current_box_action_conditioner is not None:
                if current_target is None:
                    raise RuntimeError("Current Tracker box b0 was not constructed.")
                box_feature = self.current_box_action_conditioner.encode_box(
                    current_target["current_box"].to(
                        device=action_pre["tokens"].device,
                        dtype=action_pre["tokens"].dtype,
                    )
                )
                capture_box_feature = box_feature

                def condition_residual(hidden: torch.Tensor) -> torch.Tensor:
                    delta = self.current_box_action_conditioner.delta(
                        0, hidden, box_feature
                    )
                    if self.target_action_conditioning is not None:
                        assert target_conditions is not None
                        delta = delta + self.target_action_conditioning.residual(
                            hidden, target_conditions
                        )
                    return delta

            action_hidden = self.dot.forward_action(
                action_expert=self.action_expert,
                action_pre=action_pre,
                fused_video_cache=fused_video_cache,
                condition_residual=condition_residual,
            )
            out = {
                "video": video_hidden,
                "action": action_hidden,
            }
            loss_attention_heatmap = video_latents.sum() * 0.0
            loss_ortrack_consistency = video_latents.sum() * 0.0
        elif self.current_box_action_conditioner is not None:
            if current_target is None:
                raise RuntimeError("Current Tracker box b0 was not constructed.")
            if (
                use_attention_heatmap_loss
                or use_tracker_heatmap_loss
                or use_attention_bias
                or use_gt_center_bias
                or capture_attention
            ):
                raise ValueError(
                    "Current-box Action conditioning isolates legacy attention auxiliaries."
                )
            out = self._forward_video_action_with_current_box(
                video_pre,
                action_pre,
                mask,
                current_target,
                target_conditions,
            )
            loss_attention_heatmap = video_latents.sum() * 0.0
            loss_ortrack_consistency = video_latents.sum() * 0.0
        elif self.state_expert is not None:
            if use_attention_heatmap_loss or use_tracker_heatmap_loss or use_attention_bias or use_gt_center_bias or capture_attention:
                raise ValueError("V4 StateDiT isolates attention heatmap/bias auxiliaries.")
            assert state_pre is not None
            out = self._forward_three_expert_layers(
                video_tokens=video_pre["tokens"],
                video_pre=video_pre,
                action_pre=action_pre,
                state_pre=state_pre,
                base_attention_mask=mask,
                action_valid_mask=action_valid_mask,
            )
            loss_attention_heatmap = video_latents.sum() * 0.0
            loss_ortrack_consistency = video_latents.sum() * 0.0
        elif use_attention_heatmap_loss or use_tracker_heatmap_loss or use_attention_bias or use_gt_center_bias or capture_attention:
            out = self._forward_training_mot_with_attention(
                video_pre=video_pre,
                action_pre=action_pre,
                attention_mask=mask,
                guidance_heatmap=guidance_heatmap,
                guidance_confidence=guidance_confidence,
                gt_center_xy=None if gt_center_xy is None else gt_center_xy[:, 0],
                gt_center_visible=guidance_confidence,
                valid_action_queries=action_valid_mask,
            )
            if use_attention_heatmap_loss:
                loss_attention_heatmap = self._attention_heatmap_loss(
                    out["last_action_attention"],
                    video_pre,
                    target_relative,
                    action_valid_mask,
                    guidance_heatmap,
                    guidance_confidence,
                )
            else:
                loss_attention_heatmap = video_latents.sum() * 0.0
            if use_tracker_heatmap_loss:
                loss_ortrack_consistency = self._ortrack_consistency_loss(
                    out["last_action_attention"], video_pre, guidance_heatmap,
                    guidance_confidence, action_valid_mask,
                )
            else:
                loss_ortrack_consistency = video_latents.sum() * 0.0
        else:
            if tracker_condition is not None:
                out = self._forward_training_with_tracker_fusion(
                    video_pre,
                    action_pre,
                    tracker_condition,
                    mask,
                    tracker_template_features=tracker_template_features,
                    tracker_search_geometry=tracker_search_geometry,
                    tracker_image_size=tracker_image_size,
                    action_timestep=t_action,
                )
            else:
                embeds_all = {"video": video_pre["tokens"], "action": action_pre["tokens"]}
                freqs_all = {"video": video_pre["freqs"], "action": action_pre["freqs"]}
                context_all = {
                    "video": {"context": video_pre["context"], "mask": video_pre["context_mask"]},
                    "action": {"context": action_pre["context"], "mask": action_pre["context_mask"]},
                }
                t_mod_all = {"video": video_pre["t_mod"], "action": action_pre["t_mod"]}
                mot_kwargs = {
                    "embeds_all": embeds_all,
                    "attention_mask": mask,
                    "freqs_all": freqs_all,
                    "context_all": context_all,
                    "t_mod_all": t_mod_all,
                }
                if self.mot is None:
                    raise RuntimeError("FastWAM MoT is not initialized.")
                out = self.mot(**mot_kwargs)
            loss_attention_heatmap = video_latents.sum() * 0.0
            loss_ortrack_consistency = video_latents.sum() * 0.0
        pred_video = self.video_expert.post_dit(out["video"], video_pre)
        pred_action = self.action_expert.post_dit(out["action"], action_pre)
        pred_state_flow = None
        if self.state_expert is not None:
            assert state_pre is not None
            pred_state_flow = self.state_expert.post_dit(out["state"], state_pre)

        pred_video = pred_video[:, :, 1:]
        target_video = target_video[:, :, 1:]
        video_loss_token = F.mse_loss(pred_video.float(), target_video.float(), reduction="none").mean(dim=(1, 3, 4))
        video_loss_per_sample = video_loss_token.mean(dim=1)
        video_weight = self.video_scheduler.training_weight(t_video).to(video_loss_per_sample.device, video_loss_per_sample.dtype)
        loss_video = (video_loss_per_sample * video_weight).mean()

        action_loss_token = weighted_mean_action_squared_error(
            pred_action, target_action, self.cfg
        )
        if action_valid_mask is not None:
            valid = action_valid_mask.to(device=action_loss_token.device, dtype=action_loss_token.dtype)
            action_loss_per_sample = (action_loss_token * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)
        else:
            action_loss_per_sample = action_loss_token.mean(dim=1)
        action_weight = self.action_scheduler.training_weight(t_action).to(action_loss_per_sample.device, action_loss_per_sample.dtype)
        loss_action = (action_loss_per_sample * action_weight).mean()

        loss_state_flow = video_latents.sum() * 0.0
        loss_current_box = video_latents.sum() * 0.0
        loss_current_center_spatial = video_latents.sum() * 0.0
        loss_current_box_giou = video_latents.sum() * 0.0
        loss_current_attention = video_latents.sum() * 0.0
        state_valid_ratio = video_latents.sum() * 0.0
        current_box_valid_ratio = video_latents.sum() * 0.0
        predicted_s0_box_error = video_latents.sum() * 0.0
        predicted_s0_center_error_pixels = video_latents.sum() * 0.0
        predicted_future_state_error = video_latents.sum() * 0.0
        predicted_state_boxes_v4 = None
        predicted_relative_states = None
        if self.state_expert is not None:
            if (
                pred_state_flow is None
                or target_state_flow is None
                or future_state_valid_mask is None
                or future_state_condition is None
                or noisy_future_states is None
                or clean_future_states is None
                or target_boxes is None
            ):
                raise RuntimeError("Future State loss inputs are incomplete.")
            if pred_state_flow.shape != target_state_flow.shape or pred_state_flow.shape != (b, 8, 4):
                raise ValueError("State flow input/output must be [B,8,4].")
            state_error = F.mse_loss(
                pred_state_flow.float(), target_state_flow.float(), reduction="none"
            ).mean(dim=-1)
            state_valid_float = future_state_valid_mask.to(
                device=state_error.device, dtype=state_error.dtype
            )
            state_per_sample = (state_error * state_valid_float).sum(dim=1) / state_valid_float.sum(
                dim=1
            ).clamp_min(1.0)
            assert self.state_scheduler is not None
            state_weight = self.state_scheduler.training_weight(t_action).to(state_per_sample)
            loss_state_flow = (state_per_sample * state_weight).mean()
            if not torch.isfinite(loss_state_flow):
                raise FloatingPointError("State flow loss is NaN/Inf.")

            current_box = future_state_condition["current_box"].float()
            if current_box.shape != (b, 4):
                raise ValueError("Current box must be [B,4].")
            current_valid = state_valid_mask[:, 0].to(
                device=current_box.device, dtype=current_box.dtype
            )
            current_error = F.smooth_l1_loss(
                current_box, target_boxes[:, 0].to(current_box), reduction="none"
            ).mean(dim=-1)
            loss_current_box = (current_error * current_valid).sum() / current_valid.sum().clamp_min(1.0)
            if tracker_image_size is None:
                raise RuntimeError("Current state spatial losses require Tracker image size.")
            current_tracking = self.future_state_conditioner.current_tracking_losses(
                current_box=current_box,
                current_attention=future_state_condition["current_attention"],
                full_xy=future_state_condition["full_xy"],
                target_box=target_boxes[:, 0],
                valid=current_valid,
                image_size=tracker_image_size,
                attention_sigma=float(getattr(self.cfg, "current_attention_sigma", 1.5)),
            )
            loss_current_center_spatial = current_tracking["center"]
            loss_current_box_giou = current_tracking["giou"]
            loss_current_attention = current_tracking["attention"]
            predicted_s0_center_error_pixels = current_tracking["center_error_pixels"]
            state_valid_ratio = state_valid_float.mean()
            current_box_valid_ratio = current_valid.mean()
            predicted_s0_box_error = (
                (current_box - target_boxes[:, 0].to(current_box)).abs().mean(dim=-1) * current_valid
            ).sum() / current_valid.sum().clamp_min(1.0)

            sigma = (t_action.float() / float(self.state_scheduler.num_train_timesteps)).view(-1, 1, 1)
            predicted_relative_states = noisy_future_states.float() - sigma * pred_state_flow.float()
            relative_error = (
                predicted_relative_states - clean_future_states.float()
            ).abs().mean(dim=-1)
            predicted_future_state_error = (
                relative_error * state_valid_float
            ).sum() / state_valid_float.sum().clamp_min(1.0)
            predicted_state_boxes_v4 = self.future_state_conditioner.decode_relative_states(
                current_box, predicted_relative_states
            )

        loss_center_flow = video_latents.sum() * 0.0
        loss_current_center = video_latents.sum() * 0.0
        loss_future_center = video_latents.sum() * 0.0
        loss_center_transition = video_latents.sum() * 0.0
        loss_box_l1 = video_latents.sum() * 0.0
        loss_box_giou = video_latents.sum() * 0.0
        pred_center_flow = None
        pred_state_centers = None
        pred_future_centers = None
        pred_state_boxes = None
        pred_future_boxes = None
        if self.state_expert is None and self.future_target_readout is not None and bool(
            getattr(self.cfg, "tracker_center_flow_supervision", False)
        ):
            final_future_target_tokens = out.get("final_future_target_tokens")
            current_soft_center = out.get("current_soft_center")
            current_target_size = out.get("current_target_size")
            if final_future_target_tokens is None or current_soft_center is None or current_target_size is None:
                raise RuntimeError("Future Target box supervision requires final tokens, center, and size.")
            pred_state_boxes, pred_center_flow, _ = self.future_target_readout.state_boxes(
                final_future_target_tokens,
                {"soft_center": current_soft_center, "current_size": current_target_size},
            )
            pred_state_centers = pred_state_boxes[..., :2]
            pred_future_centers = pred_state_centers[:, 1:]
            pred_future_boxes = pred_state_boxes[:, 1:]
            if target_centers is None or target_boxes is None or target_center_valid is None:
                raise RuntimeError("Future Target supervision requires centers, boxes, and validity masks.")
            state_losses = self.future_target_readout.center_state_losses(
                pred_state_centers,
                pred_center_flow,
                target_centers,
                target_center_valid,
                valid_mask,
                float(getattr(self.cfg, "tracker_future_horizon_discount", 0.9)),
            )
            loss_current_center = state_losses["current"]
            loss_future_center = state_losses["future"]
            loss_center_transition = state_losses["transition"]
            box_losses = self.future_target_readout.box_state_losses(
                pred_state_boxes,
                target_boxes,
                target_center_valid,
                valid_mask,
                float(getattr(self.cfg, "tracker_future_horizon_discount", 0.9)),
            )
            loss_box_l1 = box_losses["l1"]
            loss_box_giou = box_losses["giou"]
            loss_center_flow = (
                float(getattr(self.cfg, "tracker_current_center_loss_weight", 1.0))
                * loss_current_center
                + float(getattr(self.cfg, "tracker_future_center_loss_weight", 1.0))
                * loss_future_center
                + float(getattr(self.cfg, "tracker_center_transition_loss_weight", 0.5))
                * loss_center_transition
                + float(getattr(self.cfg, "tracker_box_l1_loss_weight", 5.0))
                * loss_box_l1
                + float(getattr(self.cfg, "tracker_box_giou_loss_weight", 2.0))
                * loss_box_giou
            )

        result = {
            "loss_video": loss_video,
            "loss_action": loss_action,
            "loss_attention_heatmap": loss_attention_heatmap,
            "loss_ortrack_consistency": loss_ortrack_consistency,
            "loss_center_flow": loss_center_flow,
            "loss_current_center": loss_current_center,
            "loss_future_center": loss_future_center,
            "loss_center_transition": loss_center_transition,
            "loss_box_l1": loss_box_l1,
            "loss_box_giou": loss_box_giou,
            "loss_current_box": loss_current_box,
            "loss_current_center_spatial": loss_current_center_spatial,
            "loss_current_box_giou": loss_current_box_giou,
            "loss_current_attention": loss_current_attention,
            "current_box_valid_ratio": current_box_valid_ratio,
            "predicted_s0_box_error": predicted_s0_box_error,
            "predicted_s0_center_error_pixels": predicted_s0_center_error_pixels,
            "pred_action_flow": pred_action.reshape(
                b, action_token_len, 1, self.cfg.action_dim
            ),
        }
        pred_action_x0 = self.action_scheduler.predict_x0(
            noisy_action.float(), pred_action.float(), t_action.float()
        )
        result["pred_action_x0"] = pred_action_x0.reshape(
            b, action_token_len, 1, self.cfg.action_dim
        )
        if action_valid_mask is not None:
            result["action_valid_mask"] = action_valid_mask
        if self.capture_value_head is not None:
            if (
                fused_video_cache is None
                or capture_box_feature is None
                or target_relative is None
            ):
                raise RuntimeError(
                    "Capture-value training requires Video K/V, current b0, and "
                    "future target-relative supervision."
                )
            horizon = int(action_tokens_clean.size(1))
            if horizon != int(getattr(self.cfg, "action_sequence_horizon", 8)):
                raise ValueError(
                    "Capture-value training action horizon does not match its head."
                )
            from .capture_value_reranker import (
                approximate_capture_outcomes,
                build_capture_value_candidates,
                capture_value_loss,
            )

            candidates = build_capture_value_candidates(
                pred_action_x0.detach(),
                action_tokens_clean.detach(),
                candidate_count=int(
                    getattr(self.cfg, "capture_value_candidate_count", 4)
                ),
                noise_std=float(
                    getattr(self.cfg, "capture_value_candidate_noise_std", 0.15)
                ),
            )
            targets = approximate_capture_outcomes(
                candidates,
                action_tokens_clean.detach(),
                target_relative[:, : horizon + 1].detach(),
                action_valid_mask,
                max_vel=float(getattr(self.cfg, "max_vel", 5.0)),
                max_yaw_rate=float(getattr(self.cfg, "max_yaw_rate", 30.0)),
                capture_distance=float(
                    getattr(self.cfg, "capture_value_capture_distance", 10.0)
                ),
            )
            video_context = torch.cat(
                [
                    fused_video_cache["canonical_key"].mean(dim=2),
                    fused_video_cache["value"].mean(dim=2),
                ],
                dim=-1,
            ).mean(dim=0)
            value_prediction = self.capture_value_head(
                candidates,
                video_context=video_context,
                target_context=capture_box_feature,
                valid_mask=action_valid_mask,
            )
            value_losses = capture_value_loss(
                value_prediction,
                targets,
                distance_score_weight=float(
                    getattr(self.cfg, "capture_value_distance_score_weight", 1.0)
                ),
                visibility_score_weight=float(
                    getattr(self.cfg, "capture_value_visibility_score_weight", 0.25)
                ),
            )
            result.update(
                {
                    "loss_capture_value": value_losses["loss"],
                    "capture_value_capture_loss": value_losses["capture"],
                    "capture_value_distance_loss": value_losses["distance"],
                    "capture_value_visibility_loss": value_losses["visibility"],
                    "capture_value_ranking_loss": value_losses["ranking"],
                    "capture_value_ranking_accuracy": value_losses["accuracy"],
                    "capture_value_target_capture": value_losses["target_capture"],
                }
            )
        if self.state_expert is not None:
            result["loss_state_flow"] = loss_state_flow
            result["state_valid_ratio"] = state_valid_ratio
            result["state_to_action_gate_mean"] = torch.tanh(
                self.state_to_action_gates.float()
            ).abs().mean()
            result["predicted_future_state_error"] = predicted_future_state_error
            result["pred_state_flow"] = pred_state_flow
        if self.current_box_action_conditioner is not None:
            result["current_box_action_gate_mean"] = (
                self.current_box_action_conditioner.gate_mean()
            )
        if predicted_state_boxes_v4 is not None:
            result["pred_state_boxes"] = predicted_state_boxes_v4
            result["pred_future_boxes"] = predicted_state_boxes_v4[:, 1:]
            result["pred_state_centers"] = predicted_state_boxes_v4[..., :2]
            result["pred_future_centers"] = predicted_state_boxes_v4[:, 1:, :2]
            result["predicted_relative_states"] = predicted_relative_states
        if pred_center_flow is not None:
            result["pred_center_flow"] = pred_center_flow
            result["pred_state_centers"] = pred_state_centers
            result["pred_future_centers"] = pred_future_centers
            result["pred_state_boxes"] = pred_state_boxes
            result["pred_future_boxes"] = pred_future_boxes
        if return_flow_predictions:
            result["pred_video_velocity"] = pred_video
        if capture_attention:
            result["last_action_attention"] = out["last_action_attention"]
            guided = out.get("last_guided_action_attention")
            result["last_guided_action_attention"] = (
                out["last_action_attention"] if guided is None else guided
            )
            result["last_raw_action_attention_logits"] = out.get("last_raw_action_attention_logits")
            result["last_effective_action_attention_logits"] = out.get(
                "last_effective_action_attention_logits"
            )
            result["center_gaussian"] = out.get("center_gaussian")
        return result

    @torch.no_grad()
    def _build_mot_attention_mask(
        self,
        video_seq_len: int,
        action_seq_len: int,
        video_tokens_per_frame: int,
        device: torch.device,
        tracker_seq_len: int = 0,
    ) -> torch.Tensor:
        tracker_seq_len = max(int(tracker_seq_len), 0)
        total_seq_len = video_seq_len + tracker_seq_len + action_seq_len
        mask = torch.zeros((total_seq_len, total_seq_len), dtype=torch.bool, device=device)
        mask[:video_seq_len, :video_seq_len] = self.video_expert.build_video_to_video_mask(
            video_seq_len=video_seq_len,
            video_tokens_per_frame=video_tokens_per_frame,
            device=device,
        )
        first_frame_tokens = min(video_tokens_per_frame, video_seq_len)
        tracker_start = video_seq_len
        action_start = video_seq_len + tracker_seq_len
        if tracker_seq_len:
            tracker_end = action_start
            mask[tracker_start:tracker_end, :first_frame_tokens] = True
            mask[tracker_start:tracker_end, tracker_start:tracker_end] = True
        mask[action_start:, :first_frame_tokens] = True
        if tracker_seq_len:
            mask[action_start:, tracker_start:action_start] = True
        mask[action_start:, action_start:] = True
        return mask

    @torch.no_grad()
    def sample_video(
        self,
        first_frame_latents: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        latent_frames: int,
        action: Optional[torch.Tensor] = None,
        num_steps: Optional[int] = None,
    ) -> torch.Tensor:
        if first_frame_latents.ndim != 5:
            raise ValueError("first_frame_latents must have shape [B, C, 1, H, W].")
        latent_frames = max(int(latent_frames), 1)
        b, c, _, h, w = first_frame_latents.shape
        device = first_frame_latents.device
        video = torch.randn(b, c, latent_frames, h, w, device=device, dtype=first_frame_latents.dtype)
        video[:, :, 0:1] = first_frame_latents
        context = context.to(device=device, dtype=first_frame_latents.dtype)
        context_mask = context_mask.to(device=device, dtype=torch.bool)
        action_cond = None if action is None else action.to(device=device, dtype=first_frame_latents.dtype)
        steps = int(num_steps or self.cfg.action_sampling_steps)
        timesteps, deltas = self.video_scheduler.build_inference_schedule(steps, device=device, dtype=first_frame_latents.dtype)
        for step_t, step_delta in zip(timesteps, deltas):
            pred_video = self.video_expert(
                x=video,
                timestep=step_t.expand(b),
                context=context,
                context_mask=context_mask,
                action=action_cond,
                fuse_vae_embedding_in_latents=True,
            )
            video = self.video_scheduler.step(pred_video, step_delta, video)
            video[:, :, 0:1] = first_frame_latents
        return video

    def _forward_action_with_tracker_fusion(
        self,
        action_tokens: torch.Tensor,
        action_pre: Dict[str, torch.Tensor],
        video_kv_cache: list[dict[str, torch.Tensor]],
        attention_mask: torch.Tensor,
        video_seq_len: int,
        tracker_condition: torch.Tensor,
        future_target_state: Optional[dict[str, torch.Tensor]] = None,
        action_timestep: Optional[torch.Tensor] = None,
        first_frame_tokens: int = 0,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        action_len = int(action_tokens.size(1))
        total_len = int(video_seq_len) + action_len
        action_mask = attention_mask[video_seq_len:total_len, :total_len]
        expert = self.mot.mixtures["action"]
        x = action_tokens
        final_future_target_tokens = None
        assert self.tracker_fusion is not None
        for layer_idx in range(self.mot.num_layers):
            block = expert.blocks[layer_idx]
            io = self.mot._build_expert_attention_io(
                expert, block, x, action_pre["freqs"], action_pre["t_mod"]
            )
            cache = video_kv_cache[layer_idx]
            mixed = self.mot._mixed_attention(
                q_cat=io[0],
                k_cat=torch.cat([cache["k"], io[1]], dim=1),
                v_cat=torch.cat([cache["v"], io[2]], dim=1),
                attention_mask=action_mask,
            )
            tracker_delta = (
                self.tracker_fusion.delta(layer_idx, x, tracker_condition)
                if self.tracker_spatial_cross_attention
                else torch.zeros_like(x)
            )
            future_delta = torch.zeros_like(x)
            if future_target_state is not None:
                assert self.future_target_readout is not None and action_timestep is not None
                video_hidden = cache.get("hidden")
                if video_hidden is None:
                    raise RuntimeError("Layer-aligned Future Target Readout requires cached Video hidden states.")
                memory_len = video_hidden.size(1) if first_frame_tokens <= 0 else min(first_frame_tokens, video_hidden.size(1))
                video_hidden = video_hidden[:, :memory_len]
                if layer_idx == self.mot.num_layers - 1:
                    future_delta, final_future_target_tokens = self.future_target_readout.delta(
                        layer_idx, x, video_hidden, future_target_state, action_timestep, return_tokens=True
                    )
                else:
                    future_delta = self.future_target_readout.delta(
                        layer_idx, x, video_hidden, future_target_state, action_timestep
                    )
            x = self._apply_action_post_with_tracker_delta(
                block, io[3], io[4], io[5], io[6], io[7], io[8], mixed,
                tracker_delta, future_delta,
                {"context": action_pre["context"], "mask": action_pre["context_mask"]},
            )
        return x, final_future_target_tokens

    @torch.no_grad()
    def _forward_action_with_video_cache_and_current_box(
        self,
        action_tokens: torch.Tensor,
        action_pre: Dict[str, torch.Tensor],
        video_kv_cache: list[dict[str, torch.Tensor]],
        attention_mask: torch.Tensor,
        video_seq_len: int,
        box_feature: torch.Tensor,
        target_conditions: Optional[Dict[str, Optional[torch.Tensor]]] = None,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Denoise Action tokens from cached RGB K/V and the observed b0."""
        if self.current_box_action_conditioner is None:
            raise RuntimeError("Current-box Action conditioning is not enabled.")
        if self.mot is None:
            raise RuntimeError("FastWAM MoT is not initialized.")
        action_length = int(action_tokens.size(1))
        total_length = int(video_seq_len) + action_length
        action_mask = attention_mask[
            video_seq_len:total_length, :total_length
        ]
        action_expert = self.mot.mixtures["action"]
        action_context = {
            "context": action_pre["context"],
            "mask": action_pre["context_mask"],
        }
        hidden = action_tokens
        last_attention = None
        for layer_idx in range(int(self.mot.num_layers)):
            block = action_expert.blocks[layer_idx]
            action_io = self.mot._build_expert_attention_io(
                action_expert,
                block,
                hidden,
                action_pre["freqs"],
                action_pre["t_mod"],
            )
            video_cache = video_kv_cache[layer_idx]
            video_key = video_cache["k"]
            video_value = video_cache["v"]
            if video_key.size(0) != action_io[0].size(0):
                if action_io[0].size(0) % video_key.size(0) != 0:
                    raise ValueError(
                        "Action candidate batch must be an integer multiple of Video KV batch."
                    )
                repeat = action_io[0].size(0) // video_key.size(0)
                video_key = video_key.repeat_interleave(repeat, dim=0)
                video_value = video_value.repeat_interleave(repeat, dim=0)
            key_cat = torch.cat([video_key, action_io[1]], dim=1)
            value_cat = torch.cat([video_value, action_io[2]], dim=1)
            mixed = self.mot._mixed_attention(
                q_cat=action_io[0],
                k_cat=key_cat,
                v_cat=value_cat,
                attention_mask=action_mask,
            )
            if return_attention and layer_idx == int(self.mot.num_layers) - 1:
                batch_size, query_len, hidden_dim = action_io[0].shape
                num_heads = int(self.mot.num_heads)
                head_dim = hidden_dim // max(num_heads, 1)
                query_heads = action_io[0].reshape(
                    batch_size, query_len, num_heads, head_dim
                ).transpose(1, 2).float()
                key_heads = key_cat.reshape(
                    batch_size, key_cat.size(1), num_heads, head_dim
                ).transpose(1, 2).float()
                scores = torch.matmul(query_heads, key_heads.transpose(-2, -1))
                scores = scores / math.sqrt(max(head_dim, 1))
                score_mask = action_mask.to(device=scores.device, dtype=torch.bool)
                scores = scores.masked_fill(
                    ~score_mask.view(1, 1, query_len, key_cat.size(1)),
                    torch.finfo(scores.dtype).min,
                )
                last_attention = torch.softmax(scores, dim=-1)[
                    ..., :video_seq_len
                ].detach()
            if self.current_box_action_conditioner.enabled_at(layer_idx):
                action_mid = self._mot_attention_context_without_ffn(
                    block, action_io, mixed, action_context
                )
                action_mid = action_mid + self.current_box_action_conditioner.delta(
                    layer_idx, action_mid, box_feature
                )
                if self.target_action_conditioning is not None:
                    if target_conditions is None:
                        raise RuntimeError(
                            "Historical Target Memory conditions were not constructed."
                        )
                    action_mid = action_mid + self.target_action_conditioning.residual(
                        action_mid, target_conditions
                    )
                hidden = self._mot_ffn(block, action_mid, action_io)
            else:
                hidden = self.mot._apply_post_with_optional_checkpoint(
                    block=block,
                    residual_x=action_io[3],
                    gate_msa=action_io[4],
                    shift_mlp=action_io[5],
                    scale_mlp=action_io[6],
                    gate_mlp=action_io[7],
                    use_gradient_checkpointing=action_io[8],
                    mixed_slice=mixed,
                    context_payload=action_context,
                )
        if return_attention:
            if last_attention is None:
                raise RuntimeError("FastWAM Current Box attention was not captured.")
            return hidden, last_attention
        return hidden

    @torch.no_grad()
    def _forward_action_with_video_cache_and_attention(
        self,
        action_tokens: torch.Tensor,
        action_pre: Dict[str, torch.Tensor],
        video_kv_cache: list[dict[str, torch.Tensor]],
        attention_mask: torch.Tensor,
        video_seq_len: int,
        guidance_distribution: Optional[torch.Tensor] = None,
        guidance_confidence: Optional[torch.Tensor] = None,
        first_frame_tokens: int = 0,
        center_gaussian: Optional[torch.Tensor] = None,
        tracker_condition: Optional[torch.Tensor] = None,
        future_target_state: Optional[dict[str, torch.Tensor]] = None,
        action_timestep: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, dict[str, Optional[torch.Tensor]]]:
        action_seq_len = int(action_tokens.shape[1])
        total_seq_len = int(video_seq_len) + action_seq_len
        action_attention_mask = attention_mask[video_seq_len:total_seq_len, :total_seq_len]
        expert = self.mot.mixtures["action"]
        x = action_tokens
        last_attention = None
        last_raw_attention = None
        last_raw_logits = None
        last_effective_logits = None
        last_tracker_attention = None
        final_future_target_tokens = None
        for layer_idx in range(self.mot.num_layers):
            block = expert.blocks[layer_idx]
            (
                q_action,
                k_action,
                v_action,
                residual_x,
                gate_msa,
                shift_mlp,
                scale_mlp,
                gate_mlp,
                use_gradient_checkpointing,
            ) = self.mot._build_expert_attention_io(
                expert=expert,
                block=block,
                x=x,
                freqs=action_pre["freqs"],
                t_mod=action_pre["t_mod"],
            )
            layer_cache = video_kv_cache[layer_idx]
            k_video = layer_cache["k"]
            v_video = layer_cache["v"]
            k_cat = torch.cat([k_video, k_action], dim=1)
            v_cat = torch.cat([v_video, v_action], dim=1)
            guided_start = max(
                0,
                int(self.mot.num_layers) - max(int(getattr(self.cfg, "gt_center_guided_layers", 3)), 0),
            )
            if center_gaussian is not None and layer_idx >= guided_start:
                mixed, raw_logits, effective_logits, raw_attention, biased_attention = self._gt_center_biased_attention(
                    q_action, k_cat, v_cat, action_attention_mask, center_gaussian,
                    first_frame_tokens, valid_action_queries=None,
                )
                if layer_idx == self.mot.num_layers - 1:
                    last_attention = biased_attention.detach()
                    last_raw_attention = raw_attention.detach()
                    last_raw_logits = raw_logits.detach()
                    last_effective_logits = effective_logits.detach()
            elif guidance_distribution is not None and layer_idx == self.mot.num_layers - 1:
                mixed, raw_attention, biased_attention = self._biased_attention(
                    q_action, k_cat, v_cat, action_attention_mask,
                    guidance_distribution, guidance_confidence, first_frame_tokens,
                )
                last_attention = biased_attention.detach()
                last_raw_attention = raw_attention.detach()
            elif layer_idx == self.mot.num_layers - 1:
                bsz, q_len, hidden = q_action.shape
                num_heads = int(self.mot.num_heads)
                head_dim = hidden // max(num_heads, 1)
                qh = q_action.reshape(bsz, q_len, num_heads, head_dim).transpose(1, 2).float()
                kh = k_cat.reshape(bsz, total_seq_len, num_heads, head_dim).transpose(1, 2).float()
                scores = torch.matmul(qh, kh.transpose(-2, -1)) / math.sqrt(max(head_dim, 1))
                mask = action_attention_mask.to(device=scores.device, dtype=torch.bool).view(1, 1, q_len, total_seq_len)
                scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
                last_attention = torch.softmax(scores, dim=-1)[..., :video_seq_len].detach()
                last_raw_attention = last_attention
                mixed = self.mot._mixed_attention(
                    q_cat=q_action,
                    k_cat=k_cat,
                    v_cat=v_cat,
                    attention_mask=action_attention_mask,
                )

            else:
                mixed = self.mot._mixed_attention(
                    q_cat=q_action,
                    k_cat=k_cat,
                    v_cat=v_cat,
                    attention_mask=action_attention_mask,
                )
            if tracker_condition is not None:
                assert self.tracker_fusion is not None
                if not self.tracker_spatial_cross_attention:
                    tracker_delta = torch.zeros_like(x)
                elif layer_idx == self.mot.num_layers - 1:
                    tracker_delta, last_tracker_attention = self.tracker_fusion.delta(
                        layer_idx, x, tracker_condition, return_attention=True
                    )
                    if last_tracker_attention is not None:
                        last_tracker_attention = last_tracker_attention.detach()
                else:
                    tracker_delta = self.tracker_fusion.delta(layer_idx, x, tracker_condition)
            future_delta = torch.zeros_like(x)
            if future_target_state is not None:
                assert self.future_target_readout is not None and action_timestep is not None
                video_hidden = layer_cache.get("hidden")
                if video_hidden is None:
                    raise RuntimeError("Layer-aligned Future Target Readout requires cached Video hidden states.")
                memory_len = video_hidden.size(1) if first_frame_tokens <= 0 else min(first_frame_tokens, video_hidden.size(1))
                video_hidden = video_hidden[:, :memory_len]
                if layer_idx == self.mot.num_layers - 1:
                    future_delta, final_future_target_tokens = self.future_target_readout.delta(
                        layer_idx, x, video_hidden, future_target_state, action_timestep, return_tokens=True
                    )
                else:
                    future_delta = self.future_target_readout.delta(
                        layer_idx, x, video_hidden, future_target_state, action_timestep
                    )
            if tracker_condition is not None or future_target_state is not None:
                x = self._apply_action_post_with_tracker_delta(
                    block, residual_x, gate_msa, shift_mlp, scale_mlp, gate_mlp,
                    use_gradient_checkpointing, mixed, tracker_delta if tracker_condition is not None else torch.zeros_like(x),
                    future_delta,
                    {"context": action_pre["context"], "mask": action_pre["context_mask"]},
                )
            else:
                x = self.mot._apply_post_with_optional_checkpoint(
                    block=block,
                    residual_x=residual_x,
                    gate_msa=gate_msa,
                    shift_mlp=shift_mlp,
                    scale_mlp=scale_mlp,
                    gate_mlp=gate_mlp,
                    use_gradient_checkpointing=use_gradient_checkpointing,
                    mixed_slice=mixed,
                    context_payload={"context": action_pre["context"], "mask": action_pre["context_mask"]},
                )
        if last_attention is None:
            raise RuntimeError("Failed to capture last-layer transformer attention.")
        return x, {
            "raw_attention": last_raw_attention,
            "effective_attention": last_attention,
            "raw_logits": last_raw_logits,
            "effective_logits": last_effective_logits,
            "tracker_attention": last_tracker_attention,
            "final_future_target_tokens": final_future_target_tokens,
        }

    @torch.no_grad()
    def sample_action(
        self,
        first_frame_latents: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        action_horizon: int,
        num_steps: Optional[int] = None,
        return_attention_maps: bool = False,
        guidance_heatmap: Optional[torch.Tensor] = None,
        guidance_confidence: Optional[torch.Tensor] = None,
        initial_action_noise: Optional[torch.Tensor] = None,
        gt_center_xy: Optional[torch.Tensor] = None,
        tracker_features: Optional[torch.Tensor] = None,
        tracker_template_features: Optional[torch.Tensor] = None,
        tracker_bbox: Optional[torch.Tensor] = None,
        tracker_response: Optional[torch.Tensor] = None,
        tracker_confidence: Optional[torch.Tensor] = None,
        tracker_search_geometry: Optional[torch.Tensor] = None,
        tracker_image_size: Optional[torch.Tensor] = None,
        action_context: Optional[torch.Tensor] = None,
        action_context_mask: Optional[torch.Tensor] = None,
        future_video_latent_frames: int = 3,
        target_box_history: Optional[torch.Tensor] = None,
        target_box_history_valid: Optional[torch.Tensor] = None,
        previous_action: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Attention-map export is intentionally rare (normally trajectory 451)
        # and has a different return type. Keep it eager instead of paying for a
        # second several-minute compilation that would only be used once.
        compile_enabled = (
            bool(getattr(self.cfg, "compile_action_sampling", False))
            and not bool(return_attention_maps)
            and self.future_target_readout is None
            and self.state_expert is None
        )
        if not compile_enabled or self._action_compile_failed or not first_frame_latents.is_cuda:
            return self._sample_action_impl(
                first_frame_latents,
                context,
                context_mask,
                action_horizon,
                num_steps,
                return_attention_maps,
                guidance_heatmap,
                guidance_confidence,
                initial_action_noise,
                gt_center_xy,
                tracker_features,
                tracker_template_features,
                tracker_bbox,
                tracker_response,
                tracker_confidence,
                tracker_search_geometry,
                tracker_image_size,
                action_context,
                action_context_mask,
                future_video_latent_frames,
                target_box_history=target_box_history,
                target_box_history_valid=target_box_history_valid,
                previous_action=previous_action,
            )

        # Official FastWAM keeps RoPE lookup tables as plain CPU attributes,
        # which makes Inductor reject CUDA Graph capture. Sampling is pinned to
        # one inference device, so move those immutable tables before tracing.
        video_freqs = getattr(self.video_expert, "freqs", None)
        if isinstance(video_freqs, (list, tuple)) and any(
            torch.is_tensor(freq) and freq.device != first_frame_latents.device for freq in video_freqs
        ):
            self.video_expert.freqs = [freq.to(first_frame_latents.device) for freq in video_freqs]
        action_freqs = getattr(self.action_expert, "freqs", None)
        if torch.is_tensor(action_freqs) and action_freqs.device != first_frame_latents.device:
            self.action_expert.freqs = action_freqs.to(first_frame_latents.device)

        compiled = self._compiled_action_sampler
        if compiled is None:
            mode = str(getattr(self.cfg, "compile_action_sampling_mode", "reduce-overhead"))
            try:
                compiled = torch.compile(
                    self._sample_action_impl,
                    backend="inductor",
                    mode=mode,
                    fullgraph=False,
                    dynamic=False,
                )
                object.__setattr__(self, "_compiled_action_sampler", compiled)
                print(
                    f"[torch.compile] action sampler enabled mode={mode} "
                    "(reduce-overhead uses Inductor CUDA Graphs)",
                    flush=True,
                )
            except Exception as exc:
                self._action_compile_failed = True
                print(f"[torch.compile] setup failed; falling back to eager: {exc!r}", flush=True)
                return self._sample_action_impl(
                    first_frame_latents,
                    context,
                    context_mask,
                    action_horizon,
                    num_steps,
                    return_attention_maps,
                    guidance_heatmap,
                    guidance_confidence,
                    initial_action_noise,
                    gt_center_xy,
                    tracker_features,
                    tracker_template_features,
                    tracker_bbox,
                    tracker_response,
                    tracker_confidence,
                    tracker_search_geometry,
                    tracker_image_size,
                    action_context,
                    action_context_mask,
                    future_video_latent_frames,
                    target_box_history=target_box_history,
                    target_box_history_valid=target_box_history_valid,
                    previous_action=previous_action,
                )
        try:
            return compiled(
                first_frame_latents,
                context,
                context_mask,
                action_horizon,
                num_steps,
                return_attention_maps,
                guidance_heatmap,
                guidance_confidence,
                initial_action_noise,
                gt_center_xy,
                tracker_features,
                tracker_template_features,
                tracker_bbox,
                tracker_response,
                tracker_confidence,
                tracker_search_geometry,
                tracker_image_size,
                action_context,
                action_context_mask,
                future_video_latent_frames,
                target_box_history=target_box_history,
                target_box_history_valid=target_box_history_valid,
                previous_action=previous_action,
            )
        except Exception as exc:
            self._action_compile_failed = True
            object.__setattr__(self, "_compiled_action_sampler", None)
            print(f"[torch.compile] execution failed; falling back to eager: {exc!r}", flush=True)
            return self._sample_action_impl(
                first_frame_latents,
                context,
                context_mask,
                action_horizon,
                num_steps,
                return_attention_maps,
                guidance_heatmap,
                guidance_confidence,
                initial_action_noise,
                gt_center_xy,
                tracker_features,
                tracker_template_features,
                tracker_bbox,
                tracker_response,
                tracker_confidence,
                tracker_search_geometry,
                tracker_image_size,
                action_context,
                action_context_mask,
                future_video_latent_frames,
                target_box_history=target_box_history,
                target_box_history_valid=target_box_history_valid,
                previous_action=previous_action,
            )

    @torch.no_grad()
    def _sample_action_impl(
        self,
        first_frame_latents: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        action_horizon: int,
        num_steps: Optional[int] = None,
        return_attention_maps: bool = False,
        guidance_heatmap: Optional[torch.Tensor] = None,
        guidance_confidence: Optional[torch.Tensor] = None,
        initial_action_noise: Optional[torch.Tensor] = None,
        gt_center_xy: Optional[torch.Tensor] = None,
        tracker_features: Optional[torch.Tensor] = None,
        tracker_template_features: Optional[torch.Tensor] = None,
        tracker_bbox: Optional[torch.Tensor] = None,
        tracker_response: Optional[torch.Tensor] = None,
        tracker_confidence: Optional[torch.Tensor] = None,
        tracker_search_geometry: Optional[torch.Tensor] = None,
        tracker_image_size: Optional[torch.Tensor] = None,
        action_context: Optional[torch.Tensor] = None,
        action_context_mask: Optional[torch.Tensor] = None,
        future_video_latent_frames: int = 3,
        target_box_history: Optional[torch.Tensor] = None,
        target_box_history_valid: Optional[torch.Tensor] = None,
        previous_action: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if first_frame_latents.ndim != 5:
            raise ValueError("first_frame_latents must have shape [B, C, 1, H, W].")
        b = first_frame_latents.size(0)
        device = first_frame_latents.device
        steps = int(num_steps or self.cfg.action_sampling_steps)
        candidate_count = (
            int(getattr(self.cfg, "capture_value_candidate_count", 4))
            if self.capture_value_reranking_enabled
            else 1
        )
        action_batch = b * candidate_count
        if initial_action_noise is None:
            from .capture_value_reranker import sample_grouped_candidate_noise

            action = sample_grouped_candidate_noise(
                b,
                candidate_count,
                action_horizon,
                int(self.cfg.action_dim),
                device=device,
                dtype=first_frame_latents.dtype,
            )
        else:
            expected = (action_batch, int(action_horizon), int(self.cfg.action_dim))
            grouped_expected = (
                b,
                candidate_count,
                int(action_horizon),
                int(self.cfg.action_dim),
            )
            legacy_expected = (b, int(action_horizon), int(self.cfg.action_dim))
            if tuple(initial_action_noise.shape) == grouped_expected:
                initial_action_noise = initial_action_noise.flatten(0, 1)
            elif (
                candidate_count > 1
                and tuple(initial_action_noise.shape) == legacy_expected
            ):
                initial_action_noise = initial_action_noise.repeat_interleave(
                    candidate_count, dim=0
                )
            elif tuple(initial_action_noise.shape) != expected:
                raise ValueError(
                    "initial_action_noise must have shape "
                    f"{expected} or {grouped_expected}, got "
                    f"{tuple(initial_action_noise.shape)}."
                )
            action = initial_action_noise.to(device=device, dtype=first_frame_latents.dtype).clone()
        future_state_sample = None
        if self.state_expert is not None:
            future_state_sample = torch.randn(
                b,
                int(getattr(self.cfg, "future_state_horizon", 8)),
                int(getattr(self.cfg, "future_state_dim", 4)),
                device=device,
                dtype=first_frame_latents.dtype,
            )
        context = context.to(device=device, dtype=first_frame_latents.dtype)
        context_mask = context_mask.to(device=device, dtype=torch.bool)
        if action_context is None:
            if action_context_mask is not None:
                raise ValueError("action_context_mask requires action_context.")
            action_context = context
            action_context_mask = context_mask
        elif action_context_mask is None:
            raise ValueError("action_context requires action_context_mask.")
        else:
            action_context = action_context.to(
                device=device, dtype=first_frame_latents.dtype
            )
            action_context_mask = action_context_mask.to(device=device, dtype=torch.bool)
        t_video = torch.zeros(b, device=device, dtype=first_frame_latents.dtype)
        video_pre = self.video_expert.pre_dit(
            x=first_frame_latents,
            timestep=t_video,
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=True,
        )
        video_len = video_pre["tokens"].size(1)
        tracker_condition = self._make_tracker_condition(
            tracker_features,
            gt_center_xy,
            tracker_bbox,
            tracker_response,
            tracker_search_geometry,
            tracker_image_size,
            device=device,
            dtype=first_frame_latents.dtype,
        )
        future_target_state = self._make_future_target_state(
            tracker_template_features,
            tracker_condition,
            tracker_search_geometry,
            tracker_image_size,
        )
        future_state_condition = self._make_future_state_condition(
            tracker_template_features,
            tracker_condition,
            tracker_search_geometry,
            tracker_image_size,
        )
        current_box_feature = None
        current_target = None
        if self.current_box_action_conditioner is not None:
            current_target = self._make_current_target_localization(
                tracker_template_features,
                tracker_condition,
                tracker_search_geometry,
                tracker_image_size,
                tracker_bbox,
            )
            if current_target is None:
                raise RuntimeError("Online sampling did not construct Tracker b0.")
            current_box_feature = self.current_box_action_conditioner.encode_box(
                current_target["current_box"].to(
                    device=device, dtype=first_frame_latents.dtype
                )
            )
        if self.target_action_conditioning is not None:
            previous_length = self.target_action_conditioning.history_length - 1
            if target_box_history is None or target_box_history.shape != (
                b,
                previous_length,
                5,
            ):
                raise ValueError(
                    f"Online target history must be [B,{previous_length},5]."
                )
            if target_box_history_valid is None or target_box_history_valid.shape != (
                b,
                previous_length,
            ):
                raise ValueError("Online target history validity shape is invalid.")
        tracker_len = 0
        static_len = int(video_len)
        guidance_distribution = None
        prepared_confidence = None
        if bool(getattr(self.cfg, "use_fastwam_attention_bias", False)):
            grid_hw = tuple(int(v) for v in video_pre["meta"]["grid_size"][-2:])
            guidance_distribution, prepared_confidence = self._prepare_guidance_distribution(
                guidance_heatmap, guidance_confidence, grid_hw
            )
        center_gaussian = None
        if bool(getattr(self.cfg, "use_gt_center_attention_bias", False)):
            if gt_center_xy is None:
                raise RuntimeError("GT center is required for privileged-teacher action sampling.")
            grid_hw = tuple(int(v) for v in video_pre["meta"]["grid_size"][-2:])
            center_gaussian = build_center_gaussian(
                gt_center_xy,
                grid_hw[0],
                grid_hw[1],
                float(getattr(self.cfg, "gt_center_attention_sigma", 1.0)),
                visible=guidance_confidence,
            )
        cached_action_len = action_horizon
        mask = self._build_mot_attention_mask(
            video_seq_len=video_len,
            action_seq_len=cached_action_len,
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=device,
            tracker_seq_len=tracker_len,
        )
        fused_video_cache = None
        video_kv_cache = None
        target_conditions = None
        extra_target_conditions = None
        if self.target_action_conditioning is not None:
            assert current_target is not None
            assert target_box_history is not None
            assert target_box_history_valid is not None
            target_conditions = self.target_action_conditioning.build_conditions(
                previous_history=target_box_history,
                previous_valid=target_box_history_valid,
                current_box=current_target["current_box"],
                current_confidence=tracker_confidence,
            )
            extra_candidate_count = candidate_count - 1
            if extra_candidate_count > 0:
                extra_target_conditions = {
                    key: (
                        value.repeat_interleave(extra_candidate_count, dim=0)
                        if torch.is_tensor(value)
                        and value.ndim > 0
                        and value.size(0) == b
                        else value
                    )
                    for key, value in target_conditions.items()
                }
        if self.dot is not None:
            video_hidden, fused_video_cache = self.dot.prefill_video(
                video_expert=self.video_expert,
                video_pre=video_pre,
                video_attention_mask=mask[:video_len, :video_len],
                tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            )
        else:
            if self.mot is None:
                raise RuntimeError("FastWAM MoT is not initialized.")
            video_kv_cache = self.mot.prefill_video_cache(
                video_tokens=video_pre["tokens"],
                video_freqs=video_pre["freqs"],
                video_t_mod=video_pre["t_mod"],
                video_context_payload={"context": video_pre["context"], "mask": video_pre["context_mask"]},
                video_attention_mask=mask[:video_len, :video_len],
            )

        def forward_conditioned_candidate(
            candidate_pre: Dict[str, torch.Tensor],
            candidate_box_feature: torch.Tensor,
            candidate_conditions: Optional[Dict[str, Optional[torch.Tensor]]],
            *,
            return_attention: bool,
        ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
            if self.current_box_action_conditioner is None:
                raise RuntimeError("Candidate sampling requires Current Box conditioning.")
            if self.dot is not None:
                if fused_video_cache is None:
                    raise RuntimeError("FasterWAM candidate sampling requires fused Video K/V.")
                return self.dot.forward_action(
                    action_expert=self.action_expert,
                    action_pre=candidate_pre,
                    fused_video_cache=fused_video_cache,
                    condition_residual=lambda hidden: (
                        self.current_box_action_conditioner.delta(
                            0, hidden, candidate_box_feature
                        )
                        + (
                            self.target_action_conditioning.residual(
                                hidden, candidate_conditions
                            )
                            if self.target_action_conditioning is not None
                            else torch.zeros_like(hidden)
                        )
                    ),
                    return_attention=return_attention,
                )
            if video_kv_cache is None:
                raise RuntimeError("FastWAM candidate sampling requires Video K/V cache.")
            return self._forward_action_with_video_cache_and_current_box(
                action_tokens=candidate_pre["tokens"],
                action_pre=candidate_pre,
                video_kv_cache=video_kv_cache,
                attention_mask=mask,
                video_seq_len=static_len,
                box_feature=candidate_box_feature,
                target_conditions=candidate_conditions,
                return_attention=return_attention,
            )
        timesteps, deltas = self.action_scheduler.build_inference_schedule(steps, device=device, dtype=first_frame_latents.dtype)
        last_attention = None
        last_future_target_tokens = None
        last_action_input = None
        last_action_timestep = None
        last_pred_state_flow = None
        last_dot_candidate_attention = None
        for step_idx, (step_t, step_delta) in enumerate(zip(timesteps, deltas)):
            t_action = step_t.expand(action_batch)
            if step_idx == len(timesteps) - 1:
                last_action_input = action.detach().clone()
                last_action_timestep = t_action.detach().clone()
            if self.capture_value_reranking_enabled:
                if (
                    current_box_feature is None
                    or (self.dot is not None and fused_video_cache is None)
                    or (self.dot is None and video_kv_cache is None)
                ):
                    raise RuntimeError(
                        "Capture-value denoising requires shared Video K/V and b0."
                    )

                # Keep candidate zero on exactly the parent policy's [B,H,A]
                # compute path. Running it in the [B*N,H,A] candidate batch can
                # change bfloat16 GEMM kernels and therefore alter the baseline
                # action even when the reranker ultimately selects candidate zero.
                grouped_action = action.reshape(
                    b,
                    candidate_count,
                    action_horizon,
                    int(self.cfg.action_dim),
                )
                base_action = grouped_action[:, 0]
                extra_action = grouped_action[:, 1:].flatten(0, 1)
                extra_count = candidate_count - 1

                base_timestep = step_t.expand(b)
                base_pre = self.action_expert.pre_dit(
                    action_tokens=base_action,
                    timestep=base_timestep,
                    context=action_context,
                    context_mask=action_context_mask,
                )
                capture_dot_attention = bool(
                    return_attention_maps and step_idx == len(timesteps) - 1
                )
                base_out = forward_conditioned_candidate(
                    base_pre,
                    current_box_feature,
                    target_conditions,
                    return_attention=capture_dot_attention,
                )
                if capture_dot_attention:
                    base_tokens, base_attention = base_out
                else:
                    base_tokens = base_out
                base_prediction = self.action_expert.post_dit(
                    base_tokens, base_pre
                )
                base_action = self.action_scheduler.step(
                    base_prediction, step_delta, base_action
                )

                extra_timestep = step_t.expand(b * extra_count)
                extra_pre = self.action_expert.pre_dit(
                    action_tokens=extra_action,
                    timestep=extra_timestep,
                    context=action_context,
                    context_mask=action_context_mask,
                )
                extra_box_feature = current_box_feature.repeat_interleave(
                    extra_count, dim=0
                )
                extra_out = forward_conditioned_candidate(
                    extra_pre,
                    extra_box_feature,
                    extra_target_conditions,
                    return_attention=capture_dot_attention,
                )
                if capture_dot_attention:
                    extra_tokens, extra_attention = extra_out
                    last_dot_candidate_attention = torch.cat(
                        [
                            base_attention[:, None],
                            extra_attention.reshape(
                                b,
                                extra_count,
                                *extra_attention.shape[1:],
                            ),
                        ],
                        dim=1,
                    )
                else:
                    extra_tokens = extra_out
                extra_prediction = self.action_expert.post_dit(
                    extra_tokens, extra_pre
                )
                extra_action = self.action_scheduler.step(
                    extra_prediction, step_delta, extra_action
                )
                action = torch.cat(
                    [
                        base_action[:, None],
                        extra_action.reshape(
                            b,
                            extra_count,
                            action_horizon,
                            int(self.cfg.action_dim),
                        ),
                    ],
                    dim=1,
                ).flatten(0, 1)
                continue
            action_pre = self.action_expert.pre_dit(
                action_tokens=action,
                timestep=t_action,
                context=action_context,
                context_mask=action_context_mask,
            )
            state_pre = None
            if self.state_expert is not None:
                if future_state_sample is None or future_state_condition is None:
                    raise RuntimeError("Online Future State inputs were not initialized.")
                state_pre = self.state_expert.pre_dit(
                    noisy_future_states=future_state_sample,
                    current_condition=future_state_condition["current_condition"],
                    timestep=t_action,
                    tracker_memory=future_state_condition["tracker_memory"],
                    state_valid_mask=torch.ones(
                        b,
                        int(getattr(self.cfg, "future_state_horizon", 8)) + 1,
                        device=device,
                        dtype=torch.bool,
                    ),
                )
            if action_pre["tokens"].size(1) != cached_action_len:
                mask = self._build_mot_attention_mask(
                    video_seq_len=video_len,
                    action_seq_len=action_pre["tokens"].size(1),
                    video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
                    device=device,
                    tracker_seq_len=tracker_len,
                )
                cached_action_len = action_pre["tokens"].size(1)
            if self.dot is not None:
                if fused_video_cache is None:
                    raise RuntimeError("Faster-WAM Video KV cache was not initialized.")
                capture_dot_attention = bool(
                    return_attention_maps and step_idx == len(timesteps) - 1
                )
                dot_out = self.dot.forward_action(
                    action_expert=self.action_expert,
                    action_pre=action_pre,
                    fused_video_cache=fused_video_cache,
                    condition_residual=(
                        None
                        if self.current_box_action_conditioner is None
                        else lambda hidden: (
                            self.current_box_action_conditioner.delta(
                                0,
                                hidden,
                                current_box_feature.repeat_interleave(
                                    candidate_count, dim=0
                                ),
                            )
                            + (
                                self.target_action_conditioning.residual(
                                    hidden,
                                    target_conditions,
                                )
                                if self.target_action_conditioning is not None
                                else torch.zeros_like(hidden)
                            )
                        )
                    ),
                    return_attention=capture_dot_attention,
                )
                if capture_dot_attention:
                    action_tokens, dot_attention = dot_out
                    last_attention = {
                        "effective_attention": dot_attention,
                        "raw_attention": dot_attention,
                        "raw_logits": None,
                        "effective_logits": None,
                        "tracker_attention": None,
                    }
                else:
                    action_tokens = dot_out
            elif self.current_box_action_conditioner is not None:
                if video_kv_cache is None or current_box_feature is None:
                    raise RuntimeError(
                        "Current-box sampling cache was not initialized."
                    )
                capture_current_box_attention = bool(
                    return_attention_maps and step_idx == len(timesteps) - 1
                )
                current_box_out = self._forward_action_with_video_cache_and_current_box(
                    action_tokens=action_pre["tokens"],
                    action_pre=action_pre,
                    video_kv_cache=video_kv_cache,
                    attention_mask=mask,
                    video_seq_len=static_len,
                    box_feature=current_box_feature,
                    target_conditions=target_conditions,
                    return_attention=capture_current_box_attention,
                )
                if capture_current_box_attention:
                    action_tokens, current_box_attention = current_box_out
                    last_attention = {
                        "effective_attention": current_box_attention,
                        "raw_attention": current_box_attention,
                        "raw_logits": None,
                        "effective_logits": None,
                        "tracker_attention": None,
                    }
                else:
                    action_tokens = current_box_out
            elif self.state_expert is not None:
                if video_kv_cache is None:
                    raise RuntimeError("FastWAM Video KV cache was not initialized.")
                assert state_pre is not None
                joint_out = self._forward_three_expert_layers(
                    video_tokens=None,
                    video_pre=video_pre,
                    action_pre=action_pre,
                    state_pre=state_pre,
                    base_attention_mask=mask,
                    action_valid_mask=None,
                    video_kv_cache=video_kv_cache,
                )
                action_tokens = joint_out["action"]
                state_tokens = joint_out["state"]
                last_pred_state_flow = self.state_expert.post_dit(state_tokens, state_pre)
                assert self.state_scheduler is not None
                future_state_sample = self.state_scheduler.step(
                    last_pred_state_flow, step_delta, future_state_sample
                )
            elif center_gaussian is not None or guidance_distribution is not None or (return_attention_maps and step_idx == len(timesteps) - 1):
                action_tokens, last_attention = self._forward_action_with_video_cache_and_attention(
                    action_tokens=action_pre["tokens"],
                    action_pre=action_pre,
                    video_kv_cache=video_kv_cache,
                    attention_mask=mask,
                    video_seq_len=static_len,
                    guidance_distribution=guidance_distribution,
                    guidance_confidence=prepared_confidence,
                    first_frame_tokens=int(video_pre["meta"]["tokens_per_frame"]),
                    center_gaussian=center_gaussian,
                    tracker_condition=tracker_condition,
                    future_target_state=future_target_state,
                    action_timestep=t_action,
                )
                last_future_target_tokens = last_attention.get("final_future_target_tokens")
            elif tracker_condition is not None:
                action_tokens, last_future_target_tokens = self._forward_action_with_tracker_fusion(
                    action_tokens=action_pre["tokens"],
                    action_pre=action_pre,
                    video_kv_cache=video_kv_cache,
                    attention_mask=mask,
                    video_seq_len=static_len,
                    tracker_condition=tracker_condition,
                    future_target_state=future_target_state,
                    action_timestep=t_action,
                    first_frame_tokens=int(video_pre["meta"]["tokens_per_frame"]),
                )
            else:
                if self.mot is None or video_kv_cache is None:
                    raise RuntimeError("FastWAM MoT Video KV cache was not initialized.")
                action_tokens = self.mot.forward_action_with_video_cache(
                    action_tokens=action_pre["tokens"],
                    action_freqs=action_pre["freqs"],
                    action_t_mod=action_pre["t_mod"],
                    action_context_payload={"context": action_pre["context"], "mask": action_pre["context_mask"]},
                    video_kv_cache=video_kv_cache,
                    attention_mask=mask,
                    video_seq_len=static_len,
                )
            pred_action = self.action_expert.post_dit(action_tokens, action_pre)
            action = self.action_scheduler.step(pred_action, step_delta, action)
        action = action.clamp(-1.0, 1.0)
        capture_value_aux = None
        if self.capture_value_reranking_enabled:
            if (
                current_box_feature is None
                or current_target is None
                or (self.dot is not None and fused_video_cache is None)
                or (self.dot is None and video_kv_cache is None)
            ):
                raise RuntimeError(
                    "Capture-value sampling requires model Video K/V and current b0."
                )
            candidates = action.reshape(
                b, candidate_count, action_horizon, int(self.cfg.action_dim)
            )
            if bool(
                getattr(self.cfg, "capture_value_structured_candidates", False)
            ):
                from .capture_value_reranker import (
                    build_structured_recenter_candidates,
                )

                candidates = build_structured_recenter_candidates(
                    candidates,
                    current_target["current_box"],
                )
            if self.capture_value_score_mode == "learned":
                if self.capture_value_head is None:
                    raise RuntimeError("Learned Capture-Value Head is not initialized.")
                if fused_video_cache is None:
                    raise RuntimeError("Learned Capture-Value scoring requires DoT Video K/V.")
                video_context = torch.cat(
                    [
                        fused_video_cache["canonical_key"].mean(dim=2),
                        fused_video_cache["value"].mean(dim=2),
                    ],
                    dim=-1,
                ).mean(dim=0)
                value_prediction = self.capture_value_head(
                    candidates,
                    video_context=video_context,
                    target_context=current_box_feature,
                )
                selected_index = value_prediction["score"].argmax(dim=1)
                selected_scalar_index = selected_index[:, None]
                capture_value_aux = {
                    "capture_value_candidates": candidates,
                    "capture_value_scores": value_prediction["score"],
                    "capture_value_selected_index": selected_index,
                    "capture_value_selected_capture_probability": torch.sigmoid(
                        value_prediction["capture_logit"]
                    ).gather(1, selected_scalar_index).squeeze(1),
                    "capture_value_selected_final_distance": (
                        value_prediction["final_distance_normalized"]
                        * float(getattr(self.cfg, "capture_value_capture_distance", 10.0))
                    ).gather(1, selected_scalar_index).squeeze(1),
                    "capture_value_selected_visibility": torch.sigmoid(
                        value_prediction["visibility_logit"]
                    ).gather(1, selected_scalar_index).squeeze(1),
                }
            elif self.capture_value_score_mode == "action_prior":
                if self.capture_action_prior is None:
                    raise RuntimeError("CaptureActionPrior is not initialized.")
                if target_box_history is None or target_box_history_valid is None:
                    raise RuntimeError("CaptureActionPrior requires target box history.")
                from .capture_action_prior import (
                    featurize_capture_action_context,
                    score_candidates_with_action_prior,
                )
                from .capture_value_reranker import select_geometric_candidate

                prior_features = featurize_capture_action_context(
                    target_box_history,
                    target_box_history_valid,
                    current_target["current_box"],
                    tracker_confidence,
                    previous_action,
                )
                prior_prediction = self.capture_action_prior(prior_features)
                dimension_weights = candidates.new_tensor(
                    getattr(
                        self.cfg,
                        "capture_action_prior_dimension_weights",
                        (0.0, 1.0, 0.0, 0.0),
                    )
                )
                value_scores = score_candidates_with_action_prior(
                    candidates,
                    prior_prediction,
                    dimension_weights=dimension_weights,
                )
                normalized_center_error = torch.linalg.vector_norm(
                    2.0 * (current_target["current_box"][:, :2].float() - 0.5),
                    dim=-1,
                )
                min_center_error = float(
                    getattr(self.cfg, "capture_value_min_center_error", 0.5)
                )
                selection = select_geometric_candidate(
                    value_scores,
                    float(getattr(self.cfg, "capture_value_selection_margin", 1.0)),
                    allow_switch=normalized_center_error >= min_center_error,
                )
                selected_index = selection["selected_index"]
                capture_value_aux = {
                    "capture_value_candidates": candidates,
                    "capture_value_scores": value_scores,
                    "capture_value_selected_index": selected_index,
                    "capture_value_raw_selected_index": selection[
                        "raw_selected_index"
                    ],
                    "capture_value_score_advantage": selection["score_advantage"],
                    "capture_value_used_fallback": selection["used_fallback"],
                    "capture_value_center_error": normalized_center_error,
                    "capture_value_switch_allowed": (
                        normalized_center_error >= min_center_error
                    ),
                    "capture_action_prior_mean": prior_prediction["mean"],
                    "capture_action_prior_std": prior_prediction["std"],
                }
            else:
                from .capture_value_reranker import (
                    score_geometric_capture_trajectories,
                    select_geometric_candidate,
                )

                value_prediction = score_geometric_capture_trajectories(
                    candidates,
                    current_target["current_box"],
                    previous_action,
                    target_box_history,
                    target_box_history_valid,
                    max_vel=float(getattr(self.cfg, "max_vel", 1.0)),
                    max_yaw_rate=float(getattr(self.cfg, "max_yaw_rate", 15.0)),
                    max_speed_norm=float(getattr(self.cfg, "max_speed_norm", 1.0)),
                    control_dt=float(getattr(self.cfg, "capture_value_control_dt", 1.0)),
                    horizontal_fov_deg=float(
                        getattr(self.cfg, "capture_value_horizontal_fov_deg", 90.0)
                    ),
                    vertical_fov_deg=float(
                        getattr(self.cfg, "capture_value_vertical_fov_deg", 90.0)
                    ),
                    depth_scale=float(
                        getattr(self.cfg, "capture_value_bbox_depth_scale", 0.2698)
                    ),
                    min_depth=float(getattr(self.cfg, "capture_value_min_depth", 1.0)),
                    max_depth=float(getattr(self.cfg, "capture_value_max_depth", 20.0)),
                    target_box_size=float(
                        getattr(self.cfg, "capture_value_target_box_size", 0.06094)
                    ),
                    box_size_sigma=float(
                        getattr(self.cfg, "capture_value_box_size_sigma", 0.01)
                    ),
                    discount=float(getattr(self.cfg, "capture_value_discount", 0.8)),
                    recenter_sigma=float(
                        getattr(self.cfg, "capture_value_recenter_sigma", 0.35)
                    ),
                    pursuit_center_sigma=float(
                        getattr(self.cfg, "capture_value_pursuit_center_sigma", 0.40)
                    ),
                    out_of_frame_weight=float(
                        getattr(self.cfg, "capture_value_out_of_frame_weight", 2.0)
                    ),
                    first_action_smooth_weight=float(
                        getattr(self.cfg, "capture_value_first_action_smooth_weight", 2.0)
                    ),
                    temporal_smooth_weight=float(
                        getattr(self.cfg, "capture_value_temporal_smooth_weight", 1.0)
                    ),
                    recenter_weight=float(
                        getattr(self.cfg, "capture_value_recenter_weight", 2.0)
                    ),
                    pursuit_weight=float(
                        getattr(self.cfg, "capture_value_pursuit_weight", 0.7)
                    ),
                    smooth_weight=float(
                        getattr(self.cfg, "capture_value_smooth_weight", 0.1)
                    ),
                    consensus_weight=float(
                        getattr(self.cfg, "capture_value_consensus_weight", 0.1)
                    ),
                    short_horizon=int(
                        getattr(self.cfg, "capture_value_short_horizon", 1)
                    ),
                )
                selection_margin = float(
                    getattr(self.cfg, "capture_value_selection_margin", 0.03)
                )
                normalized_center_error = torch.linalg.vector_norm(
                    2.0
                    * (
                        current_target["current_box"][:, :2].float()
                        - 0.5
                    ),
                    dim=-1,
                )
                min_center_error = float(
                    getattr(self.cfg, "capture_value_min_center_error", 0.5)
                )
                selection = select_geometric_candidate(
                    value_prediction["score"],
                    selection_margin,
                    allow_switch=normalized_center_error >= min_center_error,
                )
                raw_selected_index = selection["raw_selected_index"]
                score_advantage = selection["score_advantage"]
                used_fallback = selection["used_fallback"]
                selected_index = selection["selected_index"]
                selected_scalar_index = selected_index[:, None]
                selected_step_index = selected_index[:, None, None, None].expand(
                    -1, 1, action_horizon, 2
                )
                selected_center_error = value_prediction[
                    "predicted_center_error"
                ].gather(1, selected_step_index).squeeze(1)
                capture_value_aux = {
                    "capture_value_candidates": candidates,
                    "capture_value_scores": value_prediction["score"],
                    "capture_value_selected_index": selected_index,
                    "capture_value_raw_selected_index": raw_selected_index,
                    "capture_value_score_advantage": score_advantage,
                    "capture_value_used_fallback": used_fallback,
                    "capture_value_center_error": normalized_center_error,
                    "capture_value_switch_allowed": (
                        normalized_center_error >= min_center_error
                    ),
                    "capture_value_recenter_costs": value_prediction["recenter_cost"],
                    "capture_value_pursuit_costs": value_prediction["pursuit_cost"],
                    "capture_value_smooth_costs": value_prediction["smooth_cost"],
                    "capture_value_consensus_costs": value_prediction[
                        "consensus_cost"
                    ],
                    "capture_value_observed_center_velocity": value_prediction[
                        "observed_center_velocity"
                    ],
                    "capture_value_selected_final_center_error": selected_center_error[:, -1],
                    "capture_value_selected_final_box_size": value_prediction[
                        "predicted_box_size"
                    ].gather(1, selected_index[:, None, None].expand(-1, 1, action_horizon))
                    .squeeze(1)[:, -1],
                }
            gather_index = selected_index[:, None, None, None].expand(
                -1, 1, action_horizon, int(self.cfg.action_dim)
            )
            action = candidates.gather(1, gather_index).squeeze(1)
            if last_dot_candidate_attention is not None:
                attention_index = selected_index[:, None, None, None, None].expand(
                    -1,
                    1,
                    *last_dot_candidate_attention.shape[2:],
                )
                selected_attention = last_dot_candidate_attention.gather(
                    1, attention_index
                ).squeeze(1)
                last_attention = {
                    "effective_attention": selected_attention,
                    "raw_attention": selected_attention,
                    "raw_logits": None,
                    "effective_logits": None,
                    "tracker_attention": None,
                }
        predicted_center_flow = None
        predicted_state_centers = None
        future_target_centers = None
        predicted_state_boxes = None
        future_target_boxes = None
        pred_state_flow = None
        if self.current_box_action_conditioner is not None:
            current_box = current_target["current_box"].to(
                device=action.device, dtype=action.dtype
            )
            predicted_state_boxes = current_box[:, None]
            predicted_state_centers = predicted_state_boxes[..., :2]
        if self.state_expert is not None:
            if future_state_sample is None or future_state_condition is None or last_pred_state_flow is None:
                raise RuntimeError("Joint Action/State denoising did not produce Future States.")
            assert self.future_state_conditioner is not None
            predicted_state_boxes = self.future_state_conditioner.decode_relative_states(
                future_state_condition["current_box"], future_state_sample
            )
            predicted_state_centers = predicted_state_boxes[..., :2]
            future_target_centers = predicted_state_centers[:, 1:]
            future_target_boxes = predicted_state_boxes[:, 1:]
            pred_state_flow = last_pred_state_flow
        if future_target_state is not None:
            assert self.future_target_readout is not None
            if last_future_target_tokens is None:
                raise RuntimeError("Final Future Target tokens were not produced during Action sampling.")
            predicted_state_boxes, predicted_center_flow, _ = self.future_target_readout.state_boxes(
                last_future_target_tokens, future_target_state
            )
            predicted_state_centers = predicted_state_boxes[..., :2]
            future_target_centers = predicted_state_centers[:, 1:]
            future_target_boxes = predicted_state_boxes[:, 1:]
        if return_attention_maps:
            grid_size = tuple(int(x) for x in video_pre["meta"]["grid_size"])
            auxiliary = {
                "last_transformer_attention": None if last_attention is None else last_attention["effective_attention"],
                "last_transformer_raw_attention": None if last_attention is None else last_attention["raw_attention"],
                "last_transformer_raw_attention_logits": None if last_attention is None else last_attention["raw_logits"],
                "last_transformer_effective_attention_logits": None if last_attention is None else last_attention["effective_logits"],
                "last_tracker_cross_attention": None if last_attention is None else last_attention["tracker_attention"],
                "last_action_input": last_action_input,
                "last_action_timestep": last_action_timestep,
                "video_grid_size": grid_size,
                "future_target_centers": future_target_centers,
                "target_state_centers": predicted_state_centers,
                "pred_center_flow": predicted_center_flow,
                "future_target_boxes": future_target_boxes,
                "target_state_boxes": predicted_state_boxes,
                "pred_state_flow": pred_state_flow,
            }
            if capture_value_aux is not None:
                auxiliary.update(capture_value_aux)
            return action, auxiliary
        if predicted_state_centers is not None:
            auxiliary = {
                "future_target_centers": future_target_centers,
                "target_state_centers": predicted_state_centers,
                "pred_center_flow": predicted_center_flow,
                "future_target_boxes": future_target_boxes,
                "target_state_boxes": predicted_state_boxes,
                "pred_state_flow": pred_state_flow,
            }
            if capture_value_aux is not None:
                auxiliary.update(capture_value_aux)
            return action, auxiliary
        if capture_value_aux is not None:
            return action, capture_value_aux
        return action
