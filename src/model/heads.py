from __future__ import annotations

import math
import os
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .action_loss_utils import weighted_mean_action_squared_error
from .config import ModelConfig
from .encoders import MLP, _ensure_fastwam_path, _torch_dtype_from_name


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
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
        self.video_expert = self._load_official_video_expert(cfg)
        self.action_expert = self._load_official_action_expert(cfg)
        if int(self.action_expert.num_heads) != int(self.video_expert.num_heads):
            raise ValueError("Official ActionDiT num_heads must match WanVideoDiT for MoT.")
        if int(self.action_expert.attn_head_dim) != int(self.video_expert.attn_head_dim):
            raise ValueError("Official ActionDiT attn_head_dim must match WanVideoDiT for MoT.")
        if int(len(self.action_expert.blocks)) != int(len(self.video_expert.blocks)):
            raise ValueError("Official ActionDiT num_layers must match WanVideoDiT.")
        _ensure_fastwam_path(cfg)
        from fastwam.models.wan22.mot import MoT

        self.mot = MoT(
            mixtures={"video": self.video_expert, "action": self.action_expert},
            mot_checkpoint_mixed_attn=bool(getattr(cfg, "fastwam_mot_checkpoint_mixed_attn", True)),
        )
        self.video_scheduler = FlowMatchScheduler(cfg.fastwam_video_train_timesteps, cfg.fastwam_video_shift)
        self.action_scheduler = FlowMatchScheduler(cfg.fastwam_action_train_timesteps, cfg.fastwam_action_shift)

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

    def training_loss(
        self,
        video_latents: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        expert_action: torch.Tensor,
        valid_mask: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        if video_latents.ndim != 5:
            raise ValueError("video_latents must have shape [B, C, T_lat, H_lat, W_lat].")
        b = video_latents.size(0)
        action_tokens_clean, action_valid_mask = self._make_training_action_tokens(expert_action, valid_mask)
        action_tokens_clean = action_tokens_clean.to(device=video_latents.device, dtype=video_latents.dtype)
        action_token_len = int(action_tokens_clean.size(1))
        latent_transitions = max(int(video_latents.size(2)) - 1, 1)
        if action_token_len % latent_transitions != 0:
            raise ValueError(
                "FastWAM action token length must be divisible by Wan latent transitions; "
                f"got action_len={action_token_len}, latent_frames={int(video_latents.size(2))}."
            )

        noise_video = torch.randn_like(video_latents)
        t_video = self.video_scheduler.sample_training_t(b, video_latents.device, video_latents.dtype)
        noisy_video = self.video_scheduler.add_noise(video_latents, noise_video, t_video)
        target_video = self.video_scheduler.training_target(video_latents, noise_video, t_video)
        first_frame_latents = video_latents[:, :, 0:1].clone()
        noisy_video[:, :, 0:1] = first_frame_latents

        noise_action = torch.randn_like(action_tokens_clean)
        t_action = self.action_scheduler.sample_training_t(b, action_tokens_clean.device, action_tokens_clean.dtype)
        noisy_action = self.action_scheduler.add_noise(action_tokens_clean, noise_action, t_action)
        target_action = self.action_scheduler.training_target(action_tokens_clean, noise_action, t_action)

        context = context.to(device=video_latents.device, dtype=video_latents.dtype)
        context_mask = context_mask.to(device=video_latents.device, dtype=torch.bool)
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
            context=context,
            context_mask=context_mask,
        )
        mask = self._build_mot_attention_mask(
            video_seq_len=video_pre["tokens"].size(1),
            action_seq_len=action_pre["tokens"].size(1),
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_latents.device,
        )
        out = self.mot(
            embeds_all={"video": video_pre["tokens"], "action": action_pre["tokens"]},
            attention_mask=mask,
            freqs_all={"video": video_pre["freqs"], "action": action_pre["freqs"]},
            context_all={
                "video": {"context": video_pre["context"], "mask": video_pre["context_mask"]},
                "action": {"context": action_pre["context"], "mask": action_pre["context_mask"]},
            },
            t_mod_all={"video": video_pre["t_mod"], "action": action_pre["t_mod"]},
        )
        pred_video = self.video_expert.post_dit(out["video"], video_pre)
        pred_action = self.action_expert.post_dit(out["action"], action_pre)

        pred_video = pred_video[:, :, 1:]
        target_video = target_video[:, :, 1:]
        video_loss_token = F.mse_loss(pred_video.float(), target_video.float(), reduction="none").mean(dim=(1, 3, 4))
        video_loss_per_sample = video_loss_token.mean(dim=1)
        video_weight = self.video_scheduler.training_weight(t_video).to(video_loss_per_sample.device, video_loss_per_sample.dtype)
        loss_video = (video_loss_per_sample * video_weight).mean()

        action_loss_token = F.mse_loss(pred_action.float(), target_action.float(), reduction="none").mean(dim=2)
        if action_valid_mask is not None:
            valid = action_valid_mask.to(device=action_loss_token.device, dtype=action_loss_token.dtype)
            action_loss_per_sample = (action_loss_token * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)
        else:
            action_loss_per_sample = action_loss_token.mean(dim=1)
        action_weight = self.action_scheduler.training_weight(t_action).to(action_loss_per_sample.device, action_loss_per_sample.dtype)
        loss_action = (action_loss_per_sample * action_weight).mean()

        return {
            "loss_video": loss_video,
            "loss_action": loss_action,
            "pred_action": pred_action.reshape(b, action_token_len, 1, self.cfg.action_dim),
        }

    @torch.no_grad()
    def _build_mot_attention_mask(
        self,
        video_seq_len: int,
        action_seq_len: int,
        video_tokens_per_frame: int,
        device: torch.device,
    ) -> torch.Tensor:
        total_seq_len = video_seq_len + action_seq_len
        mask = torch.zeros((total_seq_len, total_seq_len), dtype=torch.bool, device=device)
        mask[:video_seq_len, :video_seq_len] = self.video_expert.build_video_to_video_mask(
            video_seq_len=video_seq_len,
            video_tokens_per_frame=video_tokens_per_frame,
            device=device,
        )
        mask[video_seq_len:, video_seq_len:] = True
        first_frame_tokens = min(video_tokens_per_frame, video_seq_len)
        mask[video_seq_len:, :first_frame_tokens] = True
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

    @torch.no_grad()
    def _forward_action_with_video_cache_and_attention(
        self,
        action_tokens: torch.Tensor,
        action_pre: Dict[str, torch.Tensor],
        video_kv_cache: list[dict[str, torch.Tensor]],
        attention_mask: torch.Tensor,
        video_seq_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        action_seq_len = int(action_tokens.shape[1])
        total_seq_len = int(video_seq_len) + action_seq_len
        action_attention_mask = attention_mask[video_seq_len:total_seq_len, :total_seq_len]
        expert = self.mot.mixtures["action"]
        x = action_tokens
        last_attention = None
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
            if layer_idx == self.mot.num_layers - 1:
                bsz, q_len, hidden = q_action.shape
                num_heads = int(self.mot.num_heads)
                head_dim = hidden // max(num_heads, 1)
                qh = q_action.reshape(bsz, q_len, num_heads, head_dim).transpose(1, 2).float()
                kh = k_cat.reshape(bsz, total_seq_len, num_heads, head_dim).transpose(1, 2).float()
                scores = torch.matmul(qh, kh.transpose(-2, -1)) / math.sqrt(max(head_dim, 1))
                mask = action_attention_mask.to(device=scores.device, dtype=torch.bool).view(1, 1, q_len, total_seq_len)
                scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
                last_attention = torch.softmax(scores, dim=-1)[..., :video_seq_len].detach()

            mixed = self.mot._mixed_attention(
                q_cat=q_action,
                k_cat=k_cat,
                v_cat=v_cat,
                attention_mask=action_attention_mask,
            )
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
        return x, last_attention

    @torch.no_grad()
    def sample_action(
        self,
        first_frame_latents: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        action_horizon: int,
        num_steps: Optional[int] = None,
        return_attention_maps: bool = False,
    ) -> torch.Tensor:
        if first_frame_latents.ndim != 5:
            raise ValueError("first_frame_latents must have shape [B, C, 1, H, W].")
        b = first_frame_latents.size(0)
        device = first_frame_latents.device
        steps = int(num_steps or self.cfg.action_sampling_steps)
        action = torch.randn(b, action_horizon, self.cfg.action_dim, device=device, dtype=first_frame_latents.dtype)
        context = context.to(device=device, dtype=first_frame_latents.dtype)
        context_mask = context_mask.to(device=device, dtype=torch.bool)
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
        cached_action_len = action_horizon
        mask = self._build_mot_attention_mask(
            video_seq_len=video_len,
            action_seq_len=cached_action_len,
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=device,
        )
        video_kv_cache = self.mot.prefill_video_cache(
            video_tokens=video_pre["tokens"],
            video_freqs=video_pre["freqs"],
            video_t_mod=video_pre["t_mod"],
            video_context_payload={"context": video_pre["context"], "mask": video_pre["context_mask"]},
            video_attention_mask=mask[:video_len, :video_len],
        )
        timesteps, deltas = self.action_scheduler.build_inference_schedule(steps, device=device, dtype=first_frame_latents.dtype)
        last_attention = None
        for step_idx, (step_t, step_delta) in enumerate(zip(timesteps, deltas)):
            t_action = step_t.expand(b)
            action_pre = self.action_expert.pre_dit(
                action_tokens=action,
                timestep=t_action,
                context=context,
                context_mask=context_mask,
            )
            if action_pre["tokens"].size(1) != cached_action_len:
                mask = self._build_mot_attention_mask(
                    video_seq_len=video_len,
                    action_seq_len=action_pre["tokens"].size(1),
                    video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
                    device=device,
                )
                cached_action_len = action_pre["tokens"].size(1)
            if return_attention_maps and step_idx == len(timesteps) - 1:
                action_tokens, last_attention = self._forward_action_with_video_cache_and_attention(
                    action_tokens=action_pre["tokens"],
                    action_pre=action_pre,
                    video_kv_cache=video_kv_cache,
                    attention_mask=mask,
                    video_seq_len=video_len,
                )
            else:
                action_tokens = self.mot.forward_action_with_video_cache(
                    action_tokens=action_pre["tokens"],
                    action_freqs=action_pre["freqs"],
                    action_t_mod=action_pre["t_mod"],
                    action_context_payload={"context": action_pre["context"], "mask": action_pre["context_mask"]},
                    video_kv_cache=video_kv_cache,
                    attention_mask=mask,
                    video_seq_len=video_len,
                )
            pred_action = self.action_expert.post_dit(action_tokens, action_pre)
            action = self.action_scheduler.step(pred_action, step_delta, action)
        action = action.clamp(-1.0, 1.0)
        if return_attention_maps:
            grid_size = tuple(int(x) for x in video_pre["meta"]["grid_size"])
            return action, {"last_transformer_attention": last_attention, "video_grid_size": grid_size}
        return action
