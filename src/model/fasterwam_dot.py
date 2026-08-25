from __future__ import annotations

import math
from typing import Any, Callable, Dict, Optional

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as gradient_checkpoint

from fastwam.models.wan22.wan_video_dit import (
    DiTBlock,
    flash_attention,
    modulate,
    precompute_freqs_cis,
    rope_apply,
    sinusoidal_embedding_1d,
)


class FasterWAMActionHead(nn.Module):
    """Single-layer action expert used by Faster-WAM's DoT module."""

    def __init__(
        self,
        *,
        action_dim: int,
        hidden_dim: int = 1024,
        ffn_dim: int = 4096,
        freq_dim: int = 256,
        num_heads: int = 24,
        attn_head_dim: int = 128,
        eps: float = 1.0e-6,
        use_gradient_checkpointing: bool = True,
    ) -> None:
        super().__init__()
        self.action_dim = int(action_dim)
        self.hidden_dim = int(hidden_dim)
        self.ffn_dim = int(ffn_dim)
        self.freq_dim = int(freq_dim)
        self.num_heads = int(num_heads)
        self.attn_head_dim = int(attn_head_dim)
        self.use_gradient_checkpointing = bool(use_gradient_checkpointing)

        if self.num_heads <= 0 or self.attn_head_dim <= 0:
            raise ValueError("Action attention dimensions must be positive.")
        if self.attn_head_dim % 2 != 0:
            raise ValueError("Action attention head dimension must be even for RoPE.")

        self.action_encoder = nn.Linear(self.action_dim, self.hidden_dim)
        self.time_embedding = nn.Sequential(
            nn.Linear(self.freq_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.time_projection = nn.Sequential(
            nn.SiLU(), nn.Linear(self.hidden_dim, self.hidden_dim * 6)
        )
        block = DiTBlock(
            hidden_dim=self.hidden_dim,
            attn_head_dim=self.attn_head_dim,
            num_heads=self.num_heads,
            ffn_dim=self.ffn_dim,
            eps=float(eps),
        )
        # Language reaches the action head only through the Video DiT hub.
        # Removing these modules also avoids unused DDP parameters.
        block.cross_attn = nn.Identity()
        block.norm3 = nn.Identity()
        self.blocks = nn.ModuleList([block])
        self.head = nn.Linear(self.hidden_dim, self.action_dim)
        self.freqs = precompute_freqs_cis(self.attn_head_dim, end=1024)

    def pre_dit(
        self,
        action_tokens: torch.Tensor,
        timestep: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        del context, context_mask
        if action_tokens.ndim != 3 or action_tokens.size(-1) != self.action_dim:
            raise ValueError(
                "action_tokens must have shape "
                f"[B,T,{self.action_dim}], got {tuple(action_tokens.shape)}."
            )
        batch_size, seq_len = action_tokens.shape[:2]
        if timestep.ndim != 1 or timestep.size(0) not in (1, batch_size):
            raise ValueError("timestep must have shape [1] or [B].")
        if timestep.size(0) == 1 and batch_size > 1:
            if self.training:
                raise ValueError("Training timesteps must match the batch size.")
            timestep = timestep.expand(batch_size)
        if seq_len > self.freqs.size(0):
            raise ValueError(
                f"Action sequence length {seq_len} exceeds RoPE cache {self.freqs.size(0)}."
            )

        tokens = self.action_encoder(action_tokens)
        t = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, timestep))
        t_mod = self.time_projection(t).unflatten(1, (6, self.hidden_dim))
        freqs = self.freqs[:seq_len].view(seq_len, 1, -1).to(tokens.device)
        return {
            "tokens": tokens,
            "freqs": freqs,
            "t": t,
            "t_mod": t_mod,
            "context": None,
            "context_mask": None,
            "meta": {"batch_size": batch_size, "seq_len": seq_len},
        }

    def post_dit(
        self, tokens: torch.Tensor, pre_state: Optional[Dict[str, Any]] = None
    ) -> torch.Tensor:
        del pre_state
        return self.head(tokens)


class FasterWAMDoT(nn.Module):
    """Cross-layer KV fusion with Video-to-Action RoPE alignment."""

    def __init__(
        self,
        *,
        video_num_layers: int,
        num_action_layers: int,
        num_heads: int,
        attn_head_dim: int,
        use_gradient_checkpointing: bool = True,
    ) -> None:
        super().__init__()
        self.video_num_layers = int(video_num_layers)
        self.num_action_layers = int(num_action_layers)
        self.num_heads = int(num_heads)
        self.attn_head_dim = int(attn_head_dim)
        self.attention_dim = self.num_heads * self.attn_head_dim
        self.use_gradient_checkpointing = bool(use_gradient_checkpointing)
        if min(
            self.video_num_layers,
            self.num_action_layers,
            self.num_heads,
            self.attn_head_dim,
        ) <= 0:
            raise ValueError("DoT dimensions must be positive.")

        self.key_channel_mixer = nn.Linear(
            self.attention_dim, self.attention_dim, bias=False
        )
        self.value_channel_mixer = nn.Linear(
            self.attention_dim, self.attention_dim, bias=False
        )
        nn.init.eye_(self.key_channel_mixer.weight)
        nn.init.eye_(self.value_channel_mixer.weight)
        self.layer_mixing = nn.Parameter(
            torch.full(
                (self.num_action_layers, self.num_heads, self.video_num_layers),
                1.0 / float(self.video_num_layers),
            )
        )

    @staticmethod
    def _split_modulation(
        block: nn.Module, t_mod: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        has_seq = t_mod.ndim == 4
        chunk_dim = 2 if has_seq else 1
        values = (
            block.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod
        ).chunk(6, dim=chunk_dim)
        if has_seq:
            values = tuple(value.squeeze(2) for value in values)
        return values

    def fuse_video_cache(
        self,
        canonical_keys: list[torch.Tensor],
        values: list[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        if len(canonical_keys) != self.video_num_layers or len(values) != self.video_num_layers:
            raise ValueError(
                f"Expected {self.video_num_layers} Video K/V layers, got "
                f"{len(canonical_keys)} and {len(values)}."
            )
        keys = self.key_channel_mixer(torch.stack(canonical_keys, dim=0))
        vals = self.value_channel_mixer(torch.stack(values, dim=0))
        layer_count, batch_size, seq_len, _ = keys.shape
        keys = keys.view(
            layer_count, batch_size, seq_len, self.num_heads, self.attn_head_dim
        )
        vals = vals.view(
            layer_count, batch_size, seq_len, self.num_heads, self.attn_head_dim
        )
        weights = self.layer_mixing.to(device=keys.device, dtype=keys.dtype)
        fused_keys = torch.einsum("ahl,lbshd->abshd", weights, keys)
        fused_values = torch.einsum("ahl,lbshd->abshd", weights, vals)
        return {
            "canonical_key": fused_keys.flatten(-2),
            "value": fused_values.flatten(-2),
        }

    def prefill_video(
        self,
        *,
        video_expert: nn.Module,
        video_pre: Dict[str, Any],
        video_attention_mask: torch.Tensor,
        tokens_per_frame: int,
    ) -> tuple[torch.Tensor, Dict[str, Any]]:
        if len(video_expert.blocks) != self.video_num_layers:
            raise ValueError("Video expert layer count does not match DoT configuration.")
        x = video_pre["tokens"]
        if video_attention_mask.shape != (x.size(1), x.size(1)):
            raise ValueError("Video attention mask must be square and match Video tokens.")
        first_frame_tokens = min(max(int(tokens_per_frame), 1), int(x.size(1)))
        canonical_keys: list[torch.Tensor] = []
        values: list[torch.Tensor] = []
        context = video_pre.get("context")
        context_mask = video_pre.get("context_mask")
        freqs = video_pre["freqs"]
        t_mod = video_pre["t_mod"]

        for block in video_expert.blocks:
            def run_layer(
                current: torch.Tensor,
                current_block: nn.Module = block,
            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                (
                    shift_msa,
                    scale_msa,
                    gate_msa,
                    shift_mlp,
                    scale_mlp,
                    gate_mlp,
                ) = self._split_modulation(current_block, t_mod)
                attn_input = modulate(
                    current_block.norm1(current), shift_msa, scale_msa
                )
                query = current_block.self_attn.norm_q(
                    current_block.self_attn.q(attn_input)
                )
                canonical_key = current_block.self_attn.norm_k(
                    current_block.self_attn.k(attn_input)
                )
                value = current_block.self_attn.v(attn_input)
                rotated_query = rope_apply(query, freqs, current_block.num_heads)
                rotated_key = rope_apply(
                    canonical_key, freqs, current_block.num_heads
                )
                mixed = flash_attention(
                    q=rotated_query,
                    k=rotated_key,
                    v=value,
                    num_heads=current_block.num_heads,
                    ctx_mask=video_attention_mask,
                )
                updated = current_block.gate(
                    current,
                    gate_msa,
                    current_block.self_attn.o(mixed),
                )
                if context is not None:
                    block_context_mask = context_mask
                    if block_context_mask is not None and block_context_mask.ndim == 3:
                        block_context_mask = block_context_mask.unsqueeze(1)
                    updated = updated + current_block.cross_attn(
                        current_block.norm3(updated),
                        context,
                        ctx_mask=block_context_mask,
                    )
                mlp_input = modulate(
                    current_block.norm2(updated), shift_mlp, scale_mlp
                )
                updated = current_block.gate(
                    updated, gate_mlp, current_block.ffn(mlp_input)
                )
                return (
                    updated,
                    canonical_key[:, :first_frame_tokens],
                    value[:, :first_frame_tokens],
                )

            if self.use_gradient_checkpointing and self.training:
                x, layer_key, layer_value = gradient_checkpoint(
                    run_layer, x, use_reentrant=False
                )
            else:
                x, layer_key, layer_value = run_layer(x)
            canonical_keys.append(layer_key)
            values.append(layer_value)
        return x, self.fuse_video_cache(canonical_keys, values)

    def forward_action(
        self,
        *,
        action_expert: FasterWAMActionHead,
        action_pre: Dict[str, Any],
        fused_video_cache: Dict[str, torch.Tensor],
        condition_residual: Optional[
            Callable[[torch.Tensor], torch.Tensor]
        ] = None,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if len(action_expert.blocks) != self.num_action_layers:
            raise ValueError("Action expert layer count does not match DoT configuration.")
        if self.num_action_layers != 1:
            raise ValueError("The Faster-WAM baseline requires exactly one Action layer.")
        block = action_expert.blocks[0]
        video_key = fused_video_cache["canonical_key"][0]
        video_value = fused_video_cache["value"][0]
        action_tokens = action_pre["tokens"]
        video_batch = int(video_key.size(0))
        action_batch = int(action_tokens.size(0))
        if action_batch != video_batch:
            if video_batch <= 0 or action_batch % video_batch != 0:
                raise ValueError(
                    "Action batch must equal or be an integer multiple of the "
                    "shared Video K/V cache batch."
                )
            repeat = action_batch // video_batch
            video_key = video_key.repeat_interleave(repeat, dim=0)
            video_value = video_value.repeat_interleave(repeat, dim=0)

        def run_action(
            current: torch.Tensor,
            current_video_key: torch.Tensor,
            current_video_value: torch.Tensor,
        ) -> torch.Tensor:
            (
                shift_msa,
                scale_msa,
                gate_msa,
                shift_mlp,
                scale_mlp,
                gate_mlp,
            ) = self._split_modulation(block, action_pre["t_mod"])
            attn_input = modulate(block.norm1(current), shift_msa, scale_msa)
            action_query = block.self_attn.norm_q(block.self_attn.q(attn_input))
            action_key = block.self_attn.norm_k(block.self_attn.k(attn_input))
            action_value = block.self_attn.v(attn_input)
            action_query = rope_apply(
                action_query, action_pre["freqs"], block.num_heads
            )
            action_key = rope_apply(
                action_key, action_pre["freqs"], block.num_heads
            )

            # Fused Video keys are in canonical (3D-RoPE-free) space. Normalize
            # them with the Action key norm and assign action-side position zero.
            aligned_video_key = block.self_attn.norm_k(current_video_key)
            zero_freqs = action_expert.freqs[:1].view(1, 1, -1).to(
                aligned_video_key.device
            )
            zero_freqs = zero_freqs.expand(aligned_video_key.size(1), -1, -1)
            aligned_video_key = rope_apply(
                aligned_video_key, zero_freqs, block.num_heads
            )
            mixed = flash_attention(
                q=action_query,
                k=torch.cat([aligned_video_key, action_key], dim=1),
                v=torch.cat([current_video_value, action_value], dim=1),
                num_heads=block.num_heads,
                ctx_mask=None,
            )
            updated = block.gate(current, gate_msa, block.self_attn.o(mixed))
            if condition_residual is not None:
                residual = condition_residual(updated)
                if residual.shape != updated.shape:
                    raise ValueError(
                        "DoT condition residual must match the Action hidden shape."
                    )
                updated = updated + residual
            mlp_input = modulate(block.norm2(updated), shift_mlp, scale_mlp)
            return block.gate(updated, gate_mlp, block.ffn(mlp_input))

        if self.use_gradient_checkpointing and self.training:
            output = gradient_checkpoint(
                run_action,
                action_tokens,
                video_key,
                video_value,
                use_reentrant=False,
            )
        else:
            output = run_action(action_tokens, video_key, video_value)
        if not return_attention:
            return output

        # Flash Attention intentionally returns only the mixed values. Recompute
        # QK scores for visualization without changing the action output path.
        (
            shift_msa,
            scale_msa,
            _,
            _,
            _,
            _,
        ) = self._split_modulation(block, action_pre["t_mod"])
        attn_input = modulate(block.norm1(action_tokens), shift_msa, scale_msa)
        action_query = block.self_attn.norm_q(block.self_attn.q(attn_input))
        action_key = block.self_attn.norm_k(block.self_attn.k(attn_input))
        action_query = rope_apply(
            action_query, action_pre["freqs"], block.num_heads
        )
        action_key = rope_apply(
            action_key, action_pre["freqs"], block.num_heads
        )
        aligned_video_key = block.self_attn.norm_k(video_key)
        zero_freqs = action_expert.freqs[:1].view(1, 1, -1).to(
            aligned_video_key.device
        )
        zero_freqs = zero_freqs.expand(aligned_video_key.size(1), -1, -1)
        aligned_video_key = rope_apply(
            aligned_video_key, zero_freqs, block.num_heads
        )

        batch_size, query_len, attention_dim = action_query.shape
        num_heads = int(block.num_heads)
        head_dim = attention_dim // num_heads
        query_heads = action_query.reshape(
            batch_size, query_len, num_heads, head_dim
        ).transpose(1, 2).float()
        all_keys = torch.cat([aligned_video_key, action_key], dim=1)
        key_heads = all_keys.reshape(
            batch_size, all_keys.size(1), num_heads, head_dim
        ).transpose(1, 2).float()
        scores = torch.matmul(query_heads, key_heads.transpose(-2, -1))
        scores = scores / math.sqrt(max(head_dim, 1))
        video_attention = torch.softmax(scores, dim=-1)[
            ..., : aligned_video_key.size(1)
        ].detach()
        return output, video_attention
