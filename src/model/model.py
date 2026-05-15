from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from data.action_mapping import norm_action_to_physical
from .config import ModelConfig
from .encoders import CLIPTextEncoder, DINOv2ImageEncoder, PrivilegedEncoder
from .fusion import CrossAttentionFusion
from .heads import DiTActionHead, DirectActionHead, PrivilegedReconHead, TeacherPredictionHeads
from .rssm import RSSM, RSSMState


class PrivilegedTeacherWorldModelDiT(nn.Module):
    """Privileged teacher world model with real CLIP text encoder and no state input."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.image_encoder = DINOv2ImageEncoder(cfg)
        self.text_encoder = CLIPTextEncoder(cfg)
        self.privileged_encoder = PrivilegedEncoder(cfg)
        self.fusion = CrossAttentionFusion(cfg)
        self.rssm = RSSM(cfg)
        self.prediction_heads = TeacherPredictionHeads(cfg)
        self.actor = DiTActionHead(cfg)
        self.direct_action = DirectActionHead(cfg)
        self.privileged_recon = PrivilegedReconHead(cfg)

    def initial_state(self, batch_size: int, device: torch.device) -> RSSMState:
        return self.rssm.init_state(batch_size, device)

    def _prepare_text(
        self,
        text_tokens: torch.Tensor,
        batch_size: int,
        seq_len: int,
        attention_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if text_tokens.ndim == 2:
            text_tokens = text_tokens.unsqueeze(1).expand(batch_size, seq_len, -1)
        elif text_tokens.ndim == 3 and text_tokens.size(1) == 1 and seq_len > 1:
            text_tokens = text_tokens.expand(batch_size, seq_len, -1)
        elif text_tokens.ndim != 3:
            raise ValueError("text_tokens must have shape [B, L], [B, 1, L], or [B, T, L].")

        if text_tokens.shape[0] != batch_size or text_tokens.shape[1] != seq_len:
            raise ValueError(
                f"text_tokens must have batch/time shape [B={batch_size}, T={seq_len}], "
                f"got {tuple(text_tokens.shape[:2])}."
            )

        if attention_mask is not None:
            if attention_mask.ndim == 2:
                attention_mask = attention_mask.unsqueeze(1).expand(batch_size, seq_len, -1)
            elif attention_mask.ndim == 3 and attention_mask.size(1) == 1 and seq_len > 1:
                attention_mask = attention_mask.expand(batch_size, seq_len, -1)
            elif attention_mask.ndim != 3:
                raise ValueError("attention_mask must have shape [B, L], [B, 1, L], or [B, T, L].")
            if attention_mask.shape[:2] != text_tokens.shape[:2]:
                raise ValueError("attention_mask batch/time shape must match text_tokens.")
        return text_tokens, attention_mask

    def _make_prev_dones(self, done: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if done is None:
            return None
        if done.ndim == 3 and done.size(-1) == 1:
            done_2d = done.squeeze(-1)
        elif done.ndim == 2:
            done_2d = done
        else:
            raise ValueError("done must have shape [B, T] or [B, T, 1].")
        prev_dones = torch.zeros_like(done_2d)
        if done_2d.size(1) > 1:
            prev_dones[:, 1:] = done_2d[:, :-1]
        return prev_dones

    def encode_sequence(
        self,
        images: torch.Tensor,
        text_tokens: torch.Tensor,
        privileged: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if images.ndim != 5:
            raise ValueError("images must have shape [B, T, C, H, W].")
        if privileged.ndim != 3:
            raise ValueError("privileged must have shape [B, T, D].")
        batch_size, seq_len, *_ = images.shape
        if privileged.shape[:2] != (batch_size, seq_len):
            raise ValueError("privileged batch/time shape must match images.")

        text_tokens, attention_mask = self._prepare_text(text_tokens, batch_size, seq_len, attention_mask)

        flat_images = images.reshape(batch_size * seq_len, *images.shape[2:])
        flat_priv = privileged.reshape(batch_size * seq_len, privileged.size(-1))
        flat_text = text_tokens.reshape(batch_size * seq_len, text_tokens.size(-1))
        flat_mask = None if attention_mask is None else attention_mask.reshape(batch_size * seq_len, attention_mask.size(-1))

        _, image_tokens = self.image_encoder(flat_images)
        _, text_seq = self.text_encoder(flat_text, flat_mask)
        privileged_token = self.privileged_encoder(flat_priv)
        obs_embed, fused_tokens = self.fusion(image_tokens, text_seq, privileged_token)

        return {
            "obs_embed": obs_embed.view(batch_size, seq_len, -1),
            "fused_tokens": fused_tokens.view(batch_size, seq_len, fused_tokens.size(1), fused_tokens.size(2)),
        }

    def forward(
        self,
        images: torch.Tensor,
        text_tokens: torch.Tensor,
        privileged: torch.Tensor,
        prev_actions: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        start_state: Optional[RSSMState] = None,
        expert_action: Optional[torch.Tensor] = None,
        valid_mask: Optional[torch.Tensor] = None,
        done: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        encoded = self.encode_sequence(images, text_tokens, privileged, attention_mask)
        prev_dones = self._make_prev_dones(done)
        priors, posts = self.rssm.observe(
            encoded["obs_embed"],
            prev_actions,
            start_state=start_state,
            prev_dones=prev_dones,
        )

        feat = self.rssm.get_feat(posts)
        prior_feat = self.rssm.get_feat(priors)
        preds = self.prediction_heads(feat)
        prior_preds = {f"prior_{k}": v for k, v in self.prediction_heads(prior_feat).items()}
        out = {
            "obs_embed": encoded["obs_embed"],
            "priors": priors,
            "posts": posts,
            "feat": feat,
            "prior_feat": prior_feat,
            **preds,
            **prior_preds,
        }
        if expert_action is not None:
            out["privileged_recon"] = self.privileged_recon(feat)
            if self.cfg.use_diffusion_actor:
                out["policy_action"] = self.actor.sample_training(
                    feat,
                    num_steps=self.cfg.action_sampling_steps,
                    deterministic=True,
                )
            else:
                out["policy_action"] = self.direct_action(feat)
        return out

    @torch.no_grad()
    def act(
        self,
        image: torch.Tensor,
        text_tokens: torch.Tensor,
        privileged: torch.Tensor,
        prev_action: torch.Tensor,
        rssm_state: Optional[RSSMState] = None,
        attention_mask: Optional[torch.Tensor] = None,
        prev_done: Optional[torch.Tensor] = None,
        deterministic: bool = True,
        num_steps: Optional[int] = None,
    ) -> Tuple[Dict[str, torch.Tensor], RSSMState]:
        if image.ndim != 4:
            raise ValueError("image must have shape [B, C, H, W].")
        if privileged.ndim != 2:
            raise ValueError("privileged must have shape [B, D].")
        batch_size = image.size(0)
        if rssm_state is None:
            rssm_state = self.initial_state(batch_size, image.device)
        if prev_done is not None:
            rssm_state = self.rssm.reset_state_by_done(rssm_state, prev_done)

        images = image.unsqueeze(1)
        privileged_seq = privileged.unsqueeze(1)

        encoded = self.encode_sequence(images, text_tokens, privileged_seq, attention_mask)
        _, post = self.rssm.obs_step(rssm_state, prev_action, encoded["obs_embed"].squeeze(1))
        feat = self.rssm.get_feat(post)
        if self.cfg.use_diffusion_actor:
            action_norm = self.actor.sample(
                feat, num_steps=num_steps or self.cfg.action_sampling_steps, deterministic=deterministic
            )
        else:
            action_norm = self.direct_action(feat)
        action_physical = norm_action_to_physical(
            action_norm,
            max_vel=self.cfg.max_vel,
            max_yaw_rate=self.cfg.max_yaw_rate,
            max_speed_norm=self.cfg.max_speed_norm,
        )
        heads = self.prediction_heads(feat)
        out = {"action": action_physical, "action_norm": action_norm, "action_physical": action_physical, **heads}
        return out, post
