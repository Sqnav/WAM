from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from data.action_mapping import norm_action_to_physical
from data.visual_guidance import project_body_to_image_features
from .config import ModelConfig
from .encoders import CLIPTextEncoder, DINOv2ImageEncoder, PrivilegedEncoder
from .fusion import CrossAttentionFusion
from .heads import DiTActionHead, DirectActionHead, TeacherPredictionHeads
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
        self.heatmap_token_proj = nn.Sequential(
            nn.LayerNorm(3),
            nn.Linear(3, cfg.image_encoder_dim),
            nn.GELU(),
            nn.Linear(cfg.image_encoder_dim, cfg.image_encoder_dim),
        )
        if not (cfg.use_target_visual_guidance and cfg.use_attention_heatmap):
            for p in self.heatmap_token_proj.parameters():
                p.requires_grad_(False)
        if cfg.use_diffusion_actor:
            for p in self.direct_action.parameters():
                p.requires_grad_(False)
        else:
            for p in self.actor.parameters():
                p.requires_grad_(False)

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
        attention_heatmaps: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if images.ndim != 5:
            raise ValueError("images must have shape [B, T, C, H, W].")
        if privileged.ndim != 3:
            raise ValueError("privileged must have shape [B, T, D].")
        batch_size, seq_len, *_ = images.shape
        if privileged.shape[:2] != (batch_size, seq_len):
            raise ValueError("privileged batch/time shape must match images.")

        single_text = text_tokens.ndim == 2 or (text_tokens.ndim == 3 and text_tokens.size(1) == 1)
        text_tokens_in = text_tokens
        attention_mask_in = attention_mask
        text_tokens, attention_mask = self._prepare_text(text_tokens, batch_size, seq_len, attention_mask)

        flat_images = images.reshape(batch_size * seq_len, *images.shape[2:])
        flat_priv = privileged.reshape(batch_size * seq_len, privileged.size(-1))
        _, image_tokens = self.image_encoder(flat_images)
        if bool(getattr(self.cfg, "use_target_visual_guidance", False)):
            extra_tokens = []
            if bool(getattr(self.cfg, "use_attention_heatmap", True)) and attention_heatmaps is not None:
                if attention_heatmaps.shape[:2] != (batch_size, seq_len):
                    raise ValueError("attention_heatmaps batch/time shape must match images.")
                heatmap_feat = project_body_to_image_features(
                    flat_priv,
                    image_hw=(images.shape[-2], images.shape[-1]),
                    fov_deg=self.cfg.visual_guidance_fov_deg,
                )
                extra_tokens.append(self.heatmap_token_proj(heatmap_feat).unsqueeze(1))
            if extra_tokens:
                image_tokens = torch.cat([image_tokens, *extra_tokens], dim=1)
        if single_text:
            text_once = text_tokens_in if text_tokens_in.ndim == 2 else text_tokens_in[:, 0]
            mask_once = None
            if attention_mask_in is not None:
                mask_once = attention_mask_in if attention_mask_in.ndim == 2 else attention_mask_in[:, 0]
            _, text_seq_once = self.text_encoder(text_once, mask_once)
            text_seq = (
                text_seq_once.unsqueeze(1)
                .expand(batch_size, seq_len, text_seq_once.size(1), text_seq_once.size(2))
                .reshape(batch_size * seq_len, text_seq_once.size(1), text_seq_once.size(2))
            )
        else:
            flat_text = text_tokens.reshape(batch_size * seq_len, text_tokens.size(-1))
            flat_mask = None if attention_mask is None else attention_mask.reshape(batch_size * seq_len, attention_mask.size(-1))
            _, text_seq = self.text_encoder(flat_text, flat_mask)
        # The policy no longer receives the privileged target vector as input.
        # ``privileged`` is still passed through this method for supervision and
        # optional target-projection guidance, but the fusion context gets only a
        # learned constant token produced from zeros.
        privileged_token = self.privileged_encoder(torch.zeros_like(flat_priv))
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
        attention_heatmaps: Optional[torch.Tensor] = None,
        start_state: Optional[RSSMState] = None,
        expert_action: Optional[torch.Tensor] = None,
        valid_mask: Optional[torch.Tensor] = None,
        done: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        encoded = self.encode_sequence(
            images,
            text_tokens,
            privileged,
            attention_mask,
            attention_heatmaps=attention_heatmaps,
        )
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
            if self.cfg.use_diffusion_actor:
                diffusion = self.actor.diffusion_loss(
                    feat,
                    expert_action.float(),
                    valid_mask=valid_mask,
                )
                out["policy_diffusion_loss"] = diffusion["loss"]
                out["policy_action_sequence"] = diffusion["pred_action"]
                out["policy_action"] = diffusion["pred_action"][..., 0, :]
                out["policy_pred_noise"] = diffusion["pred_noise"]
            else:
                out["policy_action"] = self.direct_action(feat)

            if self.cfg.train_rollout:
                out["rollout_privileged"] = self.rollout_privileged_predictions(posts, expert_action)
        return out

    def rollout_privileged_predictions(
        self,
        start_states: RSSMState,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        if actions.ndim != 3:
            raise ValueError("actions must have shape [B, T, A].")
        horizon = max(int(getattr(self.cfg, "rollout_horizon", 1)), 1)
        batch_size, seq_len, _ = actions.shape
        preds = []
        for t in range(seq_len):
            state = {k: v[:, t] for k, v in start_states.items()}
            per_t = []
            for k in range(horizon):
                action_idx = min(t + k, seq_len - 1)
                state = self.rssm.imagine_step(state, actions[:, action_idx])
                feat = self.rssm.get_feat(state)
                per_t.append(self.prediction_heads(feat)["privileged"])
            preds.append(torch.stack(per_t, dim=1))
        return torch.stack(preds, dim=1)

    def _repeat_rssm_state(self, state: RSSMState, repeat: int) -> RSSMState:
        return {
            k: v.unsqueeze(1)
            .expand(v.size(0), repeat, *v.shape[1:])
            .reshape(v.size(0) * repeat, *v.shape[1:])
            for k, v in state.items()
        }

    @torch.no_grad()
    def select_dit_action_sequence(
        self,
        feat: torch.Tensor,
        post_state: RSSMState,
        prev_action: torch.Tensor,
        num_steps: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        candidate_count = max(int(getattr(self.cfg, "dit_candidate_count", 4)), 1)
        if candidate_count <= 1:
            seq = self.actor.sample(
                feat,
                num_steps=num_steps or self.cfg.action_sampling_steps,
                deterministic=True,
            )
            return seq, {
                "candidate_scores": torch.zeros(feat.size(0), 1, device=feat.device),
                "selected_candidate": torch.zeros(feat.size(0), dtype=torch.long, device=feat.device),
            }

        batch_size = feat.size(0)
        flat_feat = feat.unsqueeze(1).expand(batch_size, candidate_count, feat.size(-1)).reshape(
            batch_size * candidate_count,
            feat.size(-1),
        )
        candidate_seq = self.actor.sample(
            flat_feat,
            num_steps=num_steps or self.cfg.action_sampling_steps,
            deterministic=False,
        )

        state = self._repeat_rssm_state(post_state, candidate_count)
        pred_privileged = []
        horizon = candidate_seq.size(1)
        for k in range(horizon):
            state = self.rssm.imagine_step(state, candidate_seq[:, k])
            rollout_feat = self.rssm.get_feat(state)
            pred_privileged.append(self.prediction_heads(rollout_feat)["privileged"])
        pred_privileged_t = torch.stack(pred_privileged, dim=1).view(
            batch_size,
            candidate_count,
            horizon,
            -1,
        )

        lateral = pred_privileged_t[..., 1].abs()
        vertical = pred_privileged_t[..., 2].abs()
        distance = torch.linalg.norm(pred_privileged_t, dim=-1)
        smooth = torch.linalg.norm(
            candidate_seq.view(batch_size, candidate_count, horizon, -1)[:, :, 0] - prev_action.unsqueeze(1),
            dim=-1,
        )
        scores = (
            float(getattr(self.cfg, "dit_candidate_lateral_weight", 1.0)) * lateral.mean(dim=-1)
            + float(getattr(self.cfg, "dit_candidate_vertical_weight", 1.0)) * vertical.mean(dim=-1)
            + float(getattr(self.cfg, "dit_candidate_distance_weight", 0.05)) * distance.mean(dim=-1)
            + float(getattr(self.cfg, "dit_candidate_smooth_weight", 0.05)) * smooth
        )

        selected = torch.argmin(scores, dim=1)
        seq = candidate_seq.view(batch_size, candidate_count, horizon, -1)[
            torch.arange(batch_size, device=feat.device),
            selected,
        ]
        return seq, {
            "candidate_scores": scores,
            "selected_candidate": selected,
        }

    @torch.no_grad()
    def act(
        self,
        image: torch.Tensor,
        text_tokens: torch.Tensor,
        privileged: torch.Tensor,
        prev_action: torch.Tensor,
        rssm_state: Optional[RSSMState] = None,
        attention_mask: Optional[torch.Tensor] = None,
        attention_heatmap: Optional[torch.Tensor] = None,
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
        attention_heatmaps = None if attention_heatmap is None else attention_heatmap.unsqueeze(1)

        encoded = self.encode_sequence(
            images,
            text_tokens,
            privileged_seq,
            attention_mask,
            attention_heatmaps=attention_heatmaps,
        )
        _, post = self.rssm.obs_step(rssm_state, prev_action, encoded["obs_embed"].squeeze(1))
        feat = self.rssm.get_feat(post)
        if self.cfg.use_diffusion_actor:
            candidate_info = None
            if bool(getattr(self.cfg, "dit_candidate_selection", False)):
                action_sequence_norm, candidate_info = self.select_dit_action_sequence(
                    feat,
                    post,
                    prev_action,
                    num_steps=num_steps,
                )
            else:
                action_sequence_norm = self.actor.sample(
                    feat, num_steps=num_steps or self.cfg.action_sampling_steps, deterministic=deterministic
                )
            action_norm = action_sequence_norm[:, 0]
        else:
            action_sequence_norm = None
            candidate_info = None
            action_norm = self.direct_action(feat)
        action_physical = norm_action_to_physical(
            action_norm,
            max_vel=self.cfg.max_vel,
            max_yaw_rate=self.cfg.max_yaw_rate,
            max_speed_norm=self.cfg.max_speed_norm,
        )
        heads = self.prediction_heads(feat)
        out = {"action": action_physical, "action_norm": action_norm, "action_physical": action_physical, **heads}
        if action_sequence_norm is not None:
            out["action_sequence_norm"] = action_sequence_norm
        if candidate_info is not None:
            out.update(candidate_info)
        return out, post
