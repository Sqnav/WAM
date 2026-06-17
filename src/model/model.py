from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from data.action_mapping import norm_action_to_physical
from .config import ModelConfig
from .encoders import TargetTokenEncoder, Wan22TextEncoder, Wan22VAEImageEncoder
from .fusion import CrossAttentionFusion
from .heads import FastWAMHead, TeacherPredictionHeads
from .rssm import RSSM, RSSMState


def migrate_legacy_state_dict_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Map old checkpoint module names to target-relative terminology."""
    replacements = (
        ("privileged_encoder.", "target_token_encoder."),
        ("fusion.priv_proj.", "fusion.target_token_proj."),
        ("prediction_heads.privileged.", "prediction_heads.next_target_relative."),
    )
    migrated: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        new_key = key
        for old, new in replacements:
            new_key = new_key.replace(old, new)
        migrated[new_key] = value
    return migrated


class TeacherWorldModelDiT(nn.Module):
    """Teacher world model with Wan2.2 visual/text encoders and no state input."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        if not cfg.use_wan22_encoders:
            raise RuntimeError("DINOv2/CLIP encoders were removed; set cfg.use_wan22_encoders=True.")
        self.image_encoder = Wan22VAEImageEncoder(cfg)
        self.text_encoder = Wan22TextEncoder(cfg)
        self.target_token_encoder = TargetTokenEncoder(cfg)
        target_context_hidden = max(
            int(getattr(cfg, "target_relative_context_hidden_dim", cfg.text_width)),
            int(cfg.target_relative_dim),
            1,
        )
        self.target_relative_context_proj = nn.Sequential(
            nn.Linear(cfg.target_relative_dim, target_context_hidden),
            nn.GELU(),
            nn.Dropout(float(getattr(cfg, "dropout", 0.0))),
            nn.Linear(target_context_hidden, cfg.text_width),
        )
        if not bool(getattr(cfg, "use_target_relative_context", False)):
            for p in self.target_relative_context_proj.parameters():
                p.requires_grad_(False)
        self.fusion = CrossAttentionFusion(cfg)
        self.rssm = RSSM(cfg) if cfg.use_rssm else None
        self.prediction_heads = TeacherPredictionHeads(cfg)
        self.fastwam = FastWAMHead(cfg) if bool(cfg.use_fastwam_mot) else None
        if self.fastwam is None:
            raise RuntimeError("Legacy MLP/DiT actors were removed; set cfg.use_fastwam_mot=True.")

    def initial_state(self, batch_size: int, device: torch.device) -> RSSMState:
        if self.rssm is None:
            raise RuntimeError("RSSM is disabled in this model.")
        return self.rssm.init_state(batch_size, device)

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

    def _make_target_relative_context_tokens(
        self,
        target_relative: torch.Tensor,
        target_device: torch.device,
        target_dtype: torch.dtype,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if target_relative.ndim != 3:
            raise ValueError("target_relative must have shape [B, T, D].")
        if not bool(getattr(self.cfg, "use_target_relative_context", False)):
            return None, None
        if target_relative.size(-1) != int(self.cfg.target_relative_dim):
            raise ValueError("target_relative feature dim must match cfg.target_relative_dim.")
        current_target = target_relative[:, 0].float()
        scale = max(float(getattr(self.cfg, "target_relative_context_scale", 1.0)), 1e-6)
        current_target = current_target / scale
        param = next(self.target_relative_context_proj.parameters())
        tokens = self.target_relative_context_proj(
            current_target.to(device=param.device, dtype=param.dtype)
        ).unsqueeze(1)
        tokens = tokens * float(getattr(self.cfg, "target_relative_token_scale", 1.0))
        tokens = tokens.to(device=target_device, dtype=target_dtype)
        mask = torch.ones(tokens.shape[:2], device=target_device, dtype=torch.bool)
        return tokens, mask

    def encode_sequence(
        self,
        images: torch.Tensor,
        text_tokens: torch.Tensor,
        target_relative: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        instructions: Optional[list[str]] = None,
        video_latents: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if images.ndim != 5:
            raise ValueError("images must have shape [B, T, C, H, W].")
        if target_relative.ndim != 3:
            raise ValueError("target_relative must have shape [B, T, D].")
        batch_size, seq_len, *_ = images.shape
        if target_relative.shape[:2] != (batch_size, seq_len):
            raise ValueError("target_relative batch/time shape must match images.")

        if not self.cfg.use_wan22_encoders:
            raise RuntimeError("DINOv2/CLIP encoders were removed; set cfg.use_wan22_encoders=True.")

        video_images = images
        action_video_freq_ratio = max(int(getattr(self.cfg, "fastwam_action_video_freq_ratio", 1)), 1)
        if action_video_freq_ratio > 1 and seq_len > 1:
            if (seq_len - 1) % action_video_freq_ratio != 0:
                raise ValueError(
                    "For FastWAM temporal sampling, (seq_len - 1) must be divisible by "
                    f"fastwam_action_video_freq_ratio; got seq_len={seq_len}, "
                    f"ratio={action_video_freq_ratio}."
                )
            video_images = images[:, ::action_video_freq_ratio]
        if (video_images.size(1) - 1) % 4 != 0:
            raise ValueError(
                "Wan VAE expects sampled video frame count T to satisfy T % 4 == 1; "
                f"got sampled_video_len={video_images.size(1)} from seq_len={seq_len}."
            )
        if video_latents is None:
            video_latents = self.image_encoder.encode_video_latents(video_images)
        else:
            if video_latents.ndim != 5:
                raise ValueError("video_latents must have shape [B, C, T_lat, H_lat, W_lat].")
            if video_latents.size(0) != batch_size:
                raise ValueError("video_latents batch size must match images.")
            video_latents = video_latents.to(device=images.device, dtype=getattr(self.image_encoder, "dtype", images.dtype))
        # Official FastWAM trains and predicts in Wan VAE latent space. The
        # flattened tokens below are only used for auxiliary observation
        # features; optional privileged target context is appended to the text
        # context that FastWAM consumes.
        video_tokens = video_latents.permute(0, 2, 3, 4, 1).reshape(
            video_latents.size(0),
            video_latents.size(2),
            -1,
            video_latents.size(1),
        ).float()
        latent_seq_len = video_tokens.size(1)
        image_tokens = video_tokens.reshape(batch_size * latent_seq_len, video_tokens.size(2), video_tokens.size(3))
        if instructions is None:
            raise ValueError("Wan2.2 text encoder requires raw instruction strings.")
        _, text_once, text_mask_once = self.text_encoder.encode_texts_with_mask(instructions, images.device)
        text_seq = (
            text_once.unsqueeze(1)
            .expand(batch_size, latent_seq_len, text_once.size(1), text_once.size(2))
            .reshape(batch_size * latent_seq_len, text_once.size(1), text_once.size(2))
        )
        # The policy no longer receives the target-relative label as input. It is
        # still passed here for supervision and optional FastWAM context, while
        # fusion receives only a learned constant null-target token.
        target_token = self.target_token_encoder(
            torch.zeros(
                batch_size * latent_seq_len,
                target_relative.size(-1),
                device=target_relative.device,
                dtype=target_relative.dtype,
            )
        )
        obs_embed, fused_tokens = self.fusion(
            image_tokens,
            text_seq,
            target_token,
        )
        fastwam_context = text_once
        fastwam_context_mask = text_mask_once
        target_context, target_context_mask = self._make_target_relative_context_tokens(
            target_relative,
            target_device=fastwam_context.device,
            target_dtype=fastwam_context.dtype,
        )
        if target_context is not None and target_context_mask is not None:
            fastwam_context = torch.cat([fastwam_context, target_context], dim=1)
            fastwam_context_mask = torch.cat([fastwam_context_mask, target_context_mask], dim=1)

        encoded_out = {
            "obs_embed": obs_embed.view(batch_size, latent_seq_len, -1),
            "fused_tokens": fused_tokens.view(batch_size, latent_seq_len, fused_tokens.size(1), fused_tokens.size(2)),
            "video_latents": video_latents,
            "text_context": fastwam_context,
            "text_context_mask": fastwam_context_mask,
        }
        return encoded_out

    def forward(
        self,
        images: torch.Tensor,
        text_tokens: torch.Tensor,
        target_relative: torch.Tensor,
        prev_actions: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        start_state: Optional[RSSMState] = None,
        expert_action: Optional[torch.Tensor] = None,
        valid_mask: Optional[torch.Tensor] = None,
        done: Optional[torch.Tensor] = None,
        instructions: Optional[list[str]] = None,
        video_latents: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        encoded = self.encode_sequence(
            images,
            text_tokens,
            target_relative,
            attention_mask,
            instructions=instructions,
            video_latents=video_latents,
        )
        prev_dones = self._make_prev_dones(done)
        if self.rssm is None:
            feat = encoded["obs_embed"]
            if self.cfg.train_next_target_relative:
                preds = self.prediction_heads(feat)
                prior_preds = {
                    "prior_next_target_relative": torch.zeros_like(preds["next_target_relative"]),
                }
            else:
                preds = {}
                prior_preds = {}
            priors = None
            posts = None
            prior_feat = feat
        else:
            priors, posts = self.rssm.observe(
                encoded["obs_embed"],
                prev_actions,
                start_state=start_state,
                prev_dones=prev_dones,
            )
            feat = self.rssm.get_feat(posts)
            prior_feat = self.rssm.get_feat(priors)

        if expert_action is not None and self.rssm is not None:
            next_target_relative = self._predict_next_target_from_action(posts, expert_action.float())
            prior_next_target_relative = self._predict_next_target_from_action(priors, expert_action.float())
            preds = {"next_target_relative": next_target_relative}
            prior_preds = {"prior_next_target_relative": prior_next_target_relative}
        elif self.rssm is not None:
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
            if self.fastwam is not None:
                if encoded.get("video_latents") is None or encoded.get("text_context") is None:
                    raise RuntimeError("Official FastWAM head requires Wan2.2 latents and raw text context.")
                fastwam_out = self.fastwam.training_loss(
                    video_latents=encoded["video_latents"],
                    context=encoded["text_context"],
                    context_mask=encoded["text_context_mask"],
                    expert_action=expert_action.float(),
                    valid_mask=valid_mask,
                )
                out["video_flow_loss"] = fastwam_out["loss_video"]
                out["policy_flow_loss"] = fastwam_out["loss_action"]
                out["policy_action_sequence"] = fastwam_out["pred_action"]
                out["policy_action"] = fastwam_out["pred_action"][..., 0, :]
            elif self.cfg.use_diffusion_actor:
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

        return out

    def _repeat_rssm_state(self, state: RSSMState, repeat: int) -> RSSMState:
        if self.rssm is None:
            raise RuntimeError("RSSM candidate rollout is unavailable when use_rssm=false.")
        return {
            k: v.unsqueeze(1)
            .expand(v.size(0), repeat, *v.shape[1:])
            .reshape(v.size(0) * repeat, *v.shape[1:])
            for k, v in state.items()
        }

    def _flatten_time_state(self, state: RSSMState) -> RSSMState:
        return {k: v.reshape(v.size(0) * v.size(1), *v.shape[2:]) for k, v in state.items()}

    def _unflatten_time_state(self, state: RSSMState, batch_size: int, seq_len: int) -> RSSMState:
        return {k: v.reshape(batch_size, seq_len, *v.shape[1:]) for k, v in state.items()}

    def _predict_next_target_from_action(
        self,
        state: RSSMState,
        action: torch.Tensor,
    ) -> torch.Tensor:
        if action.ndim != 3:
            raise ValueError("action must have shape [B, T, A].")
        if self.rssm is None:
            raise RuntimeError("RSSM imagination is unavailable when use_rssm=false.")
        batch_size, seq_len, _ = action.shape
        flat_state = self._flatten_time_state(state)
        flat_action = action.reshape(batch_size * seq_len, action.size(-1))
        future_state = self.rssm.imagine_step(flat_state, flat_action)
        future_state = self._unflatten_time_state(future_state, batch_size, seq_len)
        future_feat = self.rssm.get_feat(future_state)
        return self.prediction_heads(future_feat)["next_target_relative"]

    @torch.no_grad()
    def select_dit_action_sequence(
        self,
        feat: torch.Tensor,
        post_state: RSSMState,
        prev_action: torch.Tensor,
        current_target_relative: Optional[torch.Tensor] = None,
        num_steps: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        candidate_count = max(int(getattr(self.cfg, "dit_candidate_count", 4)), 1)
        if self.rssm is None:
            raise RuntimeError("DiT candidate selection requires RSSM; disable dit_candidate_selection.")
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
        pred_target_relative = []
        horizon = candidate_seq.size(1)
        for k in range(horizon):
            state = self.rssm.imagine_step(state, candidate_seq[:, k])
            rollout_feat = self.rssm.get_feat(state)
            pred_target_relative.append(self.prediction_heads(rollout_feat)["next_target_relative"])
        pred_target_relative_t = torch.stack(pred_target_relative, dim=1).view(
            batch_size,
            candidate_count,
            horizon,
            -1,
        )

        candidate_seq_t = candidate_seq.view(batch_size, candidate_count, horizon, -1)
        x = pred_target_relative_t[..., 0]
        y = pred_target_relative_t[..., 1]
        z = pred_target_relative_t[..., 2]
        distance = torch.linalg.norm(pred_target_relative_t, dim=-1)

        if current_target_relative is None:
            current_distance = distance[:, :, :1].detach().mean(dim=1)
        else:
            current_distance = torch.linalg.norm(current_target_relative.float(), dim=-1, keepdim=True)
        distance_scale = current_distance.clamp(min=1.0).unsqueeze(1)

        forward_for_angle = x.abs().clamp(min=1.0)
        horizontal_for_angle = torch.linalg.norm(pred_target_relative_t[..., :2], dim=-1).clamp(min=1.0)
        yaw_angle = torch.atan2(y, forward_for_angle).abs()
        pitch_angle = torch.atan2(z, horizontal_for_angle).abs()
        yaw_score = yaw_angle.mean(dim=-1)
        pitch_score = pitch_angle.mean(dim=-1)
        final_distance = distance[..., -1]
        final_distance_norm = final_distance / distance_scale.squeeze(-1)
        progress_penalty = torch.relu(final_distance - current_distance) / distance_scale.squeeze(-1)
        front_penalty = torch.relu(-x).mean(dim=-1) / distance_scale.squeeze(-1)
        smooth_prev = torch.linalg.norm(candidate_seq_t[:, :, 0] - prev_action.unsqueeze(1), dim=-1)
        if horizon > 1:
            temporal_smooth = torch.linalg.norm(candidate_seq_t[:, :, 1:] - candidate_seq_t[:, :, :-1], dim=-1).mean(dim=-1)
        else:
            temporal_smooth = torch.zeros_like(smooth_prev)
        action_effort = torch.linalg.norm(candidate_seq_t, dim=-1).mean(dim=-1)

        scores = (
            float(getattr(self.cfg, "dit_candidate_yaw_angle_weight", 1.0)) * yaw_score
            + float(getattr(self.cfg, "dit_candidate_pitch_angle_weight", 0.7)) * pitch_score
            + float(getattr(self.cfg, "dit_candidate_final_distance_weight", 0.25)) * final_distance_norm
            + float(getattr(self.cfg, "dit_candidate_progress_weight", 1.0)) * progress_penalty
            + float(getattr(self.cfg, "dit_candidate_front_weight", 0.5)) * front_penalty
            + float(getattr(self.cfg, "dit_candidate_smooth_weight", 0.05)) * smooth_prev
            + float(getattr(self.cfg, "dit_candidate_temporal_smooth_weight", 0.05)) * temporal_smooth
            + float(getattr(self.cfg, "dit_candidate_action_weight", 0.02)) * action_effort
        )

        selected = torch.argmin(scores, dim=1)
        seq = candidate_seq_t[
            torch.arange(batch_size, device=feat.device),
            selected,
        ]
        return seq, {
            "candidate_scores": scores,
            "selected_candidate": selected,
            "candidate_yaw_angle": yaw_score,
            "candidate_pitch_angle": pitch_score,
            "candidate_final_distance_norm": final_distance_norm,
            "candidate_progress_penalty": progress_penalty,
            "candidate_front_penalty": front_penalty,
            "candidate_smooth_prev": smooth_prev,
            "candidate_temporal_smooth": temporal_smooth,
            "candidate_action_effort": action_effort,
        }

    @torch.no_grad()
    def act(
        self,
        image: torch.Tensor,
        text_tokens: torch.Tensor,
        target_relative: torch.Tensor,
        prev_action: torch.Tensor,
        rssm_state: Optional[RSSMState] = None,
        attention_mask: Optional[torch.Tensor] = None,
        prev_done: Optional[torch.Tensor] = None,
        deterministic: bool = True,
        num_steps: Optional[int] = None,
        instruction: Optional[str] = None,
        save_transformer_attention: bool = False,
        save_predicted_video: bool = False,
        predicted_video_latent_frames: int = 3,
    ) -> Tuple[Dict[str, torch.Tensor], Optional[RSSMState]]:
        if image.ndim != 4:
            raise ValueError("image must have shape [B, C, H, W].")
        if target_relative.ndim != 2:
            raise ValueError("target_relative must have shape [B, D].")
        batch_size = image.size(0)
        images = image.unsqueeze(1)
        target_relative_seq = target_relative.unsqueeze(1)

        encoded = self.encode_sequence(
            images,
            text_tokens,
            target_relative_seq,
            attention_mask,
            instructions=None if instruction is None else [instruction] * batch_size,
        )
        if self.rssm is None:
            post = None
            feat = encoded["obs_embed"].squeeze(1)
        else:
            if rssm_state is None:
                rssm_state = self.initial_state(batch_size, image.device)
            if prev_done is not None:
                rssm_state = self.rssm.reset_state_by_done(rssm_state, prev_done)
            _, post = self.rssm.obs_step(rssm_state, prev_action, encoded["obs_embed"].squeeze(1))
            feat = self.rssm.get_feat(post)
        if self.fastwam is not None:
            candidate_info = None
            if encoded.get("video_latents") is None or encoded.get("text_context") is None:
                raise RuntimeError("Official FastWAM head requires Wan2.2 latents and raw text context.")
            sample_out = self.fastwam.sample_action(
                first_frame_latents=encoded["video_latents"],
                context=encoded["text_context"],
                context_mask=encoded["text_context_mask"],
                action_horizon=max(int(self.cfg.action_sequence_horizon), 1),
                num_steps=num_steps,
                return_attention_maps=save_transformer_attention,
            )
            attention_aux = None
            if isinstance(sample_out, tuple):
                action_sequence_norm, attention_aux = sample_out
            else:
                action_sequence_norm = sample_out
            action_norm = action_sequence_norm[:, 0]
            predicted_video_latents = None
            if save_predicted_video:
                predicted_video_latents = self.fastwam.sample_video(
                    first_frame_latents=encoded["video_latents"],
                    context=encoded["text_context"],
                    context_mask=encoded["text_context_mask"],
                    latent_frames=predicted_video_latent_frames,
                    action=action_sequence_norm,
                    num_steps=num_steps,
                )
        elif self.cfg.use_diffusion_actor:
            candidate_info = None
            if bool(getattr(self.cfg, "dit_candidate_selection", False)):
                if self.rssm is None:
                    raise RuntimeError("DiT candidate selection requires RSSM; disable dit_candidate_selection.")
                action_sequence_norm, candidate_info = self.select_dit_action_sequence(
                    feat,
                    post,
                    prev_action,
                    current_target_relative=target_relative,
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
        if self.fastwam is not None and "predicted_video_latents" in locals() and predicted_video_latents is not None:
            out["predicted_video_latents"] = predicted_video_latents
        if self.fastwam is not None and "attention_aux" in locals() and attention_aux is not None:
            attn = attention_aux.get("last_transformer_attention")
            grid_size = attention_aux.get("video_grid_size")
            if attn is not None and grid_size is not None:
                _, grid_h, grid_w = grid_size
                first_frame_tokens = int(grid_h) * int(grid_w)
                attn_map = attn[..., :first_frame_tokens].mean(dim=(1, 2)).reshape(attn.size(0), int(grid_h), int(grid_w))
                out["last_transformer_attention_map"] = attn_map
                out["last_transformer_attention_raw"] = attn
        if candidate_info is not None:
            out.update(candidate_info)
        return out, post
