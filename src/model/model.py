from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from data.action_mapping import norm_action_to_physical
from data.visual_guidance import project_body_to_image_features
from .config import ModelConfig
from .encoders import CLIPTextEncoder, DINOv2ImageEncoder, TargetTokenEncoder
from .fusion import CrossAttentionFusion
from .heads import DiTActionHead, DirectActionHead, TeacherPredictionHeads
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
    """Teacher world model with real CLIP text encoder and no state input."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.image_encoder = DINOv2ImageEncoder(cfg)
        self.text_encoder = CLIPTextEncoder(cfg)
        self.target_token_encoder = TargetTokenEncoder(cfg)
        self.fusion = CrossAttentionFusion(cfg)
        self.rssm = RSSM(cfg)
        self.prediction_heads = TeacherPredictionHeads(cfg)
        self.actor = DiTActionHead(cfg)
        self.direct_action = DirectActionHead(cfg)
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

    def _make_target_patch_bias(
        self,
        target_relative: torch.Tensor,
        num_image_tokens: int,
        image_hw: Tuple[int, int],
    ) -> Optional[torch.Tensor]:
        if not bool(getattr(self.cfg, "use_target_visual_guidance", False)):
            return None
        if not bool(getattr(self.cfg, "use_attention_heatmap", True)):
            return None
        if num_image_tokens <= 1:
            return None

        patch_count = num_image_tokens - 1
        grid_h = int(round(math.sqrt(patch_count)))
        grid_w = grid_h
        if grid_h * grid_w != patch_count:
            # DINOv2-base at 224 normally gives 1 CLS + 16x16 patches. If a
            # different resolution/model is used, fall back to a near-square grid.
            grid_h = int(math.floor(math.sqrt(patch_count)))
            grid_w = int(math.ceil(patch_count / max(grid_h, 1)))

        proj = project_body_to_image_features(
            target_relative,
            image_hw=image_hw,
            fov_deg=self.cfg.visual_guidance_fov_deg,
        )
        xy = proj[..., :2]
        visible = proj[..., 2:3]
        dtype = target_relative.dtype if target_relative.is_floating_point() else torch.float32
        xs = torch.linspace(0.0, 1.0, grid_w, device=target_relative.device, dtype=dtype)
        ys = torch.linspace(0.0, 1.0, grid_h, device=target_relative.device, dtype=dtype)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        center_x = xy[..., 0].to(dtype)[..., None, None]
        center_y = xy[..., 1].to(dtype)[..., None, None]
        sigma = max(float(getattr(self.cfg, "attention_heatmap_sigma", 0.08)), 1e-4)
        dist2 = (grid_x - center_x).pow(2) + (grid_y - center_y).pow(2)
        patch_bias = torch.exp(-0.5 * dist2 / (sigma * sigma))
        out_of_view_scale = float(getattr(self.cfg, "heatmap_out_of_view_bias_scale", 0.5))
        patch_bias = patch_bias * (visible.to(dtype)[..., None] + (1.0 - visible.to(dtype)[..., None]) * out_of_view_scale)
        patch_bias = patch_bias.reshape(target_relative.size(0), -1)
        if patch_bias.size(1) > patch_count:
            patch_bias = patch_bias[:, :patch_count]
        elif patch_bias.size(1) < patch_count:
            patch_bias = torch.nn.functional.pad(patch_bias, (0, patch_count - patch_bias.size(1)))
        cls_bias = torch.zeros(target_relative.size(0), 1, device=target_relative.device, dtype=dtype)
        return torch.cat([cls_bias, patch_bias], dim=1)

    def encode_sequence(
        self,
        images: torch.Tensor,
        text_tokens: torch.Tensor,
        target_relative: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        attention_heatmaps: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if images.ndim != 5:
            raise ValueError("images must have shape [B, T, C, H, W].")
        if target_relative.ndim != 3:
            raise ValueError("target_relative must have shape [B, T, D].")
        batch_size, seq_len, *_ = images.shape
        if target_relative.shape[:2] != (batch_size, seq_len):
            raise ValueError("target_relative batch/time shape must match images.")

        single_text = text_tokens.ndim == 2 or (text_tokens.ndim == 3 and text_tokens.size(1) == 1)
        text_tokens_in = text_tokens
        attention_mask_in = attention_mask
        text_tokens, attention_mask = self._prepare_text(text_tokens, batch_size, seq_len, attention_mask)

        flat_images = images.reshape(batch_size * seq_len, *images.shape[2:])
        flat_target = target_relative.reshape(batch_size * seq_len, target_relative.size(-1))
        _, image_tokens = self.image_encoder(flat_images)
        if attention_heatmaps is not None and attention_heatmaps.shape[:2] != (batch_size, seq_len):
            raise ValueError("attention_heatmaps batch/time shape must match images.")
        target_patch_bias = self._make_target_patch_bias(
            flat_target,
            num_image_tokens=image_tokens.size(1),
            image_hw=(images.shape[-2], images.shape[-1]),
        )
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
        # The policy no longer receives the target-relative label as input. It is
        # still passed here for supervision and target-projection guidance, while
        # fusion receives only a learned constant null-target token.
        target_token = self.target_token_encoder(torch.zeros_like(flat_target))
        obs_embed, fused_tokens = self.fusion(
            image_tokens,
            text_seq,
            target_token,
            target_patch_bias=target_patch_bias,
        )

        return {
            "obs_embed": obs_embed.view(batch_size, seq_len, -1),
            "fused_tokens": fused_tokens.view(batch_size, seq_len, fused_tokens.size(1), fused_tokens.size(2)),
            "target_patch_bias": (
                target_patch_bias.view(batch_size, seq_len, -1) if target_patch_bias is not None else None
            ),
        }

    def forward(
        self,
        images: torch.Tensor,
        text_tokens: torch.Tensor,
        target_relative: torch.Tensor,
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
            target_relative,
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
        if expert_action is not None:
            next_target_relative = self._predict_next_target_from_action(posts, expert_action.float())
            prior_next_target_relative = self._predict_next_target_from_action(priors, expert_action.float())
            preds = {"next_target_relative": next_target_relative}
            prior_preds = {"prior_next_target_relative": prior_next_target_relative}
        else:
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

        return out

    def _repeat_rssm_state(self, state: RSSMState, repeat: int) -> RSSMState:
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
        attention_heatmap: Optional[torch.Tensor] = None,
        prev_done: Optional[torch.Tensor] = None,
        deterministic: bool = True,
        num_steps: Optional[int] = None,
    ) -> Tuple[Dict[str, torch.Tensor], RSSMState]:
        if image.ndim != 4:
            raise ValueError("image must have shape [B, C, H, W].")
        if target_relative.ndim != 2:
            raise ValueError("target_relative must have shape [B, D].")
        batch_size = image.size(0)
        if rssm_state is None:
            rssm_state = self.initial_state(batch_size, image.device)
        if prev_done is not None:
            rssm_state = self.rssm.reset_state_by_done(rssm_state, prev_done)

        images = image.unsqueeze(1)
        target_relative_seq = target_relative.unsqueeze(1)
        attention_heatmaps = None if attention_heatmap is None else attention_heatmap.unsqueeze(1)

        encoded = self.encode_sequence(
            images,
            text_tokens,
            target_relative_seq,
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
        if candidate_info is not None:
            out.update(candidate_info)
        return out, post
