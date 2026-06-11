from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from data.action_mapping import norm_action_to_physical
from data.visual_guidance import make_attention_heatmap
from .config import ModelConfig
from .encoders import HeatmapTokenEncoder, TargetTokenEncoder, Wan22TextEncoder, Wan22VAEImageEncoder
from .fusion import CrossAttentionFusion
from .heads import FastWAMHead, TeacherPredictionHeads
from .rssm import RSSM, RSSMState


def migrate_legacy_state_dict_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Map old checkpoint module names to target-relative terminology."""
    replacements = (
        ("privileged_encoder.", "target_token_encoder."),
        ("fusion.priv_proj.", "fusion.target_token_proj."),
        ("prediction_heads.privileged.", "prediction_heads.next_target_relative."),
        ("reference_grounding_visual_proj.", "target_belief_context_proj."),
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
        self.heatmap_token_encoder = HeatmapTokenEncoder(cfg)
        heatmap_grid = max(int(getattr(cfg, "fastwam_heatmap_context_grid", 4)), 1)
        self.fastwam_heatmap_context_grid = heatmap_grid
        self.heatmap_context_proj = nn.Sequential(
            nn.Linear(1, cfg.text_width),
            nn.GELU(),
            nn.Linear(cfg.text_width, cfg.text_width),
        )
        self.target_belief_query_proj = nn.Linear(cfg.image_encoder_dim, cfg.image_encoder_dim)
        self.target_belief_ref_proj = nn.Linear(cfg.image_encoder_dim, cfg.image_encoder_dim)
        self.target_belief_context_proj = nn.Sequential(
            nn.LayerNorm(cfg.image_encoder_dim),
            nn.Linear(cfg.image_encoder_dim, cfg.text_width),
            nn.GELU(),
            nn.Linear(cfg.text_width, cfg.text_width),
        )
        if not (
            bool(getattr(cfg, "use_target_visual_guidance", False))
            and bool(getattr(cfg, "use_attention_heatmap", True))
            and bool(getattr(cfg, "use_heatmap_tensor_encoder", True))
        ):
            for p in self.heatmap_token_encoder.parameters():
                p.requires_grad_(False)
            for p in self.heatmap_context_proj.parameters():
                p.requires_grad_(False)
        if not bool(getattr(cfg, "use_target_belief_tracker", False)):
            for p in self.target_belief_query_proj.parameters():
                p.requires_grad_(False)
            for p in self.target_belief_ref_proj.parameters():
                p.requires_grad_(False)
            for p in self.target_belief_context_proj.parameters():
                p.requires_grad_(False)
        self.fusion = CrossAttentionFusion(cfg)
        self.rssm = RSSM(cfg) if cfg.use_rssm else None
        self.prediction_heads = TeacherPredictionHeads(cfg)
        self.fastwam = FastWAMHead(cfg) if cfg.use_fastwam_mot else None
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

    def _make_heatmap_context_tokens(
        self,
        attention_heatmaps: Optional[torch.Tensor],
        latent_seq_len: int,
        target_device: torch.device,
        target_dtype: torch.dtype,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if attention_heatmaps is None:
            return None, None
        if not (
            bool(getattr(self.cfg, "use_target_visual_guidance", False))
            and bool(getattr(self.cfg, "use_attention_heatmap", True))
            and bool(getattr(self.cfg, "use_heatmap_tensor_encoder", True))
        ):
            return None, None
        if attention_heatmaps.ndim != 5:
            raise ValueError("attention_heatmaps must have shape [B, T, 1, H, W].")
        batch_size = attention_heatmaps.shape[0]
        # FastWAM action inference only receives the current frame. Keep heatmap
        # context aligned with inference and avoid leaking future target positions
        # from the training window.
        current_heatmap = attention_heatmaps[:, 0]
        grid = self.fastwam_heatmap_context_grid
        pooled = F.interpolate(current_heatmap[:, :1].float(), size=(grid, grid), mode="bilinear", align_corners=False)
        scalar_tokens = pooled.flatten(2).transpose(1, 2)
        tokens = self.heatmap_context_proj(
            scalar_tokens.to(device=self.heatmap_context_proj[0].weight.device, dtype=self.heatmap_context_proj[0].weight.dtype)
        )
        tokens = tokens.to(device=target_device, dtype=target_dtype)
        mask = torch.ones(tokens.shape[:2], device=target_device, dtype=torch.bool)
        return tokens, mask

    @staticmethod
    def _token_grid_shape(patch_count: int) -> tuple[int, int]:
        grid_h = int(round(math.sqrt(max(patch_count, 1))))
        grid_w = grid_h
        if grid_h * grid_w != patch_count:
            grid_h = max(int(math.floor(math.sqrt(max(patch_count, 1)))), 1)
            grid_w = int(math.ceil(patch_count / max(grid_h, 1)))
        return grid_h, grid_w

    def _target_heatmap_weights(
        self,
        target_relative: torch.Tensor,
        image_hw: Tuple[int, int],
        patch_count: int,
    ) -> torch.Tensor:
        heatmap = make_attention_heatmap(
            target_relative.float(),
            image_hw=image_hw,
            fov_deg=self.cfg.visual_guidance_fov_deg,
            sigma=self.cfg.attention_heatmap_sigma,
        )
        grid_h, grid_w = self._token_grid_shape(patch_count)
        heatmap = F.interpolate(heatmap[:, :1].float(), size=(grid_h, grid_w), mode="bilinear", align_corners=False)
        weights = heatmap.flatten(2)
        if weights.size(-1) > patch_count:
            weights = weights[..., :patch_count]
        elif weights.size(-1) < patch_count:
            weights = F.pad(weights, (0, patch_count - weights.size(-1)))
        denom = weights.sum(dim=-1, keepdim=True)
        uniform = torch.full_like(weights, 1.0 / max(int(patch_count), 1))
        return torch.where(denom > 1e-6, weights / denom.clamp_min(1e-6), uniform)

    def _pool_reference_target_from_location(
        self,
        visual_tokens: torch.Tensor,
        target_relative: torch.Tensor,
        image_hw: Tuple[int, int],
    ) -> torch.Tensor:
        if visual_tokens.ndim != 3:
            raise ValueError("visual_tokens must have shape [B, N, C].")
        if target_relative.ndim != 2:
            raise ValueError("target_relative must have shape [B, D].")
        patch_tokens = visual_tokens.float()
        weights = self._target_heatmap_weights(target_relative, image_hw, patch_tokens.size(1)).to(patch_tokens.device)
        return torch.sum(patch_tokens * weights.transpose(1, 2), dim=1)

    def _project_target_belief_context(
        self,
        target_token: torch.Tensor,
        target_device: torch.device,
        target_dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if target_token.ndim != 2:
            raise ValueError("target_token must have shape [B, C].")
        tokens = self.target_belief_context_proj(
            target_token.to(
                device=self.target_belief_context_proj[0].weight.device,
                dtype=self.target_belief_context_proj[0].weight.dtype,
            )
        ).unsqueeze(1)
        tokens = tokens * float(getattr(self.cfg, "target_belief_token_scale", 1.0))
        tokens = tokens.to(device=target_device, dtype=target_dtype)
        mask = torch.ones(tokens.shape[:2], device=target_device, dtype=torch.bool)
        return tokens, mask

    def _make_target_belief_from_relative(
        self,
        target_relative: torch.Tensor,
        image_hw: Tuple[int, int],
        patch_count: int,
    ) -> torch.Tensor:
        weights = self._target_heatmap_weights(target_relative, image_hw, patch_count)
        return weights.squeeze(1)

    def _make_belief_context_tokens(
        self,
        belief: torch.Tensor,
        visual_tokens: torch.Tensor,
        target_device: torch.device,
        target_dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if belief.ndim != 2:
            raise ValueError("belief must have shape [B, N].")
        if visual_tokens.ndim != 3:
            raise ValueError("visual_tokens must have shape [B, N, C].")
        if belief.shape != visual_tokens.shape[:2]:
            raise ValueError("belief shape must match visual token batch/patch dimensions.")
        pooled = torch.sum(visual_tokens.float() * belief.unsqueeze(-1).float(), dim=1)
        token, mask = self._project_target_belief_context(
            pooled,
            target_device=target_device,
            target_dtype=target_dtype,
        )
        return token, mask

    def _roll_target_belief_sequence(
        self,
        video_tokens: torch.Tensor,
        target_relative: torch.Tensor,
        image_hw: Tuple[int, int],
        reference_target_relative: Optional[torch.Tensor] = None,
        reference_visual_tokens: Optional[torch.Tensor] = None,
        reference_images: Optional[torch.Tensor] = None,
        target_belief: Optional[torch.Tensor] = None,
        detach_state: bool = False,
    ) -> Dict[str, torch.Tensor]:
        if video_tokens.ndim != 4:
            raise ValueError("video_tokens must have shape [B, T, N, C].")
        if target_relative.ndim != 3:
            raise ValueError("target_relative must have shape [B, T, D].")
        batch_size, latent_seq_len, patch_count, channels = video_tokens.shape
        if reference_target_relative is None:
            raise ValueError("target belief tracker requires reference_target_relative.")
        if reference_visual_tokens is None and reference_images is not None:
            reference_visual_tokens = self.encode_reference_visual_tokens(reference_images)
        if reference_visual_tokens is None:
            raise ValueError("target belief tracker requires reference_images or reference_visual_tokens.")
        reference_token = self.init_target_belief_reference(
            reference_visual_tokens.to(device=video_tokens.device),
            reference_target_relative.to(device=target_relative.device),
            image_hw=image_hw,
        )
        if reference_token.ndim != 2 or reference_token.shape != (batch_size, channels):
            raise ValueError("target belief reference token must have shape [B, C].")

        q_proj = self.target_belief_query_proj
        r_proj = self.target_belief_ref_proj
        temperature = max(float(getattr(self.cfg, "target_belief_temperature", 0.07)), 1e-4)
        update_rate = min(max(float(getattr(self.cfg, "target_belief_update_rate", 0.25)), 0.0), 1.0)
        min_confidence = float(getattr(self.cfg, "target_belief_min_confidence", 0.05))
        motion_weight = float(getattr(self.cfg, "target_belief_motion_weight", 0.25))
        sharpness = max(float(getattr(self.cfg, "target_belief_update_sharpness", 10.0)), 0.0)

        reference_norm = F.normalize(
            r_proj(reference_token.to(device=r_proj.weight.device, dtype=r_proj.weight.dtype)).to(video_tokens.device).float(),
            dim=-1,
        )
        target_indices = torch.linspace(
            0,
            target_relative.size(1) - 1,
            latent_seq_len,
            device=target_relative.device,
        ).round().long()
        has_external_belief = target_belief is not None
        if target_belief is not None:
            if target_belief.ndim == 3 and target_belief.size(1) == 1:
                target_belief = target_belief[:, 0]
            if target_belief.shape != (batch_size, patch_count):
                raise ValueError("target_belief must have shape [B, N].")
            raw_belief = target_belief.to(device=video_tokens.device).float().clamp_min(0.0)
            denom = raw_belief.sum(dim=-1, keepdim=True)
            uniform = torch.full_like(raw_belief, 1.0 / max(patch_count, 1))
            belief = torch.where(denom > 1e-6, raw_belief / denom.clamp_min(1e-6), uniform)
        else:
            belief = torch.full(
                (batch_size, patch_count),
                1.0 / max(patch_count, 1),
                device=video_tokens.device,
                dtype=torch.float32,
            )

        gt_beliefs = []
        pred_beliefs = []
        confidences = []
        entropies = []
        for idx in range(latent_seq_len):
            current = video_tokens[:, idx]
            query = q_proj(current.to(device=q_proj.weight.device, dtype=q_proj.weight.dtype)).to(video_tokens.device).float()
            query = F.normalize(query, dim=-1)
            appearance_logits = (query * reference_norm.unsqueeze(1)).sum(dim=-1) / temperature
            motion_prior = belief.clamp_min(1e-6).log()
            fused_logits = appearance_logits + motion_weight * motion_prior
            pred = F.softmax(fused_logits, dim=-1)
            pred_conf = pred.amax(dim=-1)
            threshold = max(min_confidence, 1.0 / max(patch_count, 1))
            if sharpness > 0.0:
                gate = torch.sigmoid((pred_conf - threshold) * sharpness).view(batch_size, 1)
            else:
                gate = torch.ones(batch_size, 1, device=video_tokens.device, dtype=torch.float32)
            if has_external_belief or idx > 0:
                gate = gate * update_rate
            belief = F.normalize(gate * pred + (1.0 - gate) * belief, p=1, dim=-1)
            if detach_state:
                belief = belief.detach()
            pred_beliefs.append(belief)
            confidences.append(belief.amax(dim=-1))
            entropies.append(-(belief * belief.clamp_min(1e-6).log()).sum(dim=-1))
            gt_beliefs.append(
                self._make_target_belief_from_relative(
                    target_relative[:, target_indices[idx]],
                    image_hw,
                    patch_count,
                ).to(video_tokens.device)
            )

        return {
            "reference_token": reference_token,
            "target_belief": belief,
            "target_belief_sequence": torch.stack(pred_beliefs, dim=1),
            "target_belief_gt": torch.stack(gt_beliefs, dim=1),
            "target_belief_confidence": torch.stack(confidences, dim=1),
            "target_belief_entropy": torch.stack(entropies, dim=1),
        }

    def init_target_belief_reference(
        self,
        reference_visual_tokens: torch.Tensor,
        reference_target_relative: torch.Tensor,
        image_hw: Tuple[int, int],
    ) -> torch.Tensor:
        if reference_target_relative.ndim == 3:
            reference_target_relative = reference_target_relative[:, 0]
        return self._pool_reference_target_from_location(reference_visual_tokens, reference_target_relative, image_hw)

    @torch.no_grad()
    def encode_reference_visual_tokens(self, reference_image: torch.Tensor) -> torch.Tensor:
        if reference_image.ndim != 4:
            raise ValueError("reference_image must have shape [B, C, H, W].")
        if self.cfg.use_wan22_encoders:
            latents = self.image_encoder.encode_video_latents(reference_image.unsqueeze(1))
            return latents.permute(0, 2, 3, 4, 1).reshape(latents.size(0), latents.size(2), -1, latents.size(1))[:, 0].float()
        raise RuntimeError("Target belief reference visual tokens require cfg.use_wan22_encoders=True.")

    def encode_sequence(
        self,
        images: torch.Tensor,
        text_tokens: torch.Tensor,
        target_relative: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        attention_heatmaps: Optional[torch.Tensor] = None,
        instructions: Optional[list[str]] = None,
        video_latents: Optional[torch.Tensor] = None,
        reference_target_relative: Optional[torch.Tensor] = None,
        reference_images: Optional[torch.Tensor] = None,
        reference_visual_tokens: Optional[torch.Tensor] = None,
        target_belief: Optional[torch.Tensor] = None,
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
        # features; heatmap guidance is appended to text context instead of
        # being added into these visual tokens.
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
        if attention_heatmaps is not None and attention_heatmaps.shape[:2] != (batch_size, seq_len):
            raise ValueError("attention_heatmaps batch/time shape must match images.")
        target_belief_out = None
        if bool(getattr(self.cfg, "use_target_belief_tracker", False)):
            target_belief_out = self._roll_target_belief_sequence(
                video_tokens,
                target_relative,
                image_hw=(images.shape[-2], images.shape[-1]),
                reference_target_relative=reference_target_relative,
                reference_visual_tokens=reference_visual_tokens,
                reference_images=reference_images,
                target_belief=target_belief,
                detach_state=not self.training,
            )
        target_patch_bias = None
        # The policy no longer receives the target-relative label as input. It is
        # still passed here for supervision and target-projection guidance, while
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
            target_patch_bias=target_patch_bias,
        )
        fastwam_context = text_once
        fastwam_context_mask = text_mask_once
        heatmap_context, heatmap_context_mask = self._make_heatmap_context_tokens(
            attention_heatmaps,
            latent_seq_len=latent_seq_len,
            target_device=fastwam_context.device,
            target_dtype=fastwam_context.dtype,
        )
        if heatmap_context is not None and heatmap_context_mask is not None:
            fastwam_context = torch.cat([fastwam_context, heatmap_context], dim=1)
            fastwam_context_mask = torch.cat([fastwam_context_mask, heatmap_context_mask], dim=1)
        if bool(getattr(self.cfg, "use_target_belief_tracker", False)):
            if target_belief_out is None:
                raise RuntimeError("target belief tracker was not initialized.")
            belief_context, belief_context_mask = self._make_belief_context_tokens(
                target_belief_out["target_belief_sequence"][:, 0],
                video_tokens[:, 0],
                target_device=fastwam_context.device,
                target_dtype=fastwam_context.dtype,
            )
            fastwam_context = torch.cat([fastwam_context, belief_context], dim=1)
            fastwam_context_mask = torch.cat([fastwam_context_mask, belief_context_mask], dim=1)

        encoded_out = {
            "obs_embed": obs_embed.view(batch_size, latent_seq_len, -1),
            "fused_tokens": fused_tokens.view(batch_size, latent_seq_len, fused_tokens.size(1), fused_tokens.size(2)),
            "video_latents": video_latents,
            "text_context": fastwam_context,
            "text_context_mask": fastwam_context_mask,
            "target_patch_bias": (
                target_patch_bias.view(batch_size, latent_seq_len, -1) if target_patch_bias is not None else None
            ),
        }
        if target_belief_out is not None:
            encoded_out.update(
                {
                    "target_belief": target_belief_out["target_belief"],
                    "target_belief_sequence": target_belief_out["target_belief_sequence"],
                    "target_belief_gt": target_belief_out["target_belief_gt"],
                    "target_belief_confidence": target_belief_out["target_belief_confidence"],
                    "target_belief_entropy": target_belief_out["target_belief_entropy"],
                }
            )
        return encoded_out

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
        instructions: Optional[list[str]] = None,
        video_latents: Optional[torch.Tensor] = None,
        reference_target_relative: Optional[torch.Tensor] = None,
        reference_images: Optional[torch.Tensor] = None,
        reference_visual_tokens: Optional[torch.Tensor] = None,
        target_belief: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        encoded = self.encode_sequence(
            images,
            text_tokens,
            target_relative,
            attention_mask,
            attention_heatmaps=attention_heatmaps,
            instructions=instructions,
            video_latents=video_latents,
            reference_target_relative=reference_target_relative,
            reference_images=reference_images,
            reference_visual_tokens=reference_visual_tokens,
            target_belief=target_belief,
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

        for key in (
            "target_belief",
            "target_belief_sequence",
            "target_belief_gt",
            "target_belief_confidence",
            "target_belief_entropy",
        ):
            if encoded.get(key) is not None:
                out[key] = encoded[key]
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
        attention_heatmap: Optional[torch.Tensor] = None,
        prev_done: Optional[torch.Tensor] = None,
        deterministic: bool = True,
        num_steps: Optional[int] = None,
        instruction: Optional[str] = None,
        save_transformer_attention: bool = False,
        save_predicted_video: bool = False,
        predicted_video_latent_frames: int = 3,
        latent_mpc: bool = False,
        latent_mpc_candidate_count: Optional[int] = None,
        target_next_relative: Optional[torch.Tensor] = None,
        reference_target_relative: Optional[torch.Tensor] = None,
        reference_image: Optional[torch.Tensor] = None,
        reference_visual_tokens: Optional[torch.Tensor] = None,
        target_belief: Optional[torch.Tensor] = None,
    ) -> Tuple[Dict[str, torch.Tensor], Optional[RSSMState]]:
        if image.ndim != 4:
            raise ValueError("image must have shape [B, C, H, W].")
        if target_relative.ndim != 2:
            raise ValueError("target_relative must have shape [B, D].")
        batch_size = image.size(0)
        images = image.unsqueeze(1)
        target_relative_seq = target_relative.unsqueeze(1)
        attention_heatmaps = None if attention_heatmap is None else attention_heatmap.unsqueeze(1)

        encoded = self.encode_sequence(
            images,
            text_tokens,
            target_relative_seq,
            attention_mask,
            attention_heatmaps=attention_heatmaps,
            instructions=None if instruction is None else [instruction] * batch_size,
            reference_target_relative=reference_target_relative,
            reference_images=reference_image,
            reference_visual_tokens=reference_visual_tokens,
            target_belief=target_belief,
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
            candidate_info = None
            if latent_mpc or bool(getattr(self.cfg, "use_latent_mpc", False)):
                chosen, candidate_info = self.select_fastwam_latent_mpc_action_sequence(
                    first_frame_latents=encoded["video_latents"],
                    context=encoded["text_context"],
                    context_mask=encoded["text_context_mask"],
                    action_sequence_norm=action_sequence_norm,
                    target_relative=target_relative,
                    target_next_relative=target_next_relative,
                    num_steps=num_steps,
                    candidate_count=latent_mpc_candidate_count,
                )
                action_sequence_norm = chosen
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
        for key in (
            "target_belief",
            "target_belief_sequence",
            "target_belief_confidence",
            "target_belief_entropy",
        ):
            if encoded.get(key) is not None:
                out[key] = encoded[key]
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

    @torch.no_grad()
    def select_fastwam_latent_mpc_action_sequence(
        self,
        first_frame_latents: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        action_sequence_norm: torch.Tensor,
        target_relative: torch.Tensor,
        target_next_relative: Optional[torch.Tensor],
        num_steps: Optional[int],
        candidate_count: Optional[int] = None,
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        count = max(int(candidate_count or getattr(self.cfg, "latent_mpc_candidate_count", 4)), 1)
        if count <= 1:
            return action_sequence_norm, {
                "latent_mpc_scores": torch.zeros(action_sequence_norm.size(0), 1, device=action_sequence_norm.device),
                "latent_mpc_selected": torch.zeros(action_sequence_norm.size(0), dtype=torch.long, device=action_sequence_norm.device),
            }
        b, horizon, action_dim = action_sequence_norm.shape
        candidates = [action_sequence_norm]
        for _ in range(count - 1):
            sample = self.fastwam.sample_action(
                first_frame_latents=first_frame_latents,
                context=context,
                context_mask=context_mask,
                action_horizon=horizon,
                num_steps=num_steps,
                return_attention_maps=False,
            )
            if isinstance(sample, tuple):
                sample = sample[0]
            candidates.append(sample)
        cand = torch.stack(candidates, dim=1)
        flat_cand = cand.reshape(b * count, horizon, action_dim)
        latent_frames = max(int(getattr(self.cfg, "latent_mpc_latent_frames", 3)), 2)
        first_frame_rep = first_frame_latents.unsqueeze(1).expand(
            b,
            count,
            *first_frame_latents.shape[1:],
        ).reshape(b * count, *first_frame_latents.shape[1:])
        context_rep = context.unsqueeze(1).expand(b, count, *context.shape[1:]).reshape(b * count, *context.shape[1:])
        context_mask_rep = context_mask.unsqueeze(1).expand(b, count, *context_mask.shape[1:]).reshape(b * count, *context_mask.shape[1:])
        video_steps = max(int(getattr(self.cfg, "latent_mpc_video_sampling_steps", 4)), 1)
        pred_video = self.fastwam.sample_video(
            first_frame_latents=first_frame_rep,
            context=context_rep,
            context_mask=context_mask_rep,
            latent_frames=latent_frames,
            action=flat_cand,
            num_steps=video_steps,
        )
        _, c_lat, _, h_lat, w_lat = pred_video.shape
        anchor_heatmap = make_attention_heatmap(
            target_relative.float(),
            image_hw=(h_lat, w_lat),
            fov_deg=self.cfg.visual_guidance_fov_deg,
            sigma=self.cfg.attention_heatmap_sigma,
        ).flatten(2)
        anchor_weight = anchor_heatmap / anchor_heatmap.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        current_tokens = first_frame_latents[:, :, 0].permute(0, 2, 3, 1).reshape(b, h_lat * w_lat, c_lat).float()
        anchor = torch.sum(current_tokens * anchor_weight.transpose(1, 2), dim=1)
        anchor = F.normalize(anchor, dim=-1)

        pred_video_bc = pred_video.reshape(b, count, c_lat, latent_frames, h_lat, w_lat)
        future_video = pred_video_bc[:, :, :, 1:]
        if future_video.numel() > 0:
            future_tokens = future_video.permute(0, 1, 3, 4, 5, 2).reshape(b, count, -1, c_lat).float()
            future_tokens = F.normalize(future_tokens, dim=-1)
            target_similarity = (future_tokens * anchor[:, None, None, :]).sum(dim=-1).amax(dim=-1)
            visual_cost = 1.0 - target_similarity
            first_frame = pred_video_bc[:, :, :, :1]
            visual_change = (future_video.float() - first_frame.float()).pow(2).mean(dim=(2, 3, 4, 5))
        else:
            visual_cost = torch.zeros(b, count, device=cand.device, dtype=torch.float32)
            visual_change = torch.zeros(b, count, device=cand.device, dtype=torch.float32)
        visual_change_norm = visual_change / visual_change.detach().mean(dim=1, keepdim=True).clamp_min(1e-6)
        cand_phys = norm_action_to_physical(
            cand,
            max_vel=self.cfg.max_vel,
            max_yaw_rate=self.cfg.max_yaw_rate,
            max_speed_norm=self.cfg.max_speed_norm,
        )
        cur = target_relative.float().unsqueeze(1)
        if target_next_relative is None:
            target_motion = torch.zeros_like(cur)
        else:
            target_motion = (target_next_relative.float() - target_relative.float()).unsqueeze(1)
        first_move = cand_phys[:, :, 0, :3]
        pred_next = cur + target_motion - first_move
        distance = torch.linalg.norm(pred_next, dim=-1)
        smooth = torch.linalg.norm(cand[:, :, 0] - action_sequence_norm[:, 0].unsqueeze(1), dim=-1)
        effort = torch.linalg.norm(cand[:, :, 0], dim=-1)
        scores = (
            float(getattr(self.cfg, "latent_mpc_distance_weight", 1.0)) * distance
            + float(getattr(self.cfg, "latent_mpc_smooth_weight", 0.05)) * smooth
            + float(getattr(self.cfg, "latent_mpc_action_weight", 0.02)) * effort
            + float(getattr(self.cfg, "latent_mpc_visual_weight", 0.1)) * visual_cost
        )
        selected = torch.argmin(scores, dim=1)
        chosen = cand[torch.arange(b, device=cand.device), selected]
        return chosen, {
            "latent_mpc_scores": scores,
            "latent_mpc_selected": selected,
            "latent_mpc_distance": distance,
            "latent_mpc_smooth": smooth,
            "latent_mpc_action_effort": effort,
            "latent_mpc_visual_cost": visual_cost,
            "latent_mpc_visual_change": visual_change_norm,
        }
