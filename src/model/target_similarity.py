from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from data.visual_guidance import make_attention_heatmap, project_body_to_image

from .config import ModelConfig


class TargetSimilarityGuidance(nn.Module):
    """History-aware target similarity tokens and supervision tensors.

    The target identity is a soft crop over the first visible target frame. Each
    frame then produces a cosine similarity heatmap against that identity. A
    short causal memory applies temporal decay to recent similarity
    distributions before the module emits compact context tokens for FastWAM.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        hidden_dim = int(getattr(cfg, "target_similarity_hidden_dim", 256))
        patch_size = max(int(getattr(cfg, "target_similarity_patch_size", 16)), 1)
        vae_dim = int(getattr(cfg, "image_encoder_dim", hidden_dim))
        self.patch_encoder = nn.Sequential(
            nn.Conv2d(3, hidden_dim, kernel_size=patch_size, stride=patch_size, bias=False),
            nn.GroupNorm(1, hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
        )
        self.vae_feature_proj = nn.Sequential(
            nn.LayerNorm(vae_dim),
            nn.Linear(vae_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        text_width = int(getattr(cfg, "text_width", hidden_dim))
        self.text_width = text_width
        self.target_token_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, text_width),
        )
        self.history_token_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim + 3),
            nn.Linear(hidden_dim + 3, text_width),
            nn.GELU(),
            nn.Linear(text_width, text_width),
        )
        self.similarity_token_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim + 4),
            nn.Linear(hidden_dim + 4, text_width),
            nn.GELU(),
            nn.Linear(text_width, text_width),
        )
        cond_dim = max(int(getattr(cfg, "target_similarity_condition_dim", 5)), 1)
        self.condition_token_proj = nn.Sequential(
            nn.LayerNorm(cond_dim),
            nn.Linear(cond_dim, text_width),
            nn.GELU(),
            nn.Linear(text_width, text_width),
        )
        context_mode = self._context_mode()
        if context_mode != "dense":
            for p in self.target_token_proj.parameters():
                p.requires_grad_(False)
            for p in self.history_token_proj.parameters():
                p.requires_grad_(False)
            for p in self.similarity_token_proj.parameters():
                p.requires_grad_(False)
        if context_mode != "camera_yz_text":
            for p in self.condition_token_proj.parameters():
                p.requires_grad_(False)

    def _context_mode(self) -> str:
        mode = str(getattr(self.cfg, "target_similarity_context_mode", "dense")).lower()
        aliases = {
            "full": "dense",
            "grid": "dense",
        }
        return aliases.get(mode, mode)

    def _encode_rgb_patches(self, images: torch.Tensor) -> tuple[torch.Tensor, Tuple[int, int]]:
        if images.ndim != 5:
            raise ValueError("images must have shape [B, T, C, H, W].")
        b, t, c, h, w = images.shape
        if c != 3:
            raise ValueError(f"target similarity guidance expects RGB images, got C={c}.")
        param = next(self.patch_encoder.parameters())
        x = images.to(device=param.device, dtype=torch.float32)
        if x.numel() > 0 and float(x.detach().amax().cpu()) > 1.5:
            x = x / 255.0
        x = x.to(dtype=param.dtype)
        feat = self.patch_encoder(x.reshape(b * t, c, h, w))
        gh, gw = int(feat.size(-2)), int(feat.size(-1))
        feat = feat.permute(0, 2, 3, 1).reshape(b, t, gh * gw, feat.size(1))
        return F.normalize(feat, dim=-1), (gh, gw)

    def _encode_vae_latents(self, video_latents: torch.Tensor) -> tuple[torch.Tensor, Tuple[int, int]]:
        if video_latents.ndim != 5:
            raise ValueError("video_latents must have shape [B, C, T, H, W].")
        b, c, t, gh, gw = video_latents.shape
        param = next(self.vae_feature_proj.parameters())
        feat = video_latents.permute(0, 2, 3, 4, 1).reshape(b, t, gh * gw, c)
        feat = self.vae_feature_proj(feat.to(device=param.device, dtype=param.dtype))
        return F.normalize(feat, dim=-1), (int(gh), int(gw))

    def _encode_features(
        self,
        images: torch.Tensor,
        video_latents: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, Tuple[int, int]]:
        source = str(getattr(self.cfg, "target_similarity_feature_source", "wan_vae_latent")).lower()
        if source in {"wan_vae_latent", "vae_latent", "latent"} and video_latents is not None:
            return self._encode_vae_latents(video_latents)
        return self._encode_rgb_patches(images)

    @staticmethod
    def _align_target_relative(target_relative: torch.Tensor, length: int) -> torch.Tensor:
        if target_relative.size(1) == length:
            return target_relative
        if target_relative.size(1) <= 0:
            raise ValueError("target_relative must have at least one timestep.")
        idx = torch.linspace(
            0,
            target_relative.size(1) - 1,
            length,
            device=target_relative.device,
        ).round().long()
        return target_relative[:, idx]

    @staticmethod
    def _normalize_heatmap(heatmap: torch.Tensor) -> torch.Tensor:
        flat = heatmap.reshape(*heatmap.shape[:2], -1)
        return flat / flat.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)

    def _camera_offset_body(self) -> Tuple[float, float, float]:
        raw = getattr(self.cfg, "target_similarity_camera_offset_body", (0.46, 0.0, 0.0))
        if raw is None:
            return (0.0, 0.0, 0.0)
        vals = list(raw)
        if len(vals) < 3:
            vals = vals + [0.0] * (3 - len(vals))
        return (float(vals[0]), float(vals[1]), float(vals[2]))

    def _target_condition_from_center(
        self,
        pred_center: torch.Tensor,
        confidence: torch.Tensor,
        pred_visible: torch.Tensor,
        center_velocity: torch.Tensor,
        grid_hw: Tuple[int, int],
    ) -> torch.Tensor:
        primary = self._target_condition_primary_from_center(pred_center, grid_hw)
        return self._target_condition_from_primary(
            primary,
            confidence,
            pred_visible,
            center_velocity,
        )

    def _target_condition_primary_from_center(
        self,
        pred_center: torch.Tensor,
        grid_hw: Tuple[int, int],
    ) -> torch.Tensor:
        mode = str(getattr(self.cfg, "target_similarity_condition_mode", "image_center")).strip().lower()
        if mode in {"image_center", "center", "pixel_center", "normalized_center"}:
            return torch.cat(
                [
                    pred_center[..., :1] - 0.5,
                    pred_center[..., 1:2] - 0.5,
                ],
                dim=-1,
            )
        elif mode in {"camera_yz", "camera_bearing", "bearing", "ray_yz"}:
            gh, gw = grid_hw
            half_fov = math.radians(float(getattr(self.cfg, "target_similarity_fov_deg", 90.0))) * 0.5
            tan_half_h = max(math.tan(half_fov), 1.0e-6)
            tan_half_v = tan_half_h * (float(gh) / max(float(gw), 1.0))
            return torch.cat(
                [
                    (pred_center[..., :1] - 0.5) * (2.0 * tan_half_h),
                    (pred_center[..., 1:2] - 0.5) * (2.0 * tan_half_v),
                ],
                dim=-1,
            )
        else:
            raise ValueError(
                f"Unsupported target_similarity_condition_mode={mode!r}; "
                "expected 'image_center' or 'camera_yz'."
            )

    def _target_condition_primary_from_gt(
        self,
        target_relative: torch.Tensor,
        gt_center: torch.Tensor,
        grid_hw: Tuple[int, int],
        camera_offset_body: Tuple[float, float, float],
    ) -> torch.Tensor:
        mode = str(getattr(self.cfg, "target_similarity_condition_mode", "image_center")).strip().lower()
        if mode in {"image_center", "center", "pixel_center", "normalized_center"}:
            return torch.cat(
                [
                    gt_center[..., :1] - 0.5,
                    gt_center[..., 1:2] - 0.5,
                ],
                dim=-1,
            )
        if mode in {"camera_yz", "camera_bearing", "bearing", "ray_yz"}:
            target_camera = target_relative[..., :3].clone()
            offset = target_camera.new_tensor(list(camera_offset_body)[:3]).view(1, 1, 3)
            target_camera = target_camera - offset
            x = target_camera[..., 0:1]
            y = target_camera[..., 1:2]
            z = target_camera[..., 2:3]
            eps = torch.full_like(x, 1.0e-4)
            x_safe = torch.where(x.abs() > eps, x, torch.where(x >= 0.0, eps, -eps))
            gh, gw = grid_hw
            half_fov = math.radians(float(getattr(self.cfg, "target_similarity_fov_deg", 90.0))) * 0.5
            tan_half_h = max(math.tan(half_fov), 1.0e-6)
            tan_half_v = tan_half_h * (float(gh) / max(float(gw), 1.0))
            limits = target_camera.new_tensor([tan_half_h, tan_half_v]).view(1, 1, 2)
            primary = torch.cat([y / x_safe, z / x_safe], dim=-1)
            return primary.clamp(-limits, limits)
        raise ValueError(
            f"Unsupported target_similarity_condition_mode={mode!r}; "
            "expected 'image_center' or 'camera_yz'."
        )

    @staticmethod
    def _target_condition_from_primary(
        primary: torch.Tensor,
        confidence: torch.Tensor,
        pred_visible: torch.Tensor,
        center_velocity: torch.Tensor,
    ) -> torch.Tensor:
        return torch.cat(
            [
                primary.to(dtype=confidence.dtype),
                confidence,
                pred_visible.to(confidence.dtype),
                center_velocity.to(confidence.dtype),
            ],
            dim=-1,
        )

    def _uses_last_good_condition(self) -> bool:
        context_mode = self._context_mode()
        condition_mode = str(getattr(self.cfg, "target_similarity_condition_mode", "image_center")).strip().lower()
        return context_mode == "camera_yz_text" and condition_mode in {
            "camera_yz",
            "camera_bearing",
            "bearing",
            "ray_yz",
        }

    def _condition_primary_limits(self, grid_hw: Tuple[int, int], dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        mode = str(getattr(self.cfg, "target_similarity_condition_mode", "image_center")).strip().lower()
        if mode in {"camera_yz", "camera_bearing", "bearing", "ray_yz"}:
            gh, gw = grid_hw
            half_fov = math.radians(float(getattr(self.cfg, "target_similarity_fov_deg", 90.0))) * 0.5
            tan_half_h = max(math.tan(half_fov), 1.0e-6)
            tan_half_v = tan_half_h * (float(gh) / max(float(gw), 1.0))
            return torch.tensor([2.0 * tan_half_h, 2.0 * tan_half_v], device=device, dtype=dtype)
        return torch.tensor([0.75, 0.75], device=device, dtype=dtype)

    def _target_condition_with_last_good(
        self,
        primary: torch.Tensor,
        confidence: torch.Tensor,
        visible: torch.Tensor,
        grid_hw: Tuple[int, int],
        memory_state: Optional[Dict[str, torch.Tensor]],
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        b, t, _ = primary.shape
        device = primary.device
        dtype = primary.dtype
        has_last = torch.zeros(b, 1, device=device, dtype=torch.bool)
        last_primary = torch.zeros(b, 2, device=device, dtype=dtype)
        last_velocity = torch.zeros(b, 2, device=device, dtype=dtype)
        lost_count = torch.zeros(b, 1, device=device, dtype=dtype)

        if memory_state is not None:
            prev_primary = memory_state.get("last_good_primary")
            prev_velocity = memory_state.get("last_good_velocity")
            prev_has_last = memory_state.get("has_last_good")
            prev_lost_count = memory_state.get("lost_count")
            if torch.is_tensor(prev_primary) and prev_primary.shape == (b, 2):
                last_primary = prev_primary.to(device=device, dtype=dtype)
                has_last = torch.ones_like(has_last)
            if torch.is_tensor(prev_velocity) and prev_velocity.shape == (b, 2):
                last_velocity = prev_velocity.to(device=device, dtype=dtype)
            if torch.is_tensor(prev_has_last):
                prev_has_last = prev_has_last.to(device=device, dtype=torch.bool)
                if prev_has_last.ndim == 1:
                    prev_has_last = prev_has_last.unsqueeze(-1)
                if prev_has_last.shape == (b, 1):
                    has_last = has_last & prev_has_last
            if torch.is_tensor(prev_lost_count):
                prev_lost_count = prev_lost_count.to(device=device, dtype=dtype)
                if prev_lost_count.ndim == 1:
                    prev_lost_count = prev_lost_count.unsqueeze(-1)
                if prev_lost_count.shape == (b, 1):
                    lost_count = prev_lost_count

        primary_limit = self._condition_primary_limits(grid_hw, dtype, device).view(1, 2)
        max_extrapolate_steps = float(max(int(getattr(self.cfg, "target_similarity_history_size", 4)), 1))
        cond_parts = []
        for i in range(t):
            current_primary = primary[:, i]
            current_confidence = confidence[:, i]
            current_visible = visible[:, i].view(b, 1).to(device=device, dtype=torch.bool)
            prev_has_last = has_last
            new_velocity = torch.where(
                prev_has_last.expand(-1, 2),
                current_primary - last_primary,
                torch.zeros_like(current_primary),
            )

            next_lost_count = torch.where(
                current_visible,
                torch.zeros_like(lost_count),
                lost_count + 1.0,
            )
            extrapolate_steps = next_lost_count.clamp(max=max_extrapolate_steps)
            search_primary = last_primary + last_velocity * extrapolate_steps
            search_primary = search_primary.clamp(-primary_limit, primary_limit)
            cond_primary = torch.where(
                current_visible.expand(-1, 2),
                current_primary,
                torch.where(prev_has_last.expand(-1, 2), search_primary, current_primary),
            )
            memory_confidence = (1.0 / (1.0 + next_lost_count)).to(dtype=dtype)
            cond_confidence = torch.where(
                current_visible,
                current_confidence,
                torch.where(prev_has_last, memory_confidence, torch.zeros_like(memory_confidence)),
            )
            cond_visible = current_visible.to(dtype=dtype)
            cond_velocity_vec = torch.where(
                current_visible.expand(-1, 2),
                new_velocity,
                torch.where(prev_has_last.expand(-1, 2), last_velocity, torch.zeros_like(last_velocity)),
            )
            cond_velocity = torch.linalg.norm(cond_velocity_vec.float(), dim=-1, keepdim=True).to(dtype=dtype)
            cond_parts.append(
                self._target_condition_from_primary(
                    cond_primary.to(dtype=current_confidence.dtype),
                    cond_confidence.to(dtype=current_confidence.dtype),
                    cond_visible.to(dtype=current_confidence.dtype),
                    cond_velocity.to(dtype=current_confidence.dtype),
                )
            )

            update_mask = current_visible.expand(-1, 2)
            last_velocity = torch.where(update_mask, new_velocity, last_velocity)
            last_primary = torch.where(update_mask, current_primary, last_primary)
            has_last = has_last | current_visible
            lost_count = next_lost_count

        state = {
            "last_good_primary": last_primary,
            "last_good_velocity": last_velocity,
            "has_last_good": has_last,
            "lost_count": lost_count,
        }
        return torch.stack(cond_parts, dim=1), state

    def _predicted_tracking_visible(self, hist_prob_t: torch.Tensor, confidence: torch.Tensor) -> torch.Tensor:
        n = max(int(hist_prob_t.size(-1)), 1)
        topk = torch.topk(hist_prob_t, k=min(2, n), dim=-1).values
        top1 = topk[..., :1]
        top2 = topk[..., 1:2] if topk.size(-1) > 1 else torch.zeros_like(top1)
        margin = top1 - top2
        entropy = -(hist_prob_t.clamp_min(1.0e-8) * hist_prob_t.clamp_min(1.0e-8).log()).sum(dim=-1, keepdim=True)
        entropy = entropy / max(math.log(float(n)), 1.0e-6)
        uniform_conf = 1.0 / float(n)
        conf_min = max(
            float(getattr(self.cfg, "target_similarity_reacquire_confidence_min", 0.02)),
            uniform_conf * float(getattr(self.cfg, "target_similarity_reacquire_confidence_ratio", 2.5)),
        )
        entropy_max = float(getattr(self.cfg, "target_similarity_reacquire_entropy_max", 0.98))
        margin_min = float(getattr(self.cfg, "target_similarity_reacquire_margin_min", 0.0))
        return (
            (confidence >= conf_min)
            & (entropy <= entropy_max)
            & (margin >= margin_min)
        ).squeeze(-1)

    @staticmethod
    def _soft_center(prob: torch.Tensor, grid_hw: Tuple[int, int]) -> torch.Tensor:
        gh, gw = grid_hw
        xs = torch.linspace(0.0, 1.0, gw, device=prob.device, dtype=prob.dtype)
        ys = torch.linspace(0.0, 1.0, gh, device=prob.device, dtype=prob.dtype)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        gx = grid_x.reshape(-1)
        gy = grid_y.reshape(-1)
        cx = (prob * gx).sum(dim=-1)
        cy = (prob * gy).sum(dim=-1)
        return torch.stack([cx, cy], dim=-1)

    def _identity_from_first_visible(
        self,
        features: torch.Tensor,
        gt_prob: torch.Tensor,
        visible: torch.Tensor,
        memory_state: Optional[Dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        b, t, _, d = features.shape
        if memory_state is not None:
            prev_identity = memory_state.get("identity_token")
            if torch.is_tensor(prev_identity) and prev_identity.shape == (b, d):
                return prev_identity.to(device=features.device, dtype=features.dtype)

        any_visible = visible.any(dim=1)
        first_idx = visible.float().argmax(dim=1)
        batch_idx = torch.arange(b, device=features.device)
        ref_features = features[batch_idx, first_idx]
        ref_prob = gt_prob[batch_idx, first_idx]
        identity = (ref_prob.unsqueeze(-1) * ref_features).sum(dim=1)
        fallback = features[:, 0].mean(dim=1)
        identity = torch.where(any_visible.unsqueeze(-1), identity, fallback)
        return F.normalize(identity, dim=-1)

    def _identity_from_init_heatmap(
        self,
        features: torch.Tensor,
        init_heatmap: torch.Tensor,
    ) -> torch.Tensor:
        b, _, n, _ = features.shape
        if init_heatmap.ndim == 4 and init_heatmap.size(1) == 1:
            init_heatmap = init_heatmap[:, 0]
        if init_heatmap.ndim == 3:
            init_prob = init_heatmap.reshape(b, -1).to(device=features.device, dtype=features.dtype)
        elif init_heatmap.ndim == 2:
            init_prob = init_heatmap.to(device=features.device, dtype=features.dtype)
        else:
            raise ValueError("init_heatmap must have shape [B,H,W], [B,1,H,W], or [B,N].")
        if init_prob.size(-1) != n:
            raise ValueError(f"init_heatmap has {init_prob.size(-1)} cells but patch grid has {n}.")
        init_prob = init_prob / init_prob.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)
        identity = (init_prob.unsqueeze(-1) * features[:, 0]).sum(dim=1)
        return F.normalize(identity, dim=-1)

    def _decayed_history(
        self,
        all_values: torch.Tensor,
        prev_len: int,
        step: int,
        history_size: int,
    ) -> torch.Tensor:
        end = prev_len + step + 1
        start = max(0, end - history_size)
        window = all_values[:, start:end]
        length = int(window.size(1))
        decay = float(getattr(self.cfg, "target_similarity_decay", 0.8))
        decay = min(max(decay, 0.0), 1.0)
        weights = torch.pow(
            torch.full((length,), decay, device=window.device, dtype=window.dtype),
            torch.arange(length - 1, -1, -1, device=window.device, dtype=window.dtype),
        )
        weights = weights / weights.sum().clamp_min(1.0e-8)
        return (window * weights.view(1, length, *([1] * (window.ndim - 2)))).sum(dim=1)

    def _grid_pool_similarity_tokens(
        self,
        features: torch.Tensor,
        hist_prob: torch.Tensor,
        pred_center: torch.Tensor,
        confidence: torch.Tensor,
        grid_hw: Tuple[int, int],
    ) -> torch.Tensor:
        b, t, n, d = features.shape
        gh, gw = grid_hw
        pool_size = max(int(getattr(self.cfg, "target_similarity_grid_pool_size", 4)), 1)
        feat_map = features.reshape(b * t, gh, gw, d).permute(0, 3, 1, 2)
        prob_map = hist_prob.reshape(b * t, 1, gh, gw)
        weighted_feat = F.adaptive_avg_pool2d(feat_map * prob_map, (pool_size, pool_size))
        pooled_prob = F.adaptive_avg_pool2d(prob_map, (pool_size, pool_size))
        pooled_feat = weighted_feat / pooled_prob.clamp_min(1.0e-6)
        pooled_feat = pooled_feat.permute(0, 2, 3, 1).reshape(b, t, pool_size * pool_size, d)
        pooled_prob_flat = pooled_prob.reshape(b, t, pool_size * pool_size, 1)
        center = pred_center.unsqueeze(2).expand(-1, -1, pool_size * pool_size, -1)
        conf = confidence.unsqueeze(2).expand(-1, -1, pool_size * pool_size, -1)
        token_in = torch.cat([pooled_feat, pooled_prob_flat, center, conf], dim=-1)
        token_dtype = next(self.similarity_token_proj.parameters()).dtype
        return self.similarity_token_proj(token_in.to(dtype=token_dtype)).reshape(b, t * pool_size * pool_size, -1)

    def forward(
        self,
        images: torch.Tensor,
        target_relative: torch.Tensor,
        video_latents: Optional[torch.Tensor] = None,
        memory_state: Optional[Dict[str, torch.Tensor]] = None,
        init_heatmap: Optional[torch.Tensor] = None,
        return_state: bool = False,
        use_gt_visible_for_condition: bool = True,
    ) -> Dict[str, torch.Tensor]:
        if target_relative.ndim != 3:
            raise ValueError("target_relative must have shape [B, T, D].")
        features, grid_hw = self._encode_features(images, video_latents)
        b, t, n, d = features.shape
        if target_relative.size(0) != b:
            raise ValueError("target_relative batch size must match target similarity features.")
        target_relative = self._align_target_relative(target_relative, t)
        # DeepSpeed/bf16 can leave encoded features in bf16 while GT heatmaps are
        # float32. Keep the similarity/probability path in fp32 so einsum,
        # softmax, and auxiliary losses all see consistent, stable dtypes.
        metric_features = features.float()

        gh, gw = grid_hw
        target_relative_for_heatmap = target_relative.to(device=features.device, dtype=torch.float32)
        camera_offset_body = self._camera_offset_body()
        gt_heatmap = make_attention_heatmap(
            target_relative_for_heatmap,
            (gh, gw),
            fov_deg=float(getattr(self.cfg, "target_similarity_fov_deg", 90.0)),
            sigma=float(getattr(self.cfg, "target_similarity_heatmap_sigma", 0.05)),
            camera_offset_body=camera_offset_body,
        ).squeeze(-3)
        gt_prob = self._normalize_heatmap(gt_heatmap)
        gt_sum = gt_heatmap.reshape(b, t, -1).sum(dim=-1)
        _, gt_visible = project_body_to_image(
            target_relative_for_heatmap,
            (gh, gw),
            fov_deg=float(getattr(self.cfg, "target_similarity_fov_deg", 90.0)),
            camera_offset_body=camera_offset_body,
        )
        visible = (gt_sum > 1.0e-8) & gt_visible.to(device=features.device, dtype=torch.bool)

        if init_heatmap is not None:
            identity = self._identity_from_init_heatmap(metric_features, init_heatmap)
        else:
            identity = self._identity_from_first_visible(metric_features, gt_prob, visible, memory_state)
        identity_n = F.normalize(identity.float(), dim=-1)
        temperature = float(getattr(self.cfg, "target_similarity_temperature", 10.0))
        logits = torch.einsum("btnd,bd->btn", metric_features, identity_n) * temperature
        sim_prob = torch.softmax(logits, dim=-1)
        target_features = F.normalize((gt_prob.unsqueeze(-1) * metric_features).sum(dim=2), dim=-1)
        pred_features = F.normalize((sim_prob.unsqueeze(-1) * metric_features).sum(dim=2), dim=-1)

        history_size = max(int(getattr(self.cfg, "target_similarity_history_size", 4)), 1)
        prev_prob = None
        prev_feat = None
        prev_len = 0
        if memory_state is not None:
            prev_prob = memory_state.get("sim_prob")
            prev_feat = memory_state.get("pred_features")
            if torch.is_tensor(prev_prob) and torch.is_tensor(prev_feat):
                prev_prob = prev_prob.to(device=sim_prob.device, dtype=sim_prob.dtype)
                prev_feat = prev_feat.to(device=pred_features.device, dtype=pred_features.dtype)
                if prev_prob.ndim == 3 and prev_prob.size(0) == b and prev_prob.size(-1) == n:
                    prev_len = int(prev_prob.size(1))
                else:
                    prev_prob = None
                    prev_feat = None
        all_prob = sim_prob if prev_prob is None else torch.cat([prev_prob, sim_prob], dim=1)
        all_feat = pred_features if prev_feat is None else torch.cat([prev_feat, pred_features], dim=1)

        hist_probs = []
        hist_feats = []
        for i in range(t):
            hist_prob = self._decayed_history(all_prob, prev_len, i, history_size)
            hist_prob = hist_prob / hist_prob.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)
            hist_feat = self._decayed_history(all_feat, prev_len, i, history_size)
            hist_probs.append(hist_prob)
            hist_feats.append(hist_feat)
        hist_prob_t = torch.stack(hist_probs, dim=1)
        hist_feat_t = F.normalize(torch.stack(hist_feats, dim=1), dim=-1)
        pred_center = self._soft_center(hist_prob_t, grid_hw)
        gt_center, _ = project_body_to_image(
            target_relative_for_heatmap,
            (gh, gw),
            fov_deg=float(getattr(self.cfg, "target_similarity_fov_deg", 90.0)),
            camera_offset_body=camera_offset_body,
        )
        confidence = hist_prob_t.amax(dim=-1, keepdim=True)
        prev_center = None
        if memory_state is not None:
            prev_center = memory_state.get("pred_center")
            if torch.is_tensor(prev_center):
                prev_center = prev_center.to(device=pred_center.device, dtype=pred_center.dtype)
                if prev_center.ndim != 3 or prev_center.size(0) != b or prev_center.size(-1) != 2:
                    prev_center = None
        if prev_center is not None and prev_center.size(1) > 0:
            all_center = torch.cat([prev_center, pred_center], dim=1)
            start = prev_center.size(1) - 1
            prev_for_delta = all_center[:, start : start + t]
        else:
            prev_for_delta = torch.cat([pred_center[:, :1], pred_center[:, :-1]], dim=1)
        center_delta = pred_center - prev_for_delta
        center_velocity = torch.linalg.norm(center_delta.float(), dim=-1, keepdim=True).to(pred_center.dtype)
        pred_visible = (confidence > (1.0 / max(float(n), 1.0))).to(dtype=pred_center.dtype)
        tracking_condition_state: Dict[str, torch.Tensor] = {}
        condition_source = str(getattr(self.cfg, "target_similarity_condition_source", "predicted")).strip().lower()
        gt_primary = None
        gt_condition = None
        if condition_source in {"gt", "ground_truth", "oracle", "mixed", "gt_mix", "mix"}:
            gt_primary = self._target_condition_primary_from_gt(
                target_relative_for_heatmap,
                gt_center.to(device=features.device, dtype=torch.float32),
                grid_hw,
                camera_offset_body,
            ).to(device=features.device, dtype=pred_center.dtype)
            gt_visible_f = visible.to(device=features.device, dtype=pred_center.dtype).unsqueeze(-1)
            prev_gt_primary = torch.cat([gt_primary[:, :1], gt_primary[:, :-1]], dim=1)
            gt_primary_velocity = torch.linalg.norm((gt_primary - prev_gt_primary).float(), dim=-1, keepdim=True).to(pred_center.dtype)
            gt_condition = self._target_condition_from_primary(
                gt_primary,
                gt_visible_f,
                gt_visible_f,
                gt_primary_velocity,
            )

        if self._uses_last_good_condition():
            condition_visible = (
                visible
                if use_gt_visible_for_condition and condition_source not in {"mixed", "gt_mix", "mix"}
                else self._predicted_tracking_visible(hist_prob_t, confidence)
            )
            pred_primary = self._target_condition_primary_from_center(pred_center, grid_hw)
            predicted_condition, tracking_condition_state = self._target_condition_with_last_good(
                pred_primary,
                confidence,
                condition_visible,
                grid_hw,
                memory_state,
            )
        else:
            predicted_condition = self._target_condition_from_center(
                pred_center,
                confidence,
                pred_visible,
                center_velocity,
                grid_hw,
            )

        if condition_source in {"gt", "ground_truth", "oracle"} and gt_condition is not None:
            target_condition = gt_condition
        elif condition_source in {"mixed", "gt_mix", "mix"} and gt_condition is not None and self.training:
            mix_prob = float(getattr(self.cfg, "target_similarity_condition_gt_mix_prob", 0.25))
            mix_prob = min(max(mix_prob, 0.0), 1.0)
            if mix_prob <= 0.0:
                target_condition = predicted_condition
            elif mix_prob >= 1.0:
                target_condition = gt_condition
            else:
                mix_mask = torch.rand(
                    predicted_condition.shape[:2] + (1,),
                    device=predicted_condition.device,
                ) < mix_prob
                target_condition = torch.where(mix_mask, gt_condition, predicted_condition)
        else:
            target_condition = predicted_condition

        token_dtype = next(self.target_token_proj.parameters()).dtype
        context_mode = self._context_mode()
        if context_mode == "dense":
            target_token = self.target_token_proj(identity.to(dtype=token_dtype)).unsqueeze(1)
            history_in = torch.cat([hist_feat_t, pred_center.to(hist_feat_t.dtype), confidence.to(hist_feat_t.dtype)], dim=-1)
            history_tokens = self.history_token_proj(history_in.to(dtype=token_dtype))
            similarity_tokens = self._grid_pool_similarity_tokens(
                metric_features,
                hist_prob_t,
                pred_center.to(hist_prob_t.dtype),
                confidence,
                grid_hw,
            )
            context_tokens = torch.cat([target_token, history_tokens, similarity_tokens], dim=1)
        elif context_mode == "camera_yz_text":
            context_tokens = self.condition_token_proj(target_condition.to(dtype=token_dtype))
        else:
            raise ValueError(
                f"Unsupported target_similarity_context_mode={context_mode!r}; "
                "expected 'dense' or 'camera_yz_text'."
            )
        context_tokens = context_tokens * float(getattr(self.cfg, "target_similarity_token_scale", 1.0))
        context_mask = torch.ones(context_tokens.shape[:2], device=context_tokens.device, dtype=torch.bool)

        out: Dict[str, torch.Tensor] = {
            "context_tokens": context_tokens,
            "context_mask": context_mask,
            "target_condition": target_condition,
            "similarity_heatmap": hist_prob_t.reshape(b, t, gh, gw),
            "raw_similarity_heatmap": sim_prob.reshape(b, t, gh, gw),
            "pred_center": pred_center,
            "identity_token": identity,
            "target_features": target_features,
            "gt_heatmap": gt_heatmap,
            "gt_center": gt_center.to(device=features.device, dtype=torch.float32),
            "visible": visible,
        }
        if return_state:
            out["memory_sim_prob"] = all_prob[:, -history_size:].detach()
            out["memory_pred_features"] = all_feat[:, -history_size:].detach()
            out["memory_pred_center"] = pred_center[:, -history_size:].detach()
            out["memory_identity_token"] = identity.detach()
            for key, value in tracking_condition_state.items():
                out[f"memory_{key}"] = value.detach()
        return out
