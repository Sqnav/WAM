from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

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


S0_PARAMETER_PREFIXES = (
    "tracker.",
    "fastwam.tracker_fusion.",
    "fastwam.current_target_localizer.",
)


def normalize_s0_checkpoint_state(payload: Any) -> Dict[str, torch.Tensor]:
    """Normalize either a V5 S0 companion or a standalone UAVTracker checkpoint."""
    raw_state = (
        payload.get("model", payload.get("state_dict", payload))
        if isinstance(payload, dict)
        else payload
    )
    if not isinstance(raw_state, dict) or not raw_state:
        raise ValueError("S0 checkpoint contains no model parameters.")
    state = {
        (key.removeprefix("module.")): value
        for key, value in raw_state.items()
    }
    if all(key.startswith(S0_PARAMETER_PREFIXES) for key in state):
        return state

    tracker_markers = {"template_pos", "search_pos", "segment_embed"}
    is_standalone_tracker = (
        isinstance(payload, dict)
        and "args" in payload
        and tracker_markers.issubset(state)
    )
    if is_standalone_tracker:
        return {f"tracker.{key}": value for key, value in state.items()}

    invalid = sorted(key for key in state if not key.startswith(S0_PARAMETER_PREFIXES))
    raise ValueError(f"S0 checkpoint contains non-S0 parameters: {invalid[:8]}")


class _S0FastWAMModules(nn.Module):
    """Namespace matching the S0 module names inside TeacherWorldModelDiT."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.tracker_fusion = None
        self.current_target_localizer = None
        if bool(cfg.tracker_include_box_token):
            return
        from .current_target_localizer import CurrentTargetLocalizer
        from .tracker_fusion import LocalFeatureTrackerConditionFusion

        hidden_dim = 1024
        num_heads = 8
        self.tracker_fusion = LocalFeatureTrackerConditionFusion(
            tracker_dim=int(cfg.tracker_feature_dim),
            action_dim=hidden_dim,
            num_heads=num_heads,
            head_dim=max(hidden_dim // max(num_heads, 1), 1),
            num_layers=0,
            start_layer=0,
            grid_size=int(cfg.tracker_feature_grid_size),
            use_local_position_embedding=bool(cfg.tracker_use_local_position_embedding),
            include_box_token=False,
            detach_tracker_inputs=False,
            enable_cross_attention=False,
        )
        self.current_target_localizer = CurrentTargetLocalizer(
            tracker_dim=int(cfg.tracker_feature_dim),
            hidden_dim=hidden_dim,
            num_heads=num_heads,
        )


class S0LocalizationModel(nn.Module):
    """Standalone current-state model using either the Tracker head or Target Query."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        from pathlib import Path
        from tracking.model import UAVTracker

        self.cfg = cfg
        if cfg.tracker_mot_integration != "mot_tracker_finetune_local_feature":
            raise ValueError("S0LocalizationModel requires the trainable local-feature Tracker.")
        init_mode = str(cfg.tracker_finetune_init).strip().lower()
        use_tracker_head = bool(cfg.tracker_include_box_token)
        if init_mode == "uav_tracker":
            checkpoint_path = Path(str(cfg.tracker_finetune_checkpoint).strip())
            if not checkpoint_path.is_file():
                raise FileNotFoundError(
                    "S0 Tracker-head initialization requires tracker_finetune_checkpoint."
                )
            payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            state = (
                payload.get("model", payload.get("state_dict", payload))
                if isinstance(payload, dict)
                else payload
            )
            checkpoint_args = payload.get("args", {}) if isinstance(payload, dict) else {}
            self.tracker = UAVTracker(
                backbone=str(checkpoint_args.get("backbone", "deit_tiny_patch16_224")),
                pretrained=False,
                template_size=int(cfg.tracker_template_size),
                search_size=int(cfg.tracker_search_size),
                square_boxes=True,
                enable_head=use_tracker_head,
            )
            missing, unexpected = self.tracker.load_state_dict(state, strict=False)
            if missing or unexpected:
                raise ValueError(
                    "S0 pretrained Tracker is incompatible: "
                    f"missing={list(missing)[:8]} unexpected={list(unexpected)[:8]}"
                )
        elif init_mode == "imagenet_deit":
            pretrained_path = Path(str(cfg.tracker_backbone_pretrained_path).strip())
            if not pretrained_path.is_file():
                raise FileNotFoundError(
                    "S0LocalizationModel requires a local tracker_backbone_pretrained_path."
                )
            self.tracker = UAVTracker(
                pretrained=True,
                pretrained_path=pretrained_path,
                template_size=int(cfg.tracker_template_size),
                search_size=int(cfg.tracker_search_size),
                square_boxes=True,
                enable_head=use_tracker_head,
            )
        else:
            raise ValueError(
                "S0LocalizationModel requires tracker_finetune_init to be "
                "'uav_tracker' or 'imagenet_deit'."
            )
        self.fastwam = _S0FastWAMModules(cfg)

    def forward(
        self,
        tracker_template: torch.Tensor,
        tracker_search: torch.Tensor,
        tracker_search_geometry: torch.Tensor,
        tracker_image_size: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        tracker_param = next(self.tracker.parameters())
        use_tracker_head = bool(self.cfg.tracker_include_box_token)
        tracker_out = self.tracker(
            tracker_template.to(device=tracker_param.device, dtype=tracker_param.dtype),
            tracker_search.to(device=tracker_param.device, dtype=tracker_param.dtype),
            return_head=use_tracker_head,
        )
        if use_tracker_head:
            crop_box, confidence = self.tracker.decode_peak(tracker_out)
            current_box = self.tracker.map_crop_boxes_to_image(
                crop_box,
                tracker_search_geometry.to(device=crop_box.device),
                tracker_image_size.to(device=crop_box.device),
            )
            logits = tracker_out["center_logits"]
            attention = torch.softmax(logits.float().flatten(2), dim=-1)
            full_xy = self.tracker.full_image_grid_coordinates(
                tracker_search_geometry.to(device=logits.device),
                tracker_image_size.to(device=logits.device),
                logits.size(-2),
                logits.size(-1),
            )
            return {
                "current_box": current_box,
                "current_attention": attention,
                "full_xy": full_xy,
                "tracker_confidence": confidence,
            }

        template_features = tracker_out["template_features"]
        search_features = tracker_out["search_features"].flatten(2).transpose(1, 2)
        if self.fastwam.tracker_fusion is None or self.fastwam.current_target_localizer is None:
            raise RuntimeError("Target Query S0 modules are unavailable.")
        condition = self.fastwam.tracker_fusion.make_condition(
            tracker_features=search_features,
            tracker_bbox=None,
            tracker_search_geometry=tracker_search_geometry,
            tracker_image_size=tracker_image_size,
        )
        full_xy = self.fastwam.tracker_fusion.full_image_coordinates(
            tracker_search_geometry.to(device=condition.device),
            tracker_image_size.to(device=condition.device),
        ).to(dtype=condition.dtype)
        return self.fastwam.current_target_localizer(
            template_features.to(device=condition.device, dtype=condition.dtype),
            condition,
            full_xy,
        )


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
        tracker_context_hidden = max(
            int(getattr(cfg, "tracker_center_context_hidden_dim", 256)), 2
        )
        self.tracker_center_context_proj = nn.Sequential(
            nn.Linear(2, tracker_context_hidden),
            nn.SiLU(),
            nn.Dropout(float(getattr(cfg, "dropout", 0.0))),
            nn.Linear(tracker_context_hidden, cfg.text_width),
            nn.LayerNorm(cfg.text_width),
        )
        self.tracker_center_missing_token = nn.Parameter(torch.zeros(1, 1, cfg.text_width))
        nn.init.normal_(self.tracker_center_missing_token, std=0.02)
        if not bool(getattr(cfg, "use_tracker_center_context", False)):
            for p in self.tracker_center_context_proj.parameters():
                p.requires_grad_(False)
            self.tracker_center_missing_token.requires_grad_(False)
        self.fusion = CrossAttentionFusion(cfg)
        self.rssm = RSSM(cfg) if cfg.use_rssm else None
        self.prediction_heads = TeacherPredictionHeads(cfg)
        self.fastwam = FastWAMHead(cfg) if bool(cfg.use_fastwam_mot) else None
        if self.fastwam is None:
            raise RuntimeError("Legacy MLP/DiT actors were removed; set cfg.use_fastwam_mot=True.")
        self.tracker = None
        if cfg.tracker_mot_integration == "mot_tracker_finetune_local_feature":
            from pathlib import Path
            from tracking.model import UAVTracker

            init_mode = str(getattr(cfg, "tracker_finetune_init", "uav_tracker")).strip().lower()
            if init_mode == "uav_tracker":
                checkpoint = Path(str(cfg.tracker_finetune_checkpoint))
                if not checkpoint.is_file():
                    raise FileNotFoundError(
                        "mot_tracker_finetune_local_feature with tracker_finetune_init=uav_tracker "
                        "requires tracker_finetune_checkpoint."
                    )
                payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
                checkpoint_args = payload.get("args", {}) if isinstance(payload, dict) else {}
                include_tracker_head = bool(
                    getattr(cfg, "tracker_include_box_token", True)
                )
                self.tracker = UAVTracker(
                    backbone=str(
                        checkpoint_args.get("backbone", "deit_tiny_patch16_224")
                    ),
                    pretrained=False,
                    template_size=int(cfg.tracker_template_size),
                    search_size=int(cfg.tracker_search_size),
                    square_boxes=True,
                    enable_head=include_tracker_head,
                )
                state = payload.get("model", payload.get("state_dict", payload)) if isinstance(payload, dict) else payload
                missing, unexpected = self.tracker.load_state_dict(state, strict=False)
                if missing or unexpected:
                    message = (
                        "Tracker checkpoint is incompatible: "
                        f"missing={list(missing)[:8]} unexpected={list(unexpected)[:8]}"
                    )
                    if include_tracker_head:
                        raise ValueError(message)
                    print(f"[tracker-finetune] {message}")
            elif init_mode == "imagenet_deit":
                pretrained_path = Path(str(getattr(cfg, "tracker_backbone_pretrained_path", "")).strip())
                if not pretrained_path.is_file():
                    raise FileNotFoundError(
                        "tracker_finetune_init=imagenet_deit requires a local "
                        "tracker_backbone_pretrained_path."
                    )
                self.tracker = UAVTracker(
                    pretrained=True,
                    pretrained_path=pretrained_path,
                    template_size=int(cfg.tracker_template_size),
                    search_size=int(cfg.tracker_search_size),
                    square_boxes=True,
                    enable_head=bool(getattr(cfg, "tracker_include_box_token", True)),
                )
                print(f"[tracker-finetune] initialized DeiT backbone from {pretrained_path}")
            else:
                raise ValueError(
                    "tracker_finetune_init must be 'uav_tracker' or 'imagenet_deit', "
                    f"got {init_mode!r}."
                )

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

    def _make_tracker_center_context_tokens(
        self,
        tracker_center: Optional[torch.Tensor],
        tracker_confidence: Optional[torch.Tensor],
        target_device: torch.device,
        target_dtype: torch.dtype,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if not bool(getattr(self.cfg, "use_tracker_center_context", False)):
            return None, None
        if tracker_center is None:
            raise RuntimeError("Tracker center text context requires tracker_center.")
        if tracker_center.ndim != 3 or tracker_center.size(-1) != 2:
            raise ValueError("tracker_center must have shape [B, T, 2].")
        center = tracker_center[:, 0].float().clamp(0.0, 1.0).mul(2.0).sub(1.0)
        param = next(self.tracker_center_context_proj.parameters())
        projected = self.tracker_center_context_proj(
            center.to(device=param.device, dtype=param.dtype)
        ).unsqueeze(1)
        if tracker_confidence is None:
            confidence = torch.ones(
                projected.size(0), 1, 1, device=projected.device, dtype=projected.dtype
            )
        else:
            if tracker_confidence.ndim not in {2, 3}:
                raise ValueError("tracker_confidence must have shape [B, T] or [B, T, 1].")
            confidence = tracker_confidence.reshape(tracker_confidence.size(0), -1)[:, 0]
            confidence = confidence.to(device=projected.device, dtype=projected.dtype)
            confidence = confidence.clamp(0.0, 1.0).view(-1, 1, 1)
        missing = self.tracker_center_missing_token.to(
            device=projected.device, dtype=projected.dtype
        ).expand(projected.size(0), -1, -1)
        tokens = confidence * projected + (1.0 - confidence) * missing
        tokens = tokens * float(getattr(self.cfg, "tracker_center_token_scale", 1.0))
        tokens = tokens.to(device=target_device, dtype=target_dtype)
        mask = torch.ones(tokens.shape[:2], device=target_device, dtype=torch.bool)
        return tokens, mask

    def _first_tracker_features(
        self,
        tracker_features: Optional[torch.Tensor],
        tracker_confidence: Optional[torch.Tensor],
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        mode = str(getattr(self.cfg, "tracker_mot_integration", "none"))
        condition_mode = str(getattr(self.cfg, "tracker_condition_mode", "center_features"))
        if mode in {"none", "mot_tracker_finetune_local_feature"} or (
            mode == "frozen_deit_tracker_fusion" and "features" not in condition_mode
        ):
            return None, None
        if tracker_features is None:
            raise RuntimeError(f"tracker_mot_integration={mode!r} requires Tracker feature tokens.")
        if tracker_features.ndim == 4:
            features = tracker_features[:, 0]
        elif tracker_features.ndim == 3:
            features = tracker_features
        else:
            raise ValueError("tracker_features must have shape [B,T,N,C] or [B,N,C].")
        expected_tokens = max(int(getattr(self.cfg, "tracker_feature_grid_size", 7)), 1) ** 2
        expected_dim = int(getattr(self.cfg, "tracker_feature_dim", 192))
        if tuple(features.shape[1:]) != (expected_tokens, expected_dim):
            raise ValueError(
                "Tracker feature shape mismatch: expected "
                f"[B,{expected_tokens},{expected_dim}], got {tuple(features.shape)}."
            )
        if tracker_confidence is None:
            confidence = torch.ones(features.size(0), device=features.device, dtype=torch.float32)
        else:
            confidence = tracker_confidence.reshape(tracker_confidence.size(0), -1)[:, 0].float()
            confidence = torch.nan_to_num(confidence, nan=0.0).clamp(0.0, 1.0)
        return features, confidence

    @staticmethod
    def _first_tracker_tensor(
        value: Optional[torch.Tensor],
        *,
        unbatched_ndim: int,
        name: str,
    ) -> Optional[torch.Tensor]:
        if value is None:
            return None
        if value.ndim == unbatched_ndim + 2:
            return value[:, 0]
        if value.ndim == unbatched_ndim + 1:
            return value
        raise ValueError(
            f"{name} must have shape [B,T,...] or [B,...]; got {tuple(value.shape)}."
        )

    def encode_sequence(
        self,
        images: torch.Tensor,
        text_tokens: torch.Tensor,
        target_relative: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        instructions: Optional[list[str]] = None,
        video_latents: Optional[torch.Tensor] = None,
        guidance_heatmap: Optional[torch.Tensor] = None,
        guidance_confidence: Optional[torch.Tensor] = None,
        tracker_center: Optional[torch.Tensor] = None,
        tracker_features: Optional[torch.Tensor] = None,
        tracker_bbox: Optional[torch.Tensor] = None,
        tracker_response: Optional[torch.Tensor] = None,
        tracker_search_geometry: Optional[torch.Tensor] = None,
        tracker_image_size: Optional[torch.Tensor] = None,
        tracker_template: Optional[torch.Tensor] = None,
        tracker_search: Optional[torch.Tensor] = None,
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
        tracker_context, tracker_context_mask = self._make_tracker_center_context_tokens(
            tracker_center,
            guidance_confidence,
            target_device=fastwam_context.device,
            target_dtype=fastwam_context.dtype,
        )
        if tracker_context is not None and tracker_context_mask is not None:
            fastwam_context = torch.cat([fastwam_context, tracker_context], dim=1)
            fastwam_context_mask = torch.cat([fastwam_context_mask, tracker_context_mask], dim=1)
        action_fastwam_context = fastwam_context
        action_fastwam_context_mask = fastwam_context_mask
        raw_tracker_features, raw_tracker_confidence = self._first_tracker_features(
            tracker_features, guidance_confidence
        )
        raw_tracker_template_features = None
        raw_tracker_bbox = self._first_tracker_tensor(
            tracker_bbox, unbatched_ndim=1, name="tracker_bbox"
        )
        raw_tracker_response = self._first_tracker_tensor(
            tracker_response, unbatched_ndim=2, name="tracker_response"
        )
        raw_tracker_search_geometry = self._first_tracker_tensor(
            tracker_search_geometry,
            unbatched_ndim=1,
            name="tracker_search_geometry",
        )
        raw_tracker_image_size = self._first_tracker_tensor(
            tracker_image_size, unbatched_ndim=1, name="tracker_image_size"
        )
        if self.tracker is not None:
            if tracker_template is None or tracker_search is None:
                raise RuntimeError("Tracker fine-tuning requires tracker_template and tracker_search.")
            if raw_tracker_search_geometry is None or raw_tracker_image_size is None:
                raise RuntimeError("Tracker fine-tuning requires current search geometry and image size.")
            tracker_param = next(self.tracker.parameters())
            tracker_template = tracker_template.to(
                device=tracker_param.device, dtype=tracker_param.dtype
            )
            tracker_search = tracker_search.to(
                device=tracker_param.device, dtype=tracker_param.dtype
            )
            include_box_token = bool(getattr(self.cfg, "tracker_include_box_token", True))
            tracker_out = self.tracker(
                tracker_template,
                tracker_search,
                return_head=include_box_token,
            )
            raw_tracker_template_features = tracker_out["template_features"]
            raw_tracker_features = tracker_out["search_features"].flatten(2).transpose(1, 2)
            if include_box_token:
                crop_box, raw_tracker_confidence = self.tracker.decode_peak(tracker_out)
                raw_tracker_bbox = self.tracker.map_crop_boxes_to_image(
                    crop_box,
                    raw_tracker_search_geometry,
                    raw_tracker_image_size,
                )
            else:
                raw_tracker_bbox = None

        encoded_out = {
            "obs_embed": obs_embed.view(batch_size, latent_seq_len, -1),
            "fused_tokens": fused_tokens.view(batch_size, latent_seq_len, fused_tokens.size(1), fused_tokens.size(2)),
            "video_latents": video_latents,
            "text_context": fastwam_context,
            "text_context_mask": fastwam_context_mask,
            "action_text_context": action_fastwam_context,
            "action_text_context_mask": action_fastwam_context_mask,
            "tracker_features": raw_tracker_features,
            "tracker_template_features": raw_tracker_template_features,
            "tracker_feature_confidence": raw_tracker_confidence,
            "tracker_bbox": raw_tracker_bbox,
            "tracker_response": raw_tracker_response,
            "tracker_search_geometry": raw_tracker_search_geometry,
            "tracker_image_size": raw_tracker_image_size,
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
        guidance_heatmap: Optional[torch.Tensor] = None,
        guidance_confidence: Optional[torch.Tensor] = None,
        tracker_center: Optional[torch.Tensor] = None,
        tracker_features: Optional[torch.Tensor] = None,
        tracker_bbox: Optional[torch.Tensor] = None,
        tracker_response: Optional[torch.Tensor] = None,
        tracker_search_geometry: Optional[torch.Tensor] = None,
        tracker_image_size: Optional[torch.Tensor] = None,
        tracker_template: Optional[torch.Tensor] = None,
        tracker_search: Optional[torch.Tensor] = None,
        target_centers: Optional[torch.Tensor] = None,
        target_boxes: Optional[torch.Tensor] = None,
        target_center_valid: Optional[torch.Tensor] = None,
        target_box_history: Optional[torch.Tensor] = None,
        target_box_history_valid: Optional[torch.Tensor] = None,
        capture_fastwam_attention: bool = False,
        capture_fastwam_flow_predictions: bool = False,
        fastwam_noise_video_override: Optional[torch.Tensor] = None,
        fastwam_t_video_override: Optional[torch.Tensor] = None,
        fastwam_noise_action_override: Optional[torch.Tensor] = None,
        fastwam_t_action_override: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        encoded = self.encode_sequence(
            images,
            text_tokens,
            target_relative,
            attention_mask,
            instructions=instructions,
            video_latents=video_latents,
            guidance_confidence=guidance_confidence,
            tracker_center=tracker_center,
            tracker_features=tracker_features,
            tracker_bbox=tracker_bbox,
            tracker_response=tracker_response,
            tracker_search_geometry=tracker_search_geometry,
            tracker_image_size=tracker_image_size,
            tracker_template=tracker_template,
            tracker_search=tracker_search,
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
                capture_attention = bool(capture_fastwam_attention) or bool(
                    getattr(self.cfg, "return_gt_center_attention_for_distillation", False)
                )
                fastwam_out = self.fastwam.training_loss(
                    video_latents=encoded["video_latents"],
                    context=encoded["text_context"],
                    context_mask=encoded["text_context_mask"],
                    action_context=encoded["action_text_context"],
                    action_context_mask=encoded["action_text_context_mask"],
                    expert_action=expert_action.float(),
                    valid_mask=valid_mask,
                    target_relative=target_relative,
                    guidance_heatmap=guidance_heatmap,
                    guidance_confidence=guidance_confidence,
                    gt_center_xy=tracker_center,
                    tracker_features=encoded.get("tracker_features"),
                    tracker_template_features=encoded.get("tracker_template_features"),
                    tracker_confidence=encoded.get("tracker_feature_confidence"),
                    tracker_center=tracker_center,
                    tracker_bbox=encoded.get("tracker_bbox"),
                    tracker_response=encoded.get("tracker_response"),
                    tracker_search_geometry=encoded.get("tracker_search_geometry"),
                    tracker_image_size=encoded.get("tracker_image_size"),
                    target_box_history=target_box_history,
                    target_box_history_valid=target_box_history_valid,
                    previous_action=prev_actions[:, 0].float(),
                    target_centers=target_centers,
                    target_boxes=target_boxes,
                    target_center_valid=target_center_valid,
                    capture_attention=capture_attention,
                    return_flow_predictions=capture_fastwam_flow_predictions,
                    noise_video_override=fastwam_noise_video_override,
                    t_video_override=fastwam_t_video_override,
                    noise_action_override=fastwam_noise_action_override,
                    t_action_override=fastwam_t_action_override,
                )
                out["video_flow_loss"] = fastwam_out["loss_video"]
                out["policy_flow_loss"] = fastwam_out["loss_action"]
                out["policy_action_sequence"] = fastwam_out["pred_action_x0"]
                out["policy_action"] = fastwam_out["pred_action_x0"][..., 0, :]
                out["policy_flow_sequence"] = fastwam_out["pred_action_flow"]
                if fastwam_out.get("action_valid_mask") is not None:
                    out["policy_action_valid_mask"] = fastwam_out[
                        "action_valid_mask"
                    ]
                if capture_fastwam_flow_predictions:
                    out["video_velocity"] = fastwam_out["pred_video_velocity"]
                out["fastwam_attention_heatmap_loss"] = fastwam_out["loss_attention_heatmap"]
                out["fastwam_ortrack_consistency_loss"] = fastwam_out["loss_ortrack_consistency"]
                out["center_flow_loss"] = fastwam_out["loss_center_flow"]
                out["current_center_loss"] = fastwam_out["loss_current_center"]
                out["future_center_loss"] = fastwam_out["loss_future_center"]
                out["center_transition_loss"] = fastwam_out["loss_center_transition"]
                out["box_l1_loss"] = fastwam_out["loss_box_l1"]
                out["box_giou_loss"] = fastwam_out["loss_box_giou"]
                if fastwam_out.get("loss_state_flow") is not None:
                    out["state_flow_loss"] = fastwam_out["loss_state_flow"]
                out["current_box_loss"] = fastwam_out["loss_current_box"]
                out["current_center_spatial_loss"] = fastwam_out[
                    "loss_current_center_spatial"
                ]
                out["current_box_giou_loss"] = fastwam_out["loss_current_box_giou"]
                out["current_attention_loss"] = fastwam_out["loss_current_attention"]
                if fastwam_out.get("loss_history_future_center") is not None:
                    out["history_future_center_loss"] = fastwam_out[
                        "loss_history_future_center"
                    ]
                    out["history_future_center_error"] = fastwam_out[
                        "history_future_center_error"
                    ]
                if fastwam_out.get("loss_capture_value") is not None:
                    out["capture_value_loss"] = fastwam_out["loss_capture_value"]
                for key in (
                    "state_valid_ratio",
                    "current_box_valid_ratio",
                    "state_to_action_gate_mean",
                    "current_box_action_gate_mean",
                    "predicted_s0_box_error",
                    "predicted_s0_center_error_pixels",
                    "predicted_future_state_error",
                    "pred_state_flow",
                    "future_valid_ratio",
                    "future_h1_center_error",
                    "future_h1_iou",
                    "future_h4_center_error",
                    "future_h4_iou",
                    "future_h8_center_error",
                    "future_h8_iou",
                    "capture_value_capture_loss",
                    "capture_value_distance_loss",
                    "capture_value_visibility_loss",
                    "capture_value_ranking_loss",
                    "capture_value_ranking_accuracy",
                    "capture_value_target_capture",
                ):
                    if fastwam_out.get(key) is not None:
                        out[key] = fastwam_out[key]
                if fastwam_out.get("pred_center_flow") is not None:
                    out["pred_center_flow"] = fastwam_out["pred_center_flow"]
                    out["pred_state_centers"] = fastwam_out["pred_state_centers"]
                    out["pred_future_centers"] = fastwam_out["pred_future_centers"]
                    out["pred_state_boxes"] = fastwam_out["pred_state_boxes"]
                    out["pred_future_boxes"] = fastwam_out["pred_future_boxes"]
                elif fastwam_out.get("pred_state_boxes") is not None:
                    out["pred_state_centers"] = fastwam_out["pred_state_centers"]
                    out["pred_future_centers"] = fastwam_out["pred_future_centers"]
                    out["pred_state_boxes"] = fastwam_out["pred_state_boxes"]
                    out["pred_future_boxes"] = fastwam_out["pred_future_boxes"]
                if capture_attention:
                    out["last_action_attention"] = fastwam_out["last_action_attention"]
                    out["last_guided_action_attention"] = fastwam_out["last_guided_action_attention"]
                    for key in (
                        "last_raw_action_attention_logits",
                        "last_effective_action_attention_logits",
                        "center_gaussian",
                    ):
                        if fastwam_out.get(key) is not None:
                            out[key] = fastwam_out[key]
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
    def sample_distillation_targets(
        self,
        images: torch.Tensor,
        text_tokens: torch.Tensor,
        target_relative: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        instructions: Optional[list[str]] = None,
        video_latents: Optional[torch.Tensor] = None,
        guidance_heatmap: Optional[torch.Tensor] = None,
        guidance_confidence: Optional[torch.Tensor] = None,
        tracker_center: Optional[torch.Tensor] = None,
        tracker_features: Optional[torch.Tensor] = None,
        tracker_bbox: Optional[torch.Tensor] = None,
        tracker_response: Optional[torch.Tensor] = None,
        tracker_search_geometry: Optional[torch.Tensor] = None,
        tracker_image_size: Optional[torch.Tensor] = None,
        num_steps: Optional[int] = None,
        initial_action_noise: Optional[torch.Tensor] = None,
        return_attention_maps: bool = True,
        target_box_history: Optional[torch.Tensor] = None,
        target_box_history_valid: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Sample frozen-teacher policy targets for a Tracker-free student."""
        if self.fastwam is None:
            raise RuntimeError("FastWAM is required for policy distillation targets.")
        encoded = self.encode_sequence(
            images=images,
            text_tokens=text_tokens,
            target_relative=target_relative,
            attention_mask=attention_mask,
            instructions=instructions,
            video_latents=video_latents,
            guidance_confidence=guidance_confidence,
            tracker_center=tracker_center,
            tracker_features=tracker_features,
            tracker_bbox=tracker_bbox,
            tracker_response=tracker_response,
            tracker_search_geometry=tracker_search_geometry,
            tracker_image_size=tracker_image_size,
        )
        first_frame_latents = encoded["video_latents"][:, :, :1]
        action_horizon = max(int(self.cfg.action_sequence_horizon), 1)
        action_sampling_kwargs = {
            "first_frame_latents": first_frame_latents,
            "context": encoded["text_context"],
            "context_mask": encoded["text_context_mask"],
            "action_context": encoded["action_text_context"],
            "action_context_mask": encoded["action_text_context_mask"],
            "action_horizon": action_horizon,
            "num_steps": num_steps,
            "guidance_heatmap": guidance_heatmap,
            "guidance_confidence": guidance_confidence,
            "initial_action_noise": initial_action_noise,
            "gt_center_xy": None if tracker_center is None else tracker_center[:, 0],
            "tracker_features": encoded.get("tracker_features"),
            "tracker_template_features": encoded.get("tracker_template_features"),
            "tracker_bbox": encoded.get("tracker_bbox"),
            "tracker_response": encoded.get("tracker_response"),
            "tracker_confidence": encoded.get("tracker_feature_confidence"),
            "tracker_search_geometry": encoded.get("tracker_search_geometry"),
            "tracker_image_size": encoded.get("tracker_image_size"),
            "future_video_latent_frames": 3,
            "target_box_history": target_box_history,
            "target_box_history_valid": target_box_history_valid,
        }
        sample_out = self.fastwam.sample_action(
            **action_sampling_kwargs,
            return_attention_maps=return_attention_maps,
        )
        if isinstance(sample_out, tuple):
            action_sequence, attention_aux = sample_out
        else:
            action_sequence = sample_out
            attention_aux = None
        result: Dict[str, torch.Tensor] = {
            "teacher_action_sequence": action_sequence.detach(),
        }
        if attention_aux is not None:
            attention = attention_aux.get("last_transformer_attention")
            if attention is not None:
                result["teacher_action_attention"] = attention.detach()
            last_action_input = attention_aux.get("last_action_input")
            last_action_timestep = attention_aux.get("last_action_timestep")
            if last_action_input is not None and last_action_timestep is not None:
                result["teacher_last_action_input"] = last_action_input.detach()
                result["teacher_last_action_timestep"] = last_action_timestep.detach()
        return result

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
        guidance_heatmap: Optional[torch.Tensor] = None,
        guidance_confidence: Optional[torch.Tensor] = None,
        tracker_center: Optional[torch.Tensor] = None,
        tracker_features: Optional[torch.Tensor] = None,
        tracker_bbox: Optional[torch.Tensor] = None,
        tracker_response: Optional[torch.Tensor] = None,
        tracker_search_geometry: Optional[torch.Tensor] = None,
        tracker_image_size: Optional[torch.Tensor] = None,
        tracker_template: Optional[torch.Tensor] = None,
        tracker_search: Optional[torch.Tensor] = None,
        target_box_history: Optional[torch.Tensor] = None,
        target_box_history_valid: Optional[torch.Tensor] = None,
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
            guidance_confidence=guidance_confidence,
            tracker_center=(
                None if tracker_center is None else tracker_center.unsqueeze(1)
            ),
            tracker_features=tracker_features,
            tracker_bbox=tracker_bbox,
            tracker_response=tracker_response,
            tracker_search_geometry=tracker_search_geometry,
            tracker_image_size=tracker_image_size,
            tracker_template=tracker_template,
            tracker_search=tracker_search,
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
            action_horizon = max(int(self.cfg.action_sequence_horizon), 1)
            action_sampling_kwargs = {
                "first_frame_latents": encoded["video_latents"],
                "context": encoded["text_context"],
                "context_mask": encoded["text_context_mask"],
                "action_context": encoded["action_text_context"],
                "action_context_mask": encoded["action_text_context_mask"],
                "action_horizon": action_horizon,
                "num_steps": num_steps,
                "guidance_heatmap": guidance_heatmap,
                "guidance_confidence": guidance_confidence,
                "initial_action_noise": None,
                "gt_center_xy": tracker_center,
                "tracker_features": encoded.get("tracker_features"),
                "tracker_template_features": encoded.get("tracker_template_features"),
                "tracker_bbox": encoded.get("tracker_bbox"),
                "tracker_response": encoded.get("tracker_response"),
                "tracker_confidence": encoded.get("tracker_feature_confidence"),
                "tracker_search_geometry": encoded.get("tracker_search_geometry"),
                "tracker_image_size": encoded.get("tracker_image_size"),
                "future_video_latent_frames": predicted_video_latent_frames,
                "target_box_history": target_box_history,
                "target_box_history_valid": target_box_history_valid,
                "previous_action": prev_action,
            }
            sample_out = self.fastwam.sample_action(
                **action_sampling_kwargs,
                return_attention_maps=save_transformer_attention,
            )
            attention_aux = None
            if isinstance(sample_out, tuple):
                action_sequence_norm, attention_aux = sample_out
            else:
                action_sequence_norm = sample_out
            action_norm = action_sequence_norm[:, 0]
            predicted_video_latents = None
            joint_future_video = (
                None
                if attention_aux is None
                else attention_aux.get("future_video_latents")
            )
            if save_predicted_video and joint_future_video is not None:
                predicted_video_latents = joint_future_video
            elif save_predicted_video:
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
            for key in (
                "capture_value_candidates",
                "capture_value_scores",
                "capture_value_selected_index",
                "capture_value_raw_selected_index",
                "capture_value_score_advantage",
                "capture_value_used_fallback",
                "capture_value_recenter_costs",
                "capture_value_pursuit_costs",
                "capture_value_smooth_costs",
                "capture_value_consensus_costs",
                "capture_value_observed_center_velocity",
                "capture_value_selected_final_center_error",
                "capture_value_selected_final_box_size",
                "capture_value_selected_capture_probability",
                "capture_value_selected_final_distance",
                "capture_value_selected_visibility",
                "capture_action_prior_mean",
                "capture_action_prior_std",
            ):
                if attention_aux.get(key) is not None:
                    out[key] = attention_aux[key]
            future_target_centers = attention_aux.get("future_target_centers")
            if future_target_centers is not None:
                out["future_target_centers"] = future_target_centers
            target_state_centers = attention_aux.get("target_state_centers")
            if target_state_centers is not None:
                out["target_state_centers"] = target_state_centers
            future_target_boxes = attention_aux.get("future_target_boxes")
            if future_target_boxes is not None:
                out["future_target_boxes"] = future_target_boxes
            target_state_boxes = attention_aux.get("target_state_boxes")
            if target_state_boxes is not None:
                out["target_state_boxes"] = target_state_boxes
            pred_state_flow = attention_aux.get("pred_state_flow")
            if pred_state_flow is not None:
                out["pred_state_flow"] = pred_state_flow
            attn = attention_aux.get("last_transformer_attention")
            grid_size = attention_aux.get("video_grid_size")
            if attn is not None and grid_size is not None:
                _, grid_h, grid_w = grid_size
                first_frame_tokens = int(grid_h) * int(grid_w)
                spatial = attn[..., :first_frame_tokens].float().clamp_min(1.0e-8)
                spatial_normalized = spatial / spatial.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)
                per_query = spatial.sum(dim=1)
                per_query = per_query / per_query.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)
                out["last_transformer_attention_per_query_maps"] = per_query.reshape(
                    attn.size(0), attn.size(2), int(grid_h), int(grid_w)
                )
                out["last_transformer_attention_all_queries_map"] = spatial_normalized.mean(dim=(1, 2)).reshape(
                    attn.size(0), int(grid_h), int(grid_w)
                )
                query0 = spatial[:, :, 0].sum(dim=1)
                query0 = query0 / query0.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)
                out["last_transformer_attention_query0_map"] = query0.reshape(
                    attn.size(0), int(grid_h), int(grid_w)
                )
                out["last_transformer_attention_map"] = out["last_transformer_attention_query0_map"]
                out["last_transformer_attention_unnormalized_map"] = spatial.mean(dim=(1, 2)).reshape(
                    attn.size(0), int(grid_h), int(grid_w)
                )
                out["last_transformer_attention_raw"] = attn
                for source_key, output_key in (
                    ("last_transformer_raw_attention", "last_transformer_raw_attention"),
                    ("last_transformer_raw_attention_logits", "last_transformer_raw_attention_logits"),
                    (
                        "last_transformer_effective_attention_logits",
                        "last_transformer_effective_attention_logits",
                    ),
                ):
                    value = attention_aux.get(source_key)
                    if value is not None:
                        out[output_key] = value
            tracker_attention = attention_aux.get("last_tracker_cross_attention")
            if tracker_attention is not None:
                out["last_tracker_cross_attention"] = tracker_attention
        tracker_confidence_out = encoded.get("tracker_feature_confidence")
        if tracker_confidence_out is not None:
            out["tracker_confidence"] = tracker_confidence_out
        if candidate_info is not None:
            out.update(candidate_info)
        return out, post
