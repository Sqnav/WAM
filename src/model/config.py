from dataclasses import dataclass
from typing import Any, Dict, Tuple


@dataclass
class ModelConfig:
    # Image / DINOv2
    image_size: int = 224
    in_channels: int = 3
    image_encoder_dim: int = 768
    dinov2_model_name: str = "/data1/ysq/Worldmodel/model/dinov2-base"
    dinov2_freeze: bool = True
    dinov2_local_files_only: bool = True
    image_normalize: bool = True
    image_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    image_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)

    # Text / real pretrained CLIP text encoder
    clip_text_model_name: str = "/data1/ysq/Worldmodel/model/clip-vit-base-patch32"
    clip_text_freeze: bool = True
    clip_text_local_files_only: bool = True
    text_context_length: int = 77
    text_width: int = 512
    text_pad_id: int = 0

    # Low-dimensional inputs. State input has been removed.
    target_relative_dim: int = 3
    action_dim: int = 4
    target_token_hidden_dim: int = 128
    target_token_fusion_mode: str = "attention"  # attention | concat

    # Optional visual target guidance:
    # - global image remains unchanged
    # - target projection is used as a patch-level attention/pooling bias
    use_target_visual_guidance: bool = False
    use_attention_heatmap: bool = True
    visual_guidance_fov_deg: float = 90.0
    attention_heatmap_sigma: float = 0.08
    heatmap_attention_bias_strength: float = 2.0
    heatmap_out_of_view_bias_scale: float = 0.5

    # Fusion
    fusion_dim: int = 512
    fusion_heads: int = 8
    fusion_ffn_mult: int = 4
    use_patch_attention_pool: bool = True
    dropout: float = 0.1

    # RSSM
    rssm_deter_dim: int = 512
    rssm_stoch_dim: int = 64
    rssm_hidden_dim: int = 512
    min_std: float = 0.1

    # Prediction heads
    head_hidden_dim: int = 256
    direction_bins: int = 8
    distance_bins: int = 6

    # DiT action head
    action_dit_hidden_dim: int = 256
    action_dit_depth: int = 4
    action_dit_heads: int = 8
    # DiT actor predicts a short normalized action sequence [H, action_dim].
    # Online control executes only the first action and replans every frame.
    action_sequence_horizon: int = 3
    action_diffusion_steps: int = 20
    action_sampling_steps: int = 20
    # Optional DiT inference-time candidate selection. When enabled, sample N
    # action sequences, rollout RSSM with each sequence, and execute the first
    # action from the lowest-score sequence.
    dit_candidate_selection: bool = False
    dit_candidate_count: int = 4
    dit_candidate_lateral_weight: float = 1.0
    dit_candidate_vertical_weight: float = 1.0
    dit_candidate_distance_weight: float = 0.05
    dit_candidate_smooth_weight: float = 0.05
    # Tracking-oriented candidate score. These terms are scale-aware: they
    # prefer keeping the predicted target near the image centerline, making
    # progress toward the target, avoiding behind-the-camera states, and
    # rejecting jerky/large sampled actions.
    dit_candidate_yaw_angle_weight: float = 1.0
    dit_candidate_pitch_angle_weight: float = 0.7
    dit_candidate_final_distance_weight: float = 0.25
    dit_candidate_progress_weight: float = 1.0
    dit_candidate_front_weight: float = 0.5
    dit_candidate_action_weight: float = 0.02
    dit_candidate_temporal_smooth_weight: float = 0.05
    action_loss_weight: float = 1.0
    # MSE over action dims: yaw (norm space, index 3 when action_dim=4) vs vx,vy,vz.
    action_yaw_loss_weight: float = 5.0
    max_vel: float = 1.0
    max_yaw_rate: float = 15.0
    max_speed_norm: float = 1.0

    # Loss weights
    kl_weight: float = 0.05

    done_weight: float = 1.0

    # ----- Curriculum / WAM auxiliaries -----
    # train_kl 在方案 A 起即开启（与直连/特权重建一起约束 RSSM）；也可用 checkpoint 覆盖。
    use_diffusion_actor: bool = True
    train_kl: bool = True
    train_direct_action: bool = True
    train_next_target_relative: bool = False
    # Deprecated: prediction-head rollout supervision was removed. RSSM
    # imagination is still used at inference for DiT candidate selection.
    train_rollout: bool = False

    direct_action_loss_weight: float = 1.0
    next_target_relative_loss_weight: float = 1.0
    prior_target_relative_loss_weight: float = 0.2
    # Deprecated with train_rollout.
    rollout_loss_weight: float = 0.2
    # Deprecated with train_rollout.
    rollout_horizon: int = 3
    # x0 loss: MSE between one-step predicted clean action and expert (only when use_diffusion_actor).
    x0_action_loss_weight: float = 1.0

    @property
    def feature_dim(self) -> int:
        return self.rssm_deter_dim + self.rssm_stoch_dim


def migrate_legacy_config(raw_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Map old target-relative names used before the terminology cleanup."""
    aliases = {
        "privileged_dim": "target_relative_dim",
        "privileged_hidden_dim": "target_token_hidden_dim",
        "privileged_fusion_mode": "target_token_fusion_mode",
        "train_next_privileged": "train_next_target_relative",
        "next_privileged_loss_weight": "next_target_relative_loss_weight",
        "prior_privileged_loss_weight": "prior_target_relative_loss_weight",
    }
    out = dict(raw_cfg)
    for old, new in aliases.items():
        if old in out and new not in out:
            out[new] = out[old]
    return out
