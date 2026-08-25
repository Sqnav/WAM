from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Dict, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[3]


@dataclass
class ModelConfig:
    # Legacy image/text fields are kept only for old checkpoint/config
    # compatibility; the active model path uses Wan2.2 encoders below.
    image_size: int = 224
    in_channels: int = 3
    image_encoder_dim: int = 768
    dinov2_model_name: str = str(PROJECT_ROOT / "model/dinov2-base")
    dinov2_freeze: bool = True
    dinov2_local_files_only: bool = True
    image_normalize: bool = True
    image_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    image_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)

    clip_text_model_name: str = str(PROJECT_ROOT / "model/clip-vit-base-patch32")
    clip_text_freeze: bool = True
    clip_text_local_files_only: bool = True
    text_context_length: int = 77
    text_width: int = 512
    text_pad_id: int = 0

    # Wan2.2 text encoder + visual VAE, matching the official FastWAM entry
    # path. DINOv2/CLIP encoder modules have been removed.
    use_wan22_encoders: bool = True
    wan22_model_id: str = "Wan-AI/Wan2.2-TI2V-5B"
    wan22_tokenizer_model_id: str = "Wan-AI/Wan2.1-T2V-1.3B"
    wan22_model_base_path: str = str(PROJECT_ROOT / "model")
    wan22_fastwam_src_path: str = str(PROJECT_ROOT / "model/FastWAM/src")
    wan22_redirect_common_files: bool = True
    wan22_skip_download: bool = True
    wan22_torch_dtype: str = "bfloat16"
    wan22_text_context_length: int = 512
    wan22_text_encode_batch_size: int = 4

    # Low-dimensional inputs. State input has been removed.
    target_relative_dim: int = 3
    action_dim: int = 4
    target_token_hidden_dim: int = 128
    target_token_fusion_mode: str = "attention"  # attention | concat

    # Privileged target context. When enabled, the current target position in
    # UAV/body coordinates is projected to one extra FastWAM text-context token.
    use_target_relative_context: bool = False
    target_relative_context_scale: float = 1.0
    target_relative_token_scale: float = 1.0
    target_relative_context_hidden_dim: int = 512

    # Tracker center context. The normalized tracker bbox center is projected
    # to one extra Wan text-context token and confidence-gated against a
    # learned missing/uncertain token.
    use_tracker_center_context: bool = False
    tracker_center_context_hidden_dim: int = 256
    tracker_center_token_scale: float = 1.0
    # Frozen DeiT Tracker condition read only by late Action Expert layers.
    tracker_mot_integration: str = "none"
    # Joint Tracker/FastWAM comparison: Tracker receives gradients only from
    # the two MoT flow losses, never from a tracking supervision term.
    tracker_finetune_checkpoint: str = ""
    # Initialization for the Tracker trained jointly with FastWAM.  The
    # ImageNet option loads only the DeiT backbone; Tracker-specific position,
    # segment, and detection-head weights are freshly initialized.
    tracker_finetune_init: str = "uav_tracker"  # uav_tracker | imagenet_deit
    tracker_backbone_pretrained_path: str = ""
    tracker_template_size: int = 128
    tracker_search_size: int = 256
    # Inputs exposed to the independent Action-to-Tracker cross-attention.
    # Existing fusion checkpoints predate this field and therefore retain the
    # original center+feature behavior through this default.
    tracker_condition_mode: str = "center_features"
    tracker_feature_dim: int = 192
    tracker_feature_grid_size: int = 7
    tracker_use_local_position_embedding: bool = False
    tracker_include_box_token: bool = True
    tracker_response_grid_size: int = 7
    # Deprecated checkpoint/script compatibility fields from the removed models.
    tracker_feature_context_hidden_dim: int = 512
    tracker_feature_token_scale: float = 1.0
    # Deprecated compatibility field. Tracker feature integrations always use
    # the observed features, regardless of Tracker confidence.
    tracker_feature_confidence_gate: bool = False
    tracker_fusion_start_layer: int = 18
    tracker_fusion_gate_init: float = 0.0
    # Box-free Template-guided target token plus late Action-to-Video future
    # readout. This is intentionally separate from the generic local feature
    # integration so existing spatial-only checkpoints keep their behavior.
    tracker_future_target_alignment: bool = False
    tracker_future_target_start_layer: int = 18
    # Use the observed current state s0 as the next online Tracker search
    # anchor. This removes the external SquareTracker while keeping future
    # state predictions out of the Tracker's recurrent memory.
    tracker_model_driven_search: bool = False
    tracker_center_flow_supervision: bool = False
    tracker_center_flow_loss_weight: float = 0.1
    # Training-only perturbation of the GT-driven Search crop. Coordinates are
    # expressed as fractions of the resulting Search crop side.
    tracker_search_crop_jitter: bool = False
    tracker_search_center_jitter_std: float = 0.10
    tracker_search_center_jitter_max: float = 0.20
    tracker_search_scale_jitter: float = 0.10
    # Future Target predicts H transitions and H+1 states. States s0..s(H-1)
    # condition actions a0..a(H-1), while sH is a terminal prediction.
    tracker_state_action_alignment_version: int = 3
    tracker_current_center_loss_weight: float = 1.0
    tracker_future_center_loss_weight: float = 1.0
    tracker_center_transition_loss_weight: float = 0.5
    tracker_box_l1_loss_weight: float = 5.0
    tracker_box_giou_loss_weight: float = 2.0
    tracker_future_horizon_discount: float = 0.9
    # When false, Search spatial tokens are consumed only by the box-free
    # target tokenizer; Action queries never attend to them directly.
    tracker_spatial_cross_attention: bool = True
    # V4: independent Future State DiT registered as the third MoT expert.
    use_future_state_dit: bool = False
    future_state_dim: int = 4
    future_state_horizon: int = 8
    future_state_hidden_dim: int = 1024
    future_state_ffn_dim: int = 4096
    future_state_num_layers: int = 30
    future_state_flow_weight: float = 1.0
    current_box_weight: float = 5.0
    current_center_weight: float = 5.0
    current_box_giou_weight: float = 2.0
    current_attention_weight: float = 1.0
    current_attention_sigma: float = 1.5
    localization_warmup_steps: int = 0
    # Keep the current-target loss switch for StateDiT and standalone S0 training.
    include_current_localization_loss: bool = True
    # Current-box-only ablation: one frozen Tracker b0 is broadcast into
    # selected Action DiT layers.
    use_current_box_action_conditioning: bool = False
    current_box_action_layers: tuple[int, ...] = (18, 23, 26, 29)
    current_box_action_hidden_dim: int = 1024
    current_box_action_gate_init: float = 0.0
    freeze_current_box_action_conditioner: bool = False
    # Historical target memory for FastWAM/FasterWAM Current Box policies.
    # The history contains K-1 previous normalized Tracker boxes; the current
    # Tracker b0 is appended inside the model so training and online inference
    # use exactly the same current observation.
    use_historical_target_memory: bool = False
    target_history_length: int = 8
    target_history_hidden_dim: int = 256
    target_history_num_layers: int = 2
    target_history_num_heads: int = 8
    target_history_tracker_cache_root: str = ""
    target_conditioning_adapter_only: bool = False
    # Capture-value reranking for FastWAM/FasterWAM Current Box policies.
    # VideoDiT is prefetched once; N Action candidates share its K/V cache. The
    # learned mode remains DoT-only; geometric and action_prior are training-free.
    use_capture_value_reranking: bool = False
    capture_value_score_mode: str = "learned"  # learned | geometric | action_prior
    capture_value_candidate_count: int = 4
    capture_value_hidden_dim: int = 256
    capture_value_num_layers: int = 2
    capture_value_num_heads: int = 8
    capture_value_loss_weight: float = 0.2
    capture_value_candidate_noise_std: float = 0.15
    capture_value_capture_distance: float = 10.0
    capture_value_distance_score_weight: float = 1.0
    capture_value_visibility_score_weight: float = 0.25
    capture_value_adapter_only: bool = False
    capture_value_control_dt: float = 1.0
    capture_value_horizontal_fov_deg: float = 90.0
    capture_value_vertical_fov_deg: float = 90.0
    # Robust median(distance * normalized sqrt(box area)) on City_1/2/3 1-450.
    capture_value_bbox_depth_scale: float = 0.2698
    capture_value_min_depth: float = 1.0
    capture_value_max_depth: float = 20.0
    # Median normalized sqrt(box area) on City_1/2/3 1-450.
    capture_value_target_box_size: float = 0.06094
    capture_value_box_size_sigma: float = 0.01
    capture_value_discount: float = 0.8
    capture_value_recenter_sigma: float = 0.35
    capture_value_pursuit_center_sigma: float = 0.40
    capture_value_out_of_frame_weight: float = 2.0
    capture_value_first_action_smooth_weight: float = 2.0
    capture_value_temporal_smooth_weight: float = 1.0
    capture_value_recenter_weight: float = 2.0
    capture_value_pursuit_weight: float = 0.7
    capture_value_smooth_weight: float = 0.1
    capture_value_consensus_weight: float = 0.1
    capture_value_short_horizon: int = 1
    capture_value_selection_margin: float = 0.0
    # Recenter candidates are allowed to override the parent policy only when
    # the normalized target-center radius is large enough to need correction.
    capture_value_min_center_error: float = 0.30
    capture_action_prior_checkpoint: str = ""
    capture_action_prior_dimension_weights: tuple[float, ...] = (0.0, 1.0, 1.0, 1.0)
    capture_value_structured_candidates: bool = True
    use_tracker_memory: bool = True
    tracker_expert_hidden_dim: int = 256
    tracker_expert_ffn_dim: int = 1024
    # Training target used by Tracker attention consistency loss.
    tracker_heatmap_target_mode: str = "canonical"  # canonical | raw | raw_area
    tracker_attention_query_mode: str = "all_queries"  # query0 | all_queries

    # Fusion
    fusion_dim: int = 512
    fusion_heads: int = 8
    fusion_ffn_mult: int = 4
    use_patch_attention_pool: bool = True
    dropout: float = 0.1

    # RSSM
    use_rssm: bool = False
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
    action_sequence_horizon: int = 8
    # FastWAM temporal layout. The training window contains action timesteps;
    # video frames are sampled every N action steps before Wan VAE encoding.
    # Use every frame in a 9-frame window, yielding 8 consecutive actions and
    # 9 RGB observations (3 Wan latent frames).
    fastwam_action_video_freq_ratio: int = 1
    action_diffusion_steps: int = 20
    action_sampling_steps: int = 20
    # Compile the fixed-shape action diffusion loop. PyTorch's reduce-overhead
    # mode enables Inductor CUDA Graphs on supported CUDA devices.
    compile_action_sampling: bool = False
    compile_action_sampling_mode: str = "reduce-overhead"
    # Optional DiT inference-time candidate selection. This requires use_rssm=True.
    # The default Fast-WAM-style path keeps it disabled and directly executes
    # the first predicted action.
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
    # Fast-WAM-style default: no recurrent RSSM and no KL. The world head is a
    # training auxiliary on direct observation features.
    use_diffusion_actor: bool = True
    train_kl: bool = False
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
    # Clean-action reconstruction from the FastWAM flow prediction.
    x0_action_loss_weight: float = 0.5

    # FastWAM-style video/action MoT.
    use_fastwam_mot: bool = True
    fastwam_hidden_dim: int = 256
    fastwam_layers: int = 4
    fastwam_heads: int = 8
    fastwam_video_train_timesteps: int = 1000
    fastwam_action_train_timesteps: int = 1000
    fastwam_video_shift: float = 5.0
    fastwam_action_shift: float = 5.0
    fastwam_lambda_video: float = 1.0
    fastwam_lambda_action: float = 1.0
    fastwam_use_official_wan_experts: bool = True
    fastwam_skip_dit_load_from_pretrain: bool = False
    fastwam_action_dit_pretrained_path: str = ""
    fastwam_mot_checkpoint_mixed_attn: bool = True
    use_fasterwam_dot: bool = False
    use_fastwam_attention_heatmap_loss: bool = False
    use_fastwam_tracker_heatmap_loss: bool = False
    use_fastwam_attention_bias: bool = False
    # Privileged GT bbox-center prior injected into action-query -> first-frame
    # visual-key logits. This is separate from the legacy heatmap-log-prior path.
    use_gt_center_attention_bias: bool = False
    gt_center_attention_sigma: float = 1.0
    gt_center_attention_beta: float = 2.0
    gt_center_attention_zero_mean: bool = True
    gt_center_guided_layers: int = 3
    gt_center_guided_head_ratio: float = 0.5
    return_gt_center_attention_for_distillation: bool = False
    fastwam_heatmap_source: str = "none"
    fastwam_attention_heatmap_loss_weight: float = 0.2
    fastwam_attention_heatmap_sigma: float = 0.08
    fastwam_attention_heatmap_fov_deg: float = 90.0
    fastwam_attention_heatmap_camera_offset_body: Tuple[float, float, float] = (0.46, 0.0, 0.0)
    use_fastwam_heatmap_guidance: bool = False
    fastwam_heatmap_guidance_scale: float = 1.0
    fastwam_heatmap_guidance_sigma: float = 0.08
    fastwam_heatmap_guidance_fov_deg: float = 90.0
    fastwam_heatmap_guidance_camera_offset_body: Tuple[float, float, float] = (0.46, 0.0, 0.0)
    fastwam_ortrack_consistency_loss_weight: float = 0.2

    @property
    def feature_dim(self) -> int:
        if not self.use_rssm:
            return self.fusion_dim
        return self.rssm_deter_dim + self.rssm_stoch_dim


def migrate_legacy_config(raw_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Map legacy field names and paths stored in older checkpoints."""
    aliases = {
        "privileged_dim": "target_relative_dim",
        "privileged_hidden_dim": "target_token_hidden_dim",
        "privileged_fusion_mode": "target_token_fusion_mode",
        "train_next_privileged": "train_next_target_relative",
        "next_privileged_loss_weight": "next_target_relative_loss_weight",
        "prior_privileged_loss_weight": "prior_target_relative_loss_weight",
    }
    out = dict(raw_cfg)
    legacy_project_root = "/data1/ysq/Worldmodel"
    for key, value in out.items():
        if isinstance(value, str) and (
            value == legacy_project_root or value.startswith(f"{legacy_project_root}/")
        ):
            relative_path = value[len(legacy_project_root):].lstrip("/")
            out[key] = str(PROJECT_ROOT / relative_path) if relative_path else str(PROJECT_ROOT)
    for old, new in aliases.items():
        if old in out and new not in out:
            out[new] = out[old]
    if (
        bool(out.get("tracker_future_target_alignment", False))
        and "tracker_state_action_alignment_version" not in out
    ):
        # Old Future Target checkpoints associated action i with its post-action
        # state. Preserve that identity so the model rejects silent v1 -> v2 use.
        out["tracker_state_action_alignment_version"] = 1
    valid_fields = {field.name for field in fields(ModelConfig)}
    return {key: value for key, value in out.items() if key in valid_fields}
