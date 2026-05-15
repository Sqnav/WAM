from dataclasses import dataclass
from typing import Tuple


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
    privileged_dim: int = 3
    action_dim: int = 4
    privileged_hidden_dim: int = 128

    # Fusion
    fusion_dim: int = 512
    fusion_heads: int = 8
    fusion_ffn_mult: int = 4
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
    action_diffusion_steps: int = 20
    action_sampling_steps: int = 20
    action_loss_weight: float = 1.0
    max_vel: float = 1.0
    max_yaw_rate: float = 45.0
    max_speed_norm: float = 1.0

    # Loss weights
    kl_weight: float = 0.05

    # KL warmup：从 kl_warmup_start * kl_weight 线性升到 kl_weight
    # 设为 0 表示不使用 warmup，直接使用 kl_weight
    kl_warmup_steps: int = 10000
    kl_warmup_start: float = 0.0

    
    reward_weight: float = 1.0
    done_weight: float = 1.0

    # Additional prior auxiliary loss. This forces the action-conditioned prior
    # to predict task variables, instead of relying only on posterior features.
    prior_loss_weight: float = 0.5

    @property
    def feature_dim(self) -> int:
        return self.rssm_deter_dim + self.rssm_stoch_dim
