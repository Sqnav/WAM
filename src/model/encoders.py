from __future__ import annotations

from contextlib import nullcontext
from typing import Optional, Tuple

import torch
import torch.nn as nn

from .config import ModelConfig

try:
    from transformers import CLIPTextModel, Dinov2Model
    _TRANSFORMERS_IMPORT_ERROR = None
except Exception as e:  # pragma: no cover
    CLIPTextModel = None
    Dinov2Model = None
    _TRANSFORMERS_IMPORT_ERROR = e


class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        layers = []
        dim = in_dim
        for _ in range(num_layers - 1):
            layers.extend(
                [
                    nn.Linear(dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
            dim = hidden_dim
        layers.append(nn.Linear(dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DINOv2ImageEncoder(nn.Module):
    """DINOv2 image encoder returning CLS feature and full token sequence."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        if Dinov2Model is None:
            raise ImportError(
                "DINOv2 encoder requires transformers.Dinov2Model. "
                f"Real import error: {repr(_TRANSFORMERS_IMPORT_ERROR)}"
            )

        self.cfg = cfg
        self.model = Dinov2Model.from_pretrained(
            cfg.dinov2_model_name,
            local_files_only=cfg.dinov2_local_files_only,
            use_safetensors=True,
        )
        self.output_dim = int(self.model.config.hidden_size)
        self.cfg.image_encoder_dim = self.output_dim

        mean = torch.tensor(cfg.image_mean, dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor(cfg.image_std, dtype=torch.float32).view(1, 3, 1, 1)
        self.register_buffer("image_mean_tensor", mean, persistent=False)
        self.register_buffer("image_std_tensor", std, persistent=False)

        if cfg.dinov2_freeze:
            for p in self.model.parameters():
                p.requires_grad = False
            self.model.eval()

    def train(self, mode: bool = True) -> "DINOv2ImageEncoder":
        super().train(mode)
        if self.cfg.dinov2_freeze:
            self.model.eval()
        return self

    def _prepare_pixel_values(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4:
            raise ValueError("images must have shape [B, C, H, W].")
        x = images.float()
        if x.max() > 1.5:
            x = x / 255.0
        if self.cfg.image_normalize:
            x = (x - self.image_mean_tensor) / self.image_std_tensor
        return x

    def forward(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        pixel_values = self._prepare_pixel_values(images)
        ctx = torch.no_grad() if self.cfg.dinov2_freeze else nullcontext()
        with ctx:
            outputs = self.model(pixel_values=pixel_values)
        tokens = outputs.last_hidden_state
        pooled = tokens[:, 0]
        return pooled, tokens


# Backward-compatible alias so older imports do not break.
ViTImageEncoder = DINOv2ImageEncoder


class CLIPTextEncoder(nn.Module):
    """Real pretrained CLIP text encoder, not a randomly initialized CLIP-style transformer."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        if CLIPTextModel is None:
            raise ImportError(
                "CLIP text encoder requires transformers.CLIPTextModel. "
                f"Real import error: {repr(_TRANSFORMERS_IMPORT_ERROR)}"
            )

        self.cfg = cfg
        self.model = CLIPTextModel.from_pretrained(
            cfg.clip_text_model_name,
            local_files_only=cfg.clip_text_local_files_only,
            use_safetensors=True,
        )
        self.output_dim = int(self.model.config.hidden_size)
        self.cfg.text_width = self.output_dim

        if cfg.clip_text_freeze:
            for p in self.model.parameters():
                p.requires_grad = False
            self.model.eval()

    def train(self, mode: bool = True) -> "CLIPTextEncoder":
        super().train(mode)
        if self.cfg.clip_text_freeze:
            self.model.eval()
        return self

    def forward(
        self,
        token_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape [B, L].")
        if token_ids.size(1) > self.cfg.text_context_length:
            raise ValueError(
                f"Text sequence length {token_ids.size(1)} exceeds configured "
                f"context length {self.cfg.text_context_length}."
            )
        if attention_mask is None:
            attention_mask = token_ids.ne(self.cfg.text_pad_id).long()

        ctx = torch.no_grad() if self.cfg.clip_text_freeze else nullcontext()
        with ctx:
            outputs = self.model(input_ids=token_ids, attention_mask=attention_mask)

        tokens = outputs.last_hidden_state
        pooled = getattr(outputs, "pooler_output", None)
        if pooled is None:
            lengths = attention_mask.long().sum(dim=1).clamp(min=1) - 1
            pooled = tokens[torch.arange(tokens.size(0), device=tokens.device), lengths]
        return pooled, tokens


class TargetTokenEncoder(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.mlp = MLP(
            cfg.target_relative_dim,
            cfg.target_token_hidden_dim,
            cfg.fusion_dim,
            num_layers=2,
            dropout=cfg.dropout,
        )

    def forward(self, target_relative: torch.Tensor) -> torch.Tensor:
        return self.mlp(target_relative)
