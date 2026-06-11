from __future__ import annotations

import os
import sys
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


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


_WAN22_COMPONENT_CACHE = {}


def _torch_dtype_from_name(name: str) -> torch.dtype:
    normalized = str(name).lower()
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16", "half"}:
        return torch.float16
    if normalized in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported Wan2.2 dtype: {name!r}.")


def _ensure_fastwam_path(cfg: ModelConfig) -> None:
    if "fastwam" in sys.modules:
        return
    src_path = os.environ.get("FASTWAM_REPO") or str(getattr(cfg, "wan22_fastwam_src_path", "") or "")
    if src_path and os.path.isdir(os.path.join(src_path, "src")):
        src_path = os.path.join(src_path, "src")
    if src_path and os.path.isdir(src_path) and src_path not in sys.path:
        sys.path.insert(0, src_path)


def _load_wan22_components(cfg: ModelConfig, device: Optional[torch.device] = None):
    _ensure_fastwam_path(cfg)
    try:
        from fastwam.models.wan22.helpers.loader import _load_registered_model, _resolve_configs
        from fastwam.models.wan22.wan_video_text_encoder import HuggingfaceTokenizer
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "Wan2.2 encoders require the official FastWAM package. "
            "Set cfg.wan22_fastwam_src_path or FASTWAM_REPO to the FastWAM source checkout. "
            f"Original import error: {e!r}"
        ) from e

    device_s = "cpu" if device is None else str(device)
    dtype = _torch_dtype_from_name(cfg.wan22_torch_dtype)
    key = (
        cfg.wan22_model_id,
        cfg.wan22_tokenizer_model_id,
        cfg.wan22_model_base_path,
        cfg.wan22_redirect_common_files,
        cfg.wan22_skip_download,
        device_s,
        str(dtype),
    )
    cached = _WAN22_COMPONENT_CACHE.get(key)
    if cached is not None:
        return cached

    old_base = os.environ.get("DIFFSYNTH_MODEL_BASE_PATH")
    old_skip = os.environ.get("DIFFSYNTH_SKIP_DOWNLOAD")
    os.environ["DIFFSYNTH_MODEL_BASE_PATH"] = str(cfg.wan22_model_base_path)
    os.environ["DIFFSYNTH_SKIP_DOWNLOAD"] = "true" if cfg.wan22_skip_download else "false"
    try:
        _, text_config, vae_config, tokenizer_config = _resolve_configs(
            model_id=cfg.wan22_model_id,
            tokenizer_model_id=cfg.wan22_tokenizer_model_id,
            redirect_common_files=bool(cfg.wan22_redirect_common_files),
        )
        text_config.skip_download = bool(cfg.wan22_skip_download)
        vae_config.skip_download = bool(cfg.wan22_skip_download)
        tokenizer_config.skip_download = bool(cfg.wan22_skip_download)
        text_config.download_if_necessary()
        vae_config.download_if_necessary()
        tokenizer_config.download_if_necessary()
        text_encoder = _load_registered_model(
            text_config.path,
            "wan_video_text_encoder",
            torch_dtype=dtype,
            device=device_s,
        )
        tokenizer = HuggingfaceTokenizer(
            name=tokenizer_config.path,
            seq_len=int(cfg.wan22_text_context_length),
            clean="whitespace",
        )
        vae = _load_registered_model(
            vae_config.path,
            "wan_video_vae",
            torch_dtype=dtype,
            device=device_s,
        )
    finally:
        if old_base is None:
            os.environ.pop("DIFFSYNTH_MODEL_BASE_PATH", None)
        else:
            os.environ["DIFFSYNTH_MODEL_BASE_PATH"] = old_base
        if old_skip is None:
            os.environ.pop("DIFFSYNTH_SKIP_DOWNLOAD", None)
        else:
            os.environ["DIFFSYNTH_SKIP_DOWNLOAD"] = old_skip

    loaded = {"text_encoder": text_encoder, "tokenizer": tokenizer, "vae": vae, "dtype": dtype}
    _WAN22_COMPONENT_CACHE[key] = loaded
    return loaded


class Wan22TextEncoder(nn.Module):
    """Wan2.2 UMT5 text encoder adapter used by FastWAM."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        components = _load_wan22_components(cfg)
        self.model = components["text_encoder"]
        self.tokenizer = components["tokenizer"]
        self.output_dim = int(getattr(self.model, "dim", 4096))
        self.cfg.text_width = self.output_dim
        self.cfg.text_context_length = int(cfg.wan22_text_context_length)
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.model.eval()
        self._cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

    def train(self, mode: bool = True) -> "Wan22TextEncoder":
        super().train(mode)
        self.model.eval()
        return self

    @torch.no_grad()
    def encode_texts_with_mask(self, texts, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if isinstance(texts, str):
            texts = [texts]
        if next(self.model.parameters()).device != device:
            self.model.to(device)

        outputs: list[tuple[torch.Tensor, torch.Tensor] | None] = [None] * len(texts)
        missing: list[str] = []
        missing_positions: list[int] = []
        for i, text in enumerate(texts):
            cached = self._cache.get(text)
            if cached is None:
                missing.append(text)
                missing_positions.append(i)
            else:
                outputs[i] = cached

        chunk_size = max(int(getattr(self.cfg, "wan22_text_encode_batch_size", 4)), 1)
        for start in range(0, len(missing), chunk_size):
            chunk = missing[start : start + chunk_size]
            ids, mask = self.tokenizer(chunk, return_mask=True, add_special_tokens=True)
            ids = ids.to(device)
            mask = mask.to(device, dtype=torch.bool)
            tokens = self.model(ids, mask)
            seq_lens = mask.gt(0).sum(dim=1).long()
            for j, v in enumerate(seq_lens):
                tokens[j, v:] = 0.0
            full_mask = torch.ones_like(mask, dtype=torch.bool)
            for j, text in enumerate(chunk):
                cached = (tokens[j].detach().cpu(), full_mask[j].detach().cpu())
                self._cache[text] = cached
                outputs[missing_positions[start + j]] = cached

        if any(x is None for x in outputs):
            raise RuntimeError("Wan22TextEncoder cache assembly failed.")
        tokens = torch.stack([x[0] for x in outputs if x is not None], dim=0).to(device=device)
        mask = torch.stack([x[1] for x in outputs if x is not None], dim=0).to(device=device, dtype=torch.bool)
        pooled = tokens.mean(dim=1)
        return pooled, tokens, mask

    @torch.no_grad()
    def encode_texts(self, texts, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        pooled, tokens, _ = self.encode_texts_with_mask(texts, device)
        return pooled, tokens

    def forward(self, token_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        raise RuntimeError("Wan22TextEncoder expects raw instruction strings; call encode_texts().")


class HeatmapTokenEncoder(nn.Module):
    """Project target heatmap tensors into per-visual-token embeddings."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.proj = nn.Sequential(
            nn.Linear(1, cfg.image_encoder_dim),
            nn.GELU(),
            nn.Linear(cfg.image_encoder_dim, cfg.image_encoder_dim),
        )

    @staticmethod
    def _token_grid(token_count: int, has_cls_token: bool) -> tuple[int, int, int]:
        patch_count = token_count - 1 if has_cls_token else token_count
        grid_h = int(round(patch_count ** 0.5))
        grid_w = grid_h
        if grid_h * grid_w != patch_count:
            grid_h = max(int(patch_count ** 0.5), 1)
            grid_w = int((patch_count + grid_h - 1) // grid_h)
        return patch_count, grid_h, grid_w

    def forward(
        self,
        heatmaps: torch.Tensor,
        token_count: int,
        has_cls_token: bool,
    ) -> torch.Tensor:
        if heatmaps.ndim != 4:
            raise ValueError("heatmaps must have shape [B, 1, H, W].")
        patch_count, grid_h, grid_w = self._token_grid(token_count, has_cls_token)
        h = heatmaps.float()
        if h.size(1) != 1:
            h = h[:, :1]
        pooled = F.interpolate(h, size=(grid_h, grid_w), mode="bilinear", align_corners=False)
        flat = pooled.flatten(2).transpose(1, 2)
        if flat.size(1) > patch_count:
            flat = flat[:, :patch_count]
        elif flat.size(1) < patch_count:
            flat = F.pad(flat, (0, 0, 0, patch_count - flat.size(1)))
        emb = self.proj(flat.to(dtype=self.proj[0].weight.dtype))
        if has_cls_token:
            cls = torch.zeros(emb.size(0), 1, emb.size(-1), device=emb.device, dtype=emb.dtype)
            emb = torch.cat([cls, emb], dim=1)
        return emb


class Wan22VAEImageEncoder(nn.Module):
    """Wan2.2 visual VAE adapter returning latent patch tokens."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        components = _load_wan22_components(cfg)
        self.vae = components["vae"]
        self.dtype = components["dtype"]
        self.output_dim = int(getattr(self.vae, "z_dim", getattr(self.vae.model, "z_dim", 48)))
        self.cfg.image_encoder_dim = self.output_dim
        for p in self.vae.parameters():
            p.requires_grad_(False)
        self.vae.eval()

    def train(self, mode: bool = True) -> "Wan22VAEImageEncoder":
        super().train(mode)
        self.vae.eval()
        return self

    def _normalize_video(self, images: torch.Tensor) -> torch.Tensor:
        x = images.float()
        if x.max() > 1.5:
            x = x / 255.0
        return (x * 2.0 - 1.0).clamp(-1.0, 1.0)

    @torch.no_grad()
    def encode_video_latents(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 5:
            raise ValueError("images must have shape [B, T, C, H, W].")
        device = images.device
        if next(self.vae.parameters()).device != device:
            self.vae.to(device=device, dtype=self.dtype)
        video = self._normalize_video(images).to(device=device, dtype=self.dtype)
        video_list = [sample.permute(1, 0, 2, 3).contiguous() for sample in video]
        return self.vae.encode(video_list, device=device, tiled=False)

    @torch.no_grad()
    def encode_video(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        latents = self.encode_video_latents(images)
        # [B, C, T_lat, H_lat, W_lat] -> [B, T_lat, H_lat*W_lat, C]
        tokens = latents.permute(0, 2, 3, 4, 1).reshape(latents.size(0), latents.size(2), -1, latents.size(1))
        pooled = tokens.mean(dim=2)
        return pooled.float(), tokens.float()

    @torch.no_grad()
    def decode_video_latents(self, latents: torch.Tensor) -> torch.Tensor:
        if latents.ndim != 5:
            raise ValueError("latents must have shape [B, C, T_lat, H_lat, W_lat].")
        device = latents.device
        if next(self.vae.parameters()).device != device:
            self.vae.to(device=device, dtype=self.dtype)
        video = self.vae.decode(latents.to(device=device, dtype=self.dtype), device=device, tiled=False)
        video = video.detach().float().clamp(-1.0, 1.0)
        video = ((video + 1.0) * 127.5).clamp(0, 255).to(torch.uint8)
        return video.permute(0, 2, 3, 4, 1).cpu()

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if images.ndim != 4:
            raise ValueError("Wan22VAEImageEncoder.forward expects [B,C,H,W].")
        pooled, tokens = self.encode_video(images.unsqueeze(1))
        return pooled[:, 0], tokens[:, 0]


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
        dtype = next(self.mlp.parameters()).dtype
        return self.mlp(target_relative.to(dtype=dtype))
