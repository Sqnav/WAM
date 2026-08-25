from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn


class ConvHead(nn.Module):
    def __init__(self, channels: int, square_boxes: bool = False) -> None:
        super().__init__()
        self.square_boxes = bool(square_boxes)
        self.shared = nn.Sequential(
            nn.Conv2d(channels, 256, 3, padding=1),
            nn.GroupNorm(32, 256),
            nn.GELU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.GroupNorm(32, 256),
            nn.GELU(),
        )
        self.center = nn.Conv2d(256, 1, 1)
        self.size = nn.Conv2d(256, 1 if self.square_boxes else 2, 1)
        self.offset = nn.Conv2d(256, 2, 1)
        nn.init.constant_(self.center.bias, -2.19)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.shared(x)
        size = self.size(x).sigmoid()
        if self.square_boxes:
            size = size.repeat(1, 2, 1, 1)
        return {
            "center_logits": self.center(x),
            "size": size,
            "offset": self.offset(x).sigmoid(),
        }


class UAVTracker(nn.Module):
    """A single-stream template/search tracker with no auxiliary teacher."""

    def __init__(
        self,
        backbone: str = "deit_tiny_patch16_224",
        pretrained: bool = True,
        pretrained_path: Optional[Path] = None,
        template_size: int = 128,
        search_size: int = 256,
        square_boxes: bool = False,
        enable_head: bool = True,
    ) -> None:
        super().__init__()
        import timm

        base = timm.create_model(backbone, pretrained=pretrained and pretrained_path is None, num_classes=0)
        if pretrained_path is not None:
            checkpoint = torch.load(pretrained_path, map_location="cpu", weights_only=True)
            state = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
            missing, unexpected = base.load_state_dict(state, strict=False)
            if len(missing) > 4 or len(unexpected) > 4:
                print(f"pretrained load: missing={len(missing)} unexpected={len(unexpected)}")
        if not hasattr(base, "patch_embed") or not hasattr(base, "blocks"):
            raise ValueError(f"Backbone {backbone!r} does not expose ViT patch tokens")
        self.patch_embed = base.patch_embed
        self.patch_embed.strict_img_size = False
        self.blocks = base.blocks
        self.norm = base.norm
        self.pos_drop = base.pos_drop
        self.embed_dim = int(base.embed_dim)
        self.patch_size = int(base.patch_embed.patch_size[0])
        self.template_size = int(template_size)
        self.search_size = int(search_size)
        self.square_boxes = bool(square_boxes)
        self.template_grid = self.template_size // self.patch_size
        self.search_grid = self.search_size // self.patch_size
        self.template_pos = nn.Parameter(torch.zeros(1, self.template_grid**2, self.embed_dim))
        self.search_pos = nn.Parameter(torch.zeros(1, self.search_grid**2, self.embed_dim))
        self.segment_embed = nn.Parameter(torch.zeros(1, 2, self.embed_dim))
        nn.init.trunc_normal_(self.template_pos, std=0.02)
        nn.init.trunc_normal_(self.search_pos, std=0.02)
        nn.init.trunc_normal_(self.segment_embed, std=0.02)
        self.head = (
            ConvHead(self.embed_dim, square_boxes=self.square_boxes)
            if bool(enable_head)
            else None
        )

    def _tokens(self, image: torch.Tensor) -> torch.Tensor:
        tokens = self.patch_embed(image)
        return tokens.flatten(2).transpose(1, 2) if tokens.ndim == 4 else tokens

    def forward(
        self,
        template: torch.Tensor,
        search: torch.Tensor,
        *,
        return_head: bool = True,
    ) -> Dict[str, torch.Tensor]:
        z = self._tokens(template)
        x = self._tokens(search)
        if z.shape[1] != self.template_pos.shape[1] or x.shape[1] != self.search_pos.shape[1]:
            raise ValueError("Input sizes do not match the configured template/search sizes")
        z = z + self.template_pos + self.segment_embed[:, 0:1]
        x = x + self.search_pos + self.segment_embed[:, 1:2]
        tokens = self.pos_drop(torch.cat([z, x], dim=1))
        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.norm(tokens)
        template_tokens = tokens[:, : self.template_grid**2]
        search_tokens = tokens[:, -self.search_grid**2 :]
        feature = search_tokens.transpose(1, 2).reshape(
            search.shape[0], self.embed_dim, self.search_grid, self.search_grid
        )
        if return_head and self.head is None:
            raise RuntimeError("UAVTracker detection head is disabled for this model.")
        outputs: Dict[str, torch.Tensor] = self.head(feature) if return_head else {}
        # Target-aware search representation after template/search attention.
        # Tracking losses ignore it; WAM integrations consume it.
        outputs["template_features"] = template_tokens
        outputs["search_features"] = feature
        return outputs

    @staticmethod
    def decode(outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Decode normalized cxcywh boxes using a differentiable soft center."""
        logits = outputs["center_logits"]
        batch, _, height, width = logits.shape
        prob = F.softmax(logits.flatten(2), dim=-1).reshape(batch, 1, height, width)
        ys = torch.arange(height, device=logits.device, dtype=logits.dtype)
        xs = torch.arange(width, device=logits.device, dtype=logits.dtype)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        off_x = (prob * outputs["offset"][:, 0:1]).sum((2, 3))
        off_y = (prob * outputs["offset"][:, 1:2]).sum((2, 3))
        cx = ((prob[:, 0] * grid_x).sum((1, 2)).unsqueeze(1) + off_x) / width
        cy = ((prob[:, 0] * grid_y).sum((1, 2)).unsqueeze(1) + off_y) / height
        size = (prob * outputs["size"]).sum((2, 3))
        return torch.cat([cx, cy, size], dim=1)

    @staticmethod
    def decode_peak(outputs: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decode the highest-confidence cell for every sample in a batch."""
        logits = outputs["center_logits"]
        if logits.ndim != 4 or logits.size(1) != 1:
            raise ValueError("center_logits must be [B,1,H,W].")
        batch, _, height, width = logits.shape
        flat_logits = logits[:, 0].flatten(1)
        indices = flat_logits.argmax(dim=-1)
        rows = torch.div(indices, width, rounding_mode="floor")
        columns = indices.remainder(width)
        batch_indices = torch.arange(batch, device=logits.device)
        offsets = outputs["offset"][batch_indices, :, rows, columns]
        sizes = outputs["size"][batch_indices, :, rows, columns]
        centers = torch.stack(
            [
                (columns.to(offsets) + offsets[:, 0]) / float(width),
                (rows.to(offsets) + offsets[:, 1]) / float(height),
            ],
            dim=-1,
        )
        boxes = torch.cat([centers, sizes], dim=-1)
        confidence = flat_logits.sigmoid().gather(1, indices[:, None]).squeeze(1)
        return boxes, confidence

    @staticmethod
    def map_crop_boxes_to_image(
        crop_boxes: torch.Tensor,
        search_geometry: torch.Tensor,
        image_size: torch.Tensor,
    ) -> torch.Tensor:
        """Map crop-normalized cxcywh boxes to clipped full-image normalized boxes."""
        if crop_boxes.ndim != 2 or crop_boxes.size(-1) != 4:
            raise ValueError("crop_boxes must be [B,4] normalized cxcywh.")
        if search_geometry.shape != (crop_boxes.size(0), 3):
            raise ValueError("search_geometry must be [B,3] as [x1,y1,side].")
        if image_size.shape != (crop_boxes.size(0), 2):
            raise ValueError("image_size must be [B,2] as [height,width].")

        geometry = torch.nan_to_num(
            search_geometry.to(crop_boxes).float(), nan=0.0, posinf=0.0, neginf=0.0
        )
        image_hw = torch.nan_to_num(
            image_size.to(crop_boxes).float(), nan=1.0, posinf=1.0, neginf=1.0
        ).clamp_min(1.0)
        crop = crop_boxes.float()
        search_x, search_y, search_side = geometry.unbind(dim=-1)
        image_h, image_w = image_hw.unbind(dim=-1)
        box_w = crop[:, 2] * search_side
        box_h = crop[:, 3] * search_side
        center_x = search_x + crop[:, 0] * search_side
        center_y = search_y + crop[:, 1] * search_side

        left = (center_x - 0.5 * box_w).clamp_min(0.0)
        top = (center_y - 0.5 * box_h).clamp_min(0.0)
        left = torch.minimum(left, (image_w - 1.0).clamp_min(0.0))
        top = torch.minimum(top, (image_h - 1.0).clamp_min(0.0))
        right = torch.minimum(
            torch.maximum(center_x + 0.5 * box_w, left + 1.0), image_w
        )
        bottom = torch.minimum(
            torch.maximum(center_y + 0.5 * box_h, top + 1.0), image_h
        )
        full_width = right - left
        full_height = bottom - top
        result = torch.stack(
            [
                (left + 0.5 * full_width) / image_w,
                (top + 0.5 * full_height) / image_h,
                full_width / image_w,
                full_height / image_h,
            ],
            dim=-1,
        )
        return result.clamp(0.0, 1.0)

    @staticmethod
    def full_image_grid_coordinates(
        search_geometry: torch.Tensor,
        image_size: torch.Tensor,
        grid_height: int,
        grid_width: int,
    ) -> torch.Tensor:
        """Map Search feature-cell centers to normalized full-image coordinates."""
        if search_geometry.ndim != 2 or search_geometry.size(-1) != 3:
            raise ValueError("search_geometry must be [B,3] as [x1,y1,side].")
        if image_size.shape != (search_geometry.size(0), 2):
            raise ValueError("image_size must be [B,2] as [height,width].")
        if grid_height < 1 or grid_width < 1:
            raise ValueError("Tracker feature grid dimensions must be positive.")
        geometry = torch.nan_to_num(
            search_geometry.float(), nan=0.0, posinf=0.0, neginf=0.0
        )
        image_hw = torch.nan_to_num(
            image_size.float(), nan=1.0, posinf=1.0, neginf=1.0
        ).clamp_min(1.0)
        y = (torch.arange(grid_height, device=geometry.device).float() + 0.5) / float(
            grid_height
        )
        x = (torch.arange(grid_width, device=geometry.device).float() + 0.5) / float(
            grid_width
        )
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        local = torch.stack([xx, yy], dim=-1).reshape(1, -1, 2)
        x1, y1, side = geometry.unbind(dim=-1)
        image_h, image_w = image_hw.unbind(dim=-1)
        full_x = (x1[:, None] + local[..., 0] * side[:, None]) / image_w[:, None]
        full_y = (y1[:, None] + local[..., 1] * side[:, None]) / image_h[:, None]
        return torch.stack([full_x, full_y], dim=-1)

    def parameter_groups(self, lr: float, backbone_multiplier: float = 0.1):
        head_names = ("head.", "template_pos", "search_pos", "segment_embed")
        head, backbone = [], []
        for name, parameter in self.named_parameters():
            (head if name.startswith(head_names) else backbone).append(parameter)
        return [
            {"params": head, "lr": lr},
            {"params": backbone, "lr": lr * backbone_multiplier},
        ]
