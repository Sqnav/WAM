from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn.functional as F
from torch import nn


def _aligned_iou_and_giou(
    boxes1: torch.Tensor, boxes2: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Aligned IoU/GIoU for normalized cxcywh boxes."""
    half1 = boxes1[..., 2:].clamp_min(1.0e-6) * 0.5
    half2 = boxes2[..., 2:].clamp_min(1.0e-6) * 0.5
    first1, second1 = boxes1[..., :2] - half1, boxes1[..., :2] + half1
    first2, second2 = boxes2[..., :2] - half2, boxes2[..., :2] + half2
    intersection_wh = (
        torch.minimum(second1, second2) - torch.maximum(first1, first2)
    ).clamp_min(0.0)
    intersection = intersection_wh.prod(dim=-1)
    area1 = (second1 - first1).clamp_min(0.0).prod(dim=-1)
    area2 = (second2 - first2).clamp_min(0.0).prod(dim=-1)
    union = (area1 + area2 - intersection).clamp_min(1.0e-8)
    iou = intersection / union
    enclosing_wh = (
        torch.maximum(second1, second2) - torch.minimum(first1, first2)
    ).clamp_min(0.0)
    enclosing = enclosing_wh.prod(dim=-1).clamp_min(1.0e-8)
    giou = iou - (enclosing - union) / enclosing
    return iou, giou


class CurrentTargetLocalizer(nn.Module):
    """Template-guided Target Query that predicts only the observed box b0."""

    def __init__(self, tracker_dim: int, hidden_dim: int, num_heads: int = 8) -> None:
        super().__init__()
        self.tracker_dim = int(tracker_dim)
        self.hidden_dim = int(hidden_dim)
        self.template_projection = nn.Sequential(
            nn.LayerNorm(tracker_dim),
            nn.Linear(tracker_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.target_query = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.template_pool = nn.MultiheadAttention(
            hidden_dim, num_heads=num_heads, batch_first=True
        )
        self.target_search = nn.MultiheadAttention(
            hidden_dim, num_heads=num_heads, batch_first=True
        )
        size_hidden = max(hidden_dim // 4, 1)
        self.current_size_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, size_hidden),
            nn.GELU(),
            nn.Linear(size_hidden, 2),
        )
        nn.init.trunc_normal_(self.target_query, std=0.02)
        nn.init.constant_(self.current_size_head[-1].bias, -3.0)

    def forward(
        self,
        template_features: torch.Tensor,
        search_tokens: torch.Tensor,
        full_xy: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if template_features.ndim != 3 or template_features.shape[1:] != (
            64,
            self.tracker_dim,
        ):
            raise ValueError(
                f"template_features must be [B,64,{self.tracker_dim}], "
                f"got {tuple(template_features.shape)}."
            )
        batch = search_tokens.size(0)
        if search_tokens.shape != (batch, 256, self.hidden_dim):
            raise ValueError(
                f"search_tokens must be [B,256,{self.hidden_dim}], "
                f"got {tuple(search_tokens.shape)}."
            )
        if full_xy.shape != (batch, 256, 2):
            raise ValueError(f"full_xy must be [B,256,2], got {tuple(full_xy.shape)}.")
        template = self.template_projection(template_features.to(search_tokens))
        query = self.target_query.to(search_tokens).expand(batch, -1, -1)
        identity, _ = self.template_pool(query, template, template, need_weights=False)
        current_token, attention = self.target_search(
            identity,
            search_tokens,
            search_tokens,
            need_weights=True,
            average_attn_weights=True,
        )
        attention = attention.float().clamp_min(1.0e-8)
        attention = attention / attention.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)
        center = torch.bmm(attention.float(), full_xy.float()).squeeze(1)
        size = torch.sigmoid(
            self.current_size_head(current_token.squeeze(1)).float()
        ).clamp(1.0e-4, 1.0)
        current_box = torch.cat([center.clamp(0.0, 1.0), size], dim=-1)
        if not torch.isfinite(current_box).all():
            raise FloatingPointError("Current Target localization produced NaN/Inf.")
        return {
            "current_box": current_box,
            "current_token": current_token.squeeze(1),
            "current_attention": attention,
            "full_xy": full_xy.float(),
        }

    @staticmethod
    def losses(
        prediction: Dict[str, torch.Tensor],
        target_box: torch.Tensor,
        valid: torch.Tensor,
        image_size: torch.Tensor,
        attention_sigma: float,
    ) -> Dict[str, torch.Tensor]:
        current_box = prediction["current_box"].float()
        attention = prediction["current_attention"].float()
        full_xy = prediction["full_xy"].float()
        batch = current_box.size(0)
        if target_box.shape != (batch, 4):
            raise ValueError("Current target box must be [B,4].")
        target = target_box.to(current_box)
        sample_valid = valid.to(current_box).reshape(batch)
        denominator = sample_valid.sum().clamp_min(1.0)

        l1 = F.smooth_l1_loss(current_box, target, reduction="none").mean(dim=-1)
        _, giou = _aligned_iou_and_giou(current_box, target)
        grid_size = int(round(math.sqrt(full_xy.size(1))))
        if grid_size * grid_size != full_xy.size(1):
            raise ValueError("Current Target Search tokens must form a square grid.")
        span = (full_xy.amax(dim=1) - full_xy.amin(dim=1)).clamp_min(1.0e-6)
        crop_span = span * (float(grid_size) / float(max(grid_size - 1, 1)))
        center_delta = (current_box[:, :2] - target[:, :2]) / crop_span
        center = F.smooth_l1_loss(
            center_delta, torch.zeros_like(center_delta), reduction="none"
        ).mean(dim=-1)
        cell_span = crop_span / float(grid_size)
        offsets = (full_xy - target[:, None, :2]) / cell_span[:, None, :]
        sigma = max(float(attention_sigma), 1.0e-3)
        target_attention = torch.softmax(
            -0.5 * offsets.square().sum(dim=-1) / (sigma * sigma), dim=-1
        )
        predicted_attention = attention[:, 0].clamp_min(1.0e-8)
        attention_kl = (
            target_attention
            * (
                target_attention.clamp_min(1.0e-8).log()
                - predicted_attention.log()
            )
        ).sum(dim=-1)
        image_wh = image_size.to(current_box)[:, [1, 0]]
        center_pixels = (
            (current_box[:, :2] - target[:, :2]) * image_wh
        ).square().sum(dim=-1).sqrt()

        def masked(value: torch.Tensor) -> torch.Tensor:
            return (value * sample_valid).sum() / denominator

        return {
            "box_l1": masked(l1),
            "center": masked(center),
            "giou": masked(1.0 - giou),
            "attention": masked(attention_kl),
            "center_error_pixels": masked(center_pixels),
            "valid_ratio": sample_valid.mean(),
        }
