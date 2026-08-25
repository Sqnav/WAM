from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F


def _xyxy(box: torch.Tensor) -> torch.Tensor:
    center, size = box[:, :2], box[:, 2:].clamp_min(1e-5)
    return torch.cat([center - size / 2, center + size / 2], dim=1)


def generalized_iou_loss(pred: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    pred, target = _xyxy(pred), _xyxy(target)
    inter_lt = torch.maximum(pred[:, :2], target[:, :2])
    inter_rb = torch.minimum(pred[:, 2:], target[:, 2:])
    inter = (inter_rb - inter_lt).clamp_min(0).prod(1)
    pred_area = (pred[:, 2:] - pred[:, :2]).clamp_min(0).prod(1)
    target_area = (target[:, 2:] - target[:, :2]).clamp_min(0).prod(1)
    union = (pred_area + target_area - inter).clamp_min(1e-6)
    iou = inter / union
    cover_lt = torch.minimum(pred[:, :2], target[:, :2])
    cover_rb = torch.maximum(pred[:, 2:], target[:, 2:])
    cover = (cover_rb - cover_lt).clamp_min(0).prod(1).clamp_min(1e-6)
    giou = iou - (cover - union) / cover
    return (1.0 - giou).mean(), iou.mean()


def gaussian_targets(boxes: torch.Tensor, height: int, width: int) -> torch.Tensor:
    ys = torch.arange(height, device=boxes.device, dtype=boxes.dtype)
    xs = torch.arange(width, device=boxes.device, dtype=boxes.dtype)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    cx, cy = boxes[:, 0] * width, boxes[:, 1] * height
    sigma_x = (boxes[:, 2] * width / 6).clamp_min(0.7)
    sigma_y = (boxes[:, 3] * height / 6).clamp_min(0.7)
    exponent = (
        (grid_x[None] - cx[:, None, None]).square() / (2 * sigma_x[:, None, None].square())
        + (grid_y[None] - cy[:, None, None]).square() / (2 * sigma_y[:, None, None].square())
    )
    return torch.exp(-exponent).unsqueeze(1)


def tracking_loss(model, outputs: Dict[str, torch.Tensor], target: torch.Tensor) -> Dict[str, torch.Tensor]:
    logits = outputs["center_logits"]
    heatmap = gaussian_targets(target, logits.shape[-2], logits.shape[-1])
    location = F.binary_cross_entropy_with_logits(logits, heatmap)
    pred_box = model.decode(outputs)
    l1 = F.l1_loss(pred_box, target)
    giou, iou = generalized_iou_loss(pred_box, target)
    total = location + 5.0 * l1 + 2.0 * giou
    return {"total": total, "location": location, "l1": l1, "giou": giou, "iou": iou}
