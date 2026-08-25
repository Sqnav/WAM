from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import cv2
import numpy as np
import torch

from tracking.data import MEAN, STD, crop_target
from tracking.evaluate import crop_search, decode_peak, map_box
from tracking.model import UAVTracker


def _image_tensor(image: np.ndarray) -> torch.Tensor:
    array = np.asarray(image, dtype=np.uint8)
    return torch.from_numpy(array.copy()).permute(2, 0, 1).float().div_(255.0)


def _crop_geometry(box: Sequence[float], factor: float) -> Tuple[int, int, int]:
    x, y, width, height = [float(v) for v in box]
    side = max(int(math.ceil(math.sqrt(max(width * height, 1.0)) * factor)), 2)
    return int(math.floor(x + width / 2 - side / 2)), int(math.floor(y + height / 2 - side / 2)), side


def _response_to_image(
    response: np.ndarray,
    previous_box: Sequence[float],
    search_factor: float,
    image_hw: Tuple[int, int],
) -> np.ndarray:
    height, width = image_hw
    x1, y1, side = _crop_geometry(previous_box, search_factor)
    resized = cv2.resize(np.asarray(response, dtype=np.float32), (side, side), interpolation=cv2.INTER_CUBIC)
    full = np.zeros((height, width), dtype=np.float32)
    dx1, dy1 = max(x1, 0), max(y1, 0)
    dx2, dy2 = min(x1 + side, width), min(y1 + side, height)
    if dx2 > dx1 and dy2 > dy1:
        sx1, sy1 = dx1 - x1, dy1 - y1
        full[dy1:dy2, dx1:dx2] = resized[sy1 : sy1 + dy2 - dy1, sx1 : sx1 + dx2 - dx1]
    return np.maximum(full, 0.0)


def _confidence_heatmap(response: np.ndarray, confidence: float) -> np.ndarray:
    response = np.maximum(np.asarray(response, dtype=np.float32), 0.0)
    response /= max(float(response.sum()), 1.0e-8)
    uniform = np.full_like(response, 1.0 / max(response.size, 1))
    confidence = float(np.clip(confidence, 0.0, 1.0))
    result = confidence * response + (1.0 - confidence) * uniform
    return result / max(float(result.sum()), 1.0e-8)


class SquareTracker:
    """Online adapter for the locally trained square-box UAV tracker."""

    def __init__(self, checkpoint: Path, device: str = "cuda", feature_grid_size: int = 7) -> None:
        checkpoint = Path(checkpoint).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Square Tracker checkpoint not found: {checkpoint}")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        args = payload.get("args", {})
        self.model = UAVTracker(
            backbone=args.get("backbone", "deit_tiny_patch16_224"),
            pretrained=False,
            template_size=int(args.get("template_size", 128)),
            search_size=int(args.get("search_size", 256)),
            square_boxes=bool(args.get("square_boxes", True)),
        )
        self.model.load_state_dict(payload["model"], strict=True)
        self.device = torch.device(device)
        self.model.to(self.device).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.template_size = int(self.model.template_size)
        self.search_size = int(self.model.search_size)
        self.template_factor = 2.0
        self.search_factor = 4.0
        self.feature_grid_size = max(int(feature_grid_size), 1)
        self.state: List[float] | None = None
        self.template: torch.Tensor | None = None

    def initialize(self, image: np.ndarray, bbox: Sequence[float]) -> Dict[str, Any]:
        tensor = _image_tensor(image)
        template, _ = crop_target(tensor, bbox, self.template_factor, self.template_size)
        self.template = template.unsqueeze(0).to(self.device)
        self.state = [float(v) for v in bbox]
        return {"template": image, "resize_factor": 1.0}

    @torch.inference_mode()
    def track(self, image: np.ndarray) -> Dict[str, Any]:
        if self.state is None or self.template is None:
            raise RuntimeError("SquareTracker must be initialized before track().")
        tensor = _image_tensor(image)
        previous_box = list(self.state)
        search, geometry = crop_search(tensor, previous_box, self.search_factor, self.search_size)
        outputs = self.model(self.template, search.unsqueeze(0).to(self.device))
        normalized_box, confidence = decode_peak(outputs)
        self.state = map_box(normalized_box.cpu(), geometry, tensor.shape[-2:])
        response_crop = outputs["center_logits"].sigmoid()[0, 0].float().cpu().numpy()
        response = _response_to_image(response_crop, previous_box, self.search_factor, tensor.shape[-2:])
        feature_grid = outputs["search_features"].float()
        if tuple(feature_grid.shape[-2:]) != (
            self.feature_grid_size,
            self.feature_grid_size,
        ):
            feature_grid = torch.nn.functional.adaptive_avg_pool2d(
                feature_grid,
                (self.feature_grid_size, self.feature_grid_size),
            )
        feature_tokens = feature_grid.flatten(2).transpose(1, 2)[0].cpu().numpy()
        return {
            "bbox": list(self.state),
            "confidence": float(confidence),
            "response": response,
            "heatmap": _confidence_heatmap(response, confidence),
            "search_region": np.asarray(image, dtype=np.uint8),
            "search_crop_xy_size": list(_crop_geometry(previous_box, self.search_factor)),
            "feature_tokens": feature_tokens,
            # These are the normalized crops used to advance the online
            # Tracker state. Joint MoT Tracker inference consumes identical
            # inputs with its fine-tuned Tracker weights.
            "tracker_template": self.template,
            "tracker_search": search.unsqueeze(0).to(self.device),
        }
