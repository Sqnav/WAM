from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset


MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def load_manifest(path: Path, split: str) -> List[Dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get(split)
    if not isinstance(records, list) or not records:
        raise ValueError(f"Manifest {path} has no non-empty {split!r} split")
    return records


def _image_tensor(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    return torch.from_numpy(array).permute(2, 0, 1).float().div_(255.0)


def crop_target(
    image: torch.Tensor,
    box: Sequence[float],
    factor: float,
    output_size: int,
    center_jitter: float = 0.0,
    scale_jitter: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    x, y, w, h = map(float, box)
    base = math.sqrt(max(w * h, 1.0)) * factor
    scale = math.exp(random.gauss(0.0, scale_jitter))
    crop_size = max(base * scale, 2.0)
    cx = x + 0.5 * w + random.gauss(0.0, center_jitter * crop_size)
    cy = y + 0.5 * h + random.gauss(0.0, center_jitter * crop_size)
    height, width = image.shape[-2:]
    x1, y1 = int(math.floor(cx - crop_size / 2)), int(math.floor(cy - crop_size / 2))
    side = max(int(math.ceil(crop_size)), 2)
    left, top = max(-x1, 0), max(-y1, 0)
    right, bottom = max(x1 + side - width, 0), max(y1 + side - height, 0)
    padded = F.pad(image, (left, right, top, bottom), value=0.0)
    crop = padded[:, y1 + top : y1 + top + side, x1 + left : x1 + left + side]
    crop = F.interpolate(crop.unsqueeze(0), (output_size, output_size), mode="bilinear", align_corners=False)[0]
    normalized = torch.tensor(
        [(x + 0.5 * w - x1) / side, (y + 0.5 * h - y1) / side, w / side, h / side],
        dtype=torch.float32,
    ).clamp(0.0, 1.0)
    return (crop - MEAN) / STD, normalized


def square_box(box: Sequence[float]) -> List[float]:
    x, y, width, height = map(float, box)
    side = max(width, height)
    return [x + 0.5 * (width - side), y + 0.5 * (height - side), side, side]


class UAVTrackingDataset(Dataset):
    def __init__(
        self,
        manifest: Path,
        split: str,
        samples_per_epoch: int,
        max_gap: int = 40,
        template_size: int = 128,
        search_size: int = 256,
        square_boxes: bool = False,
    ) -> None:
        self.records = load_manifest(manifest, split)
        self.samples_per_epoch = int(samples_per_epoch)
        self.max_gap = int(max_gap)
        self.template_size = int(template_size)
        self.search_size = int(search_size)
        self.training = split == "train"
        self.square_boxes = bool(square_boxes)

    def __len__(self) -> int:
        return self.samples_per_epoch

    def __getitem__(self, _: int) -> Dict[str, torch.Tensor]:
        for _attempt in range(20):
            record = random.choice(self.records)
            boxes = record["boxes_xywh"]
            valid = [i for i, box in enumerate(boxes) if box is not None and box[2] > 1 and box[3] > 1]
            if len(valid) < 2:
                continue
            search_idx = random.choice(valid[1:])
            candidates = [i for i in valid if 0 <= search_idx - i <= self.max_gap and i < search_idx]
            if not candidates:
                continue
            template_idx = random.choice(candidates)
            root = Path(record["trajectory"])
            template_box = boxes[template_idx]
            search_box = boxes[search_idx]
            if self.square_boxes:
                template_box = square_box(template_box)
                search_box = square_box(search_box)
            template = _image_tensor(root / record["frames"][template_idx])
            search = _image_tensor(root / record["frames"][search_idx])
            template, _ = crop_target(template, template_box, 2.0, self.template_size)
            search, target = crop_target(
                search,
                search_box,
                4.0,
                self.search_size,
                center_jitter=0.15 if self.training else 0.0,
                scale_jitter=0.15 if self.training else 0.0,
            )
            if self.training and random.random() < 0.5:
                template = template.flip(-1)
                search = search.flip(-1)
                target[0] = 1.0 - target[0]
            return {"template": template, "search": search, "box": target}
        raise RuntimeError("Could not sample a valid template/search pair")
