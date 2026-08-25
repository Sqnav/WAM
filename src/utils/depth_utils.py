from __future__ import annotations

from typing import Any

import numpy as np


def depth_array_to_inverse_depth(
    depth: Any,
    *,
    scale_m_to_uint16: float = 100.0,
    max_depth_m: float = 655.35,
    eps_m: float = 1.0,
) -> np.ndarray:
    """Convert depth values to normalized inverse depth in [0, 1].

    The near end is saturated at eps_m. Using 1m by default avoids compressing
    common 10-100m scenes into almost-zero values.
    """
    arr = np.asarray(depth, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[..., 0]
    max_depth = max(float(max_depth_m), float(eps_m))
    scale = max(float(scale_m_to_uint16), 1.0e-6)
    finite = np.nan_to_num(arr, nan=max_depth, posinf=max_depth, neginf=max_depth)
    if finite.size and float(np.nanmax(finite)) > max_depth + 1.0:
        depth_m = finite / scale
    else:
        depth_m = finite
    depth_m = np.clip(depth_m, float(eps_m), max_depth)
    inv = 1.0 / depth_m
    inv_max = 1.0 / float(eps_m)
    inv_min = 1.0 / max_depth
    inv = (inv - inv_min) / max(inv_max - inv_min, 1.0e-8)
    return np.clip(inv, 0.0, 1.0).astype(np.float32)


def depth_tensor_to_inverse_depth(
    depth,
    *,
    scale_m_to_uint16: float = 100.0,
    max_depth_m: float = 655.35,
    eps_m: float = 1.0,
):
    import torch

    arr = depth.detach().float()
    if arr.ndim >= 3 and arr.shape[-1] in {1, 3}:
        arr = arr[..., 0]
    max_depth = max(float(max_depth_m), float(eps_m))
    scale = max(float(scale_m_to_uint16), 1.0e-6)
    arr = torch.nan_to_num(arr, nan=max_depth, posinf=max_depth, neginf=max_depth)
    if arr.numel() > 0 and bool((arr.detach().amax() > max_depth + 1.0).cpu().item()):
        depth_m = arr / scale
    else:
        depth_m = arr
    depth_m = depth_m.clamp(min=float(eps_m), max=max_depth)
    inv = 1.0 / depth_m
    inv_max = 1.0 / float(eps_m)
    inv_min = 1.0 / max_depth
    inv = (inv - inv_min) / max(inv_max - inv_min, 1.0e-8)
    return inv.clamp(0.0, 1.0)
