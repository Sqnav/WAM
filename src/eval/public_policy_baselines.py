from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np


def _step_seed(base_seed: int, scene_id: str, trajectory_name: str, step: int) -> int:
    payload = f"{int(base_seed)}:{scene_id}:{trajectory_name}:{int(step)}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little", signed=False)


@dataclass
class RandomActionPolicy:
    """Reproducible isotropic random velocity and yaw-rate action chunks."""

    seed: int
    horizon: int = 8

    def infer(
        self,
        *,
        scene_id: str,
        trajectory_name: str,
        step: int,
        **_: Any,
    ) -> np.ndarray:
        rng = np.random.default_rng(_step_seed(self.seed, scene_id, trajectory_name, step))
        directions = rng.normal(size=(self.horizon, 3)).astype(np.float32)
        directions /= np.maximum(np.linalg.norm(directions, axis=-1, keepdims=True), 1.0e-6)
        # Uniform radius gives every admissible speed equal probability without
        # over-representing cube corners that are later clipped to the speed ball.
        velocities = directions * rng.uniform(0.0, 1.0, size=(self.horizon, 1)).astype(np.float32)
        yaw_rate = rng.uniform(-1.0, 1.0, size=(self.horizon, 1)).astype(np.float32)
        return np.concatenate([velocities, yaw_rate], axis=-1)

    def reset(self) -> None:
        return None


class Pi05WebsocketPolicy:
    """Thin client for an official OpenPI policy server."""

    def __init__(self, host: str, port: int, horizon: int = 8) -> None:
        try:
            from openpi_client import image_tools
            from openpi_client import websocket_client_policy
        except ImportError as exc:  # pragma: no cover - depends on the separate OpenPI setup
            raise ImportError(
                "pi05 evaluation requires openpi-client. Run code/scripts/setup_openpi_uav.sh first."
            ) from exc

        self._image_tools = image_tools
        self._client = websocket_client_policy.WebsocketClientPolicy(host=host, port=port)
        self.horizon = int(horizon)

    def infer(
        self,
        *,
        rgb: np.ndarray,
        instruction: str,
        previous_action_physical: np.ndarray,
        **_: Any,
    ) -> np.ndarray:
        image = self._image_tools.convert_to_uint8(
            self._image_tools.resize_with_pad(np.asarray(rgb, dtype=np.uint8), 224, 224)
        )
        result = self._client.infer(
            {
                "observation/image": image,
                "observation/state": np.asarray(previous_action_physical, dtype=np.float32).reshape(4),
                "prompt": str(instruction),
            }
        )
        actions = np.asarray(result.get("actions"), dtype=np.float32)
        if actions.ndim != 2 or actions.shape[1] < 4:
            raise RuntimeError(f"pi05 server returned invalid actions shape: {actions.shape}")
        if actions.shape[0] < self.horizon:
            raise RuntimeError(
                f"pi05 server returned {actions.shape[0]} actions, expected at least {self.horizon}"
            )
        actions = actions[: self.horizon, :4]
        if not np.isfinite(actions).all():
            raise RuntimeError("pi05 server returned non-finite actions")
        return actions

    def reset(self) -> None:
        self._client.reset()


def make_public_policy(args: Any):
    if args.policy_backend == "random":
        return RandomActionPolicy(seed=int(args.seed), horizon=int(args.public_action_horizon))
    if args.policy_backend == "pi05":
        return Pi05WebsocketPolicy(
            host=str(args.pi05_policy_host),
            port=int(args.pi05_policy_port),
            horizon=int(args.public_action_horizon),
        )
    return None
