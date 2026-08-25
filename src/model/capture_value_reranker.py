from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def sample_grouped_candidate_noise(
    batch_size: int,
    candidate_count: int,
    horizon: int,
    action_dim: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Sample candidates while advancing the main RNG only for candidate zero.

    This keeps candidate zero bitwise comparable with single-candidate policy
    inference. Extra candidates use a forked RNG state and therefore cannot
    perturb the policy sample used at this or the next control step.
    """
    if batch_size < 1 or candidate_count < 1 or horizon < 1 or action_dim < 1:
        raise ValueError("Candidate noise dimensions must all be positive.")
    base = torch.randn(
        batch_size, 1, horizon, action_dim, device=device, dtype=dtype
    )
    if candidate_count == 1:
        return base.flatten(0, 1)
    fork_devices = [device.index or 0] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=fork_devices):
        extra = torch.randn(
            batch_size,
            candidate_count - 1,
            horizon,
            action_dim,
            device=device,
            dtype=dtype,
        )
    return torch.cat([base, extra], dim=1).flatten(0, 1)


def build_structured_recenter_candidates(
    candidates: torch.Tensor,
    current_box: torch.Tensor,
    *,
    recenter_gains: tuple[float, ...] = (0.35, 0.55, 0.75),
) -> torch.Tensor:
    """Replace extra a0 samples with physically feasible recenter proposals.

    Candidate zero remains untouched. The alternatives use increasing gains
    for image-plane lateral/vertical correction and spend the remaining unit
    translation budget on forward pursuit. Only a0 is changed because online
    control replans at every camera frame.
    """
    if candidates.ndim != 4 or candidates.size(-1) != 4:
        raise ValueError("candidates must be [B,N,H,4].")
    if current_box.shape != (candidates.size(0), 4):
        raise ValueError("current_box must be [B,4].")
    if candidates.size(1) < 2:
        return candidates
    output = candidates.clone()
    center_error = 2.0 * (current_box.float()[:, :2] - 0.5)
    proposal_count = min(candidates.size(1) - 1, len(recenter_gains))
    for proposal_index in range(proposal_count):
        gain = float(recenter_gains[proposal_index])
        lateral = (gain * center_error[:, 0]).clamp(-1.0, 1.0)
        vertical = (gain * center_error[:, 1]).clamp(-1.0, 1.0)
        forward = (
            1.0 - lateral.square() - vertical.square()
        ).clamp_min(0.0).sqrt()
        yaw_gain = 1.0 + 0.5 * proposal_index
        yaw = (yaw_gain * center_error[:, 0]).clamp(-1.0, 1.0)
        output[:, proposal_index + 1, 0] = torch.stack(
            [forward, lateral, vertical, yaw], dim=-1
        ).to(dtype=output.dtype)
    return output


def _normalized_actions_after_physical_clipping(
    actions: torch.Tensor,
    *,
    max_vel: float,
    max_yaw_rate: float,
    max_speed_norm: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return runtime-equivalent physical and normalized actions."""
    if actions.size(-1) != 4:
        raise ValueError("Geometric trajectory scoring requires [vx,vy,vz,yaw].")
    max_vel = max(float(max_vel), 1.0e-6)
    max_yaw_rate = max(float(max_yaw_rate), 1.0e-6)
    translation = actions.float()[..., :3].clamp(-1.0, 1.0) * max_vel
    cap = float(max_speed_norm)
    if cap > 0.0:
        norm = torch.linalg.vector_norm(translation, dim=-1, keepdim=True)
        translation = translation * torch.where(
            norm > cap,
            cap / norm.clamp_min(1.0e-12),
            torch.ones_like(norm),
        )
    yaw_rate = actions.float()[..., 3:4].clamp(-1.0, 1.0) * max_yaw_rate
    physical = torch.cat([translation, yaw_rate], dim=-1)
    normalized = torch.cat(
        [translation / max_vel, yaw_rate / max_yaw_rate], dim=-1
    )
    return physical, normalized


def score_geometric_capture_trajectories(
    candidates: torch.Tensor,
    current_box: torch.Tensor,
    previous_action: Optional[torch.Tensor],
    target_box_history: Optional[torch.Tensor] = None,
    target_box_history_valid: Optional[torch.Tensor] = None,
    *,
    max_vel: float,
    max_yaw_rate: float,
    max_speed_norm: float,
    control_dt: float = 1.0,
    horizontal_fov_deg: float = 90.0,
    vertical_fov_deg: float = 90.0,
    depth_scale: float = 0.2698,
    min_depth: float = 1.0,
    max_depth: float = 20.0,
    target_box_size: float = 0.06094,
    box_size_sigma: float = 0.01,
    discount: float = 0.8,
    recenter_sigma: float = 0.35,
    pursuit_center_sigma: float = 0.40,
    out_of_frame_weight: float = 2.0,
    first_action_smooth_weight: float = 2.0,
    temporal_smooth_weight: float = 1.0,
    recenter_weight: float = 2.0,
    pursuit_weight: float = 0.7,
    smooth_weight: float = 0.1,
    consensus_weight: float = 0.1,
    short_horizon: int = 1,
) -> Dict[str, torch.Tensor]:
    """Score action samples without pretending that the moving target is static.

    Only the first few controls affect recentering and pursuit because online
    control executes ``a0`` and replans. Candidate zero is the unmodified policy
    sample and supplies the learned forward-speed reference. The remaining tail
    is used only to reject temporally noisy samples.
    """
    if candidates.ndim != 4 or candidates.size(-1) != 4:
        raise ValueError("Candidates must have shape [B,N,H,4].")
    batch_size, candidate_count, horizon, _ = candidates.shape
    if current_box.shape != (batch_size, 4):
        raise ValueError("Current box must have shape [B,4] normalized cxcywh.")
    if horizon < 1:
        raise ValueError("At least one candidate action is required.")
    if float(control_dt) <= 0.0:
        raise ValueError("control_dt must be positive.")

    device = candidates.device
    candidates_f = candidates.float()
    box = current_box.to(device=device, dtype=torch.float32).clamp(0.0, 1.0)
    physical, normalized = _normalized_actions_after_physical_clipping(
        candidates_f,
        max_vel=max_vel,
        max_yaw_rate=max_yaw_rate,
        max_speed_norm=max_speed_norm,
    )

    half_fov_x = math.radians(float(horizontal_fov_deg)) * 0.5
    half_fov_y = math.radians(float(vertical_fov_deg)) * 0.5
    if not (0.0 < half_fov_x < math.pi * 0.5):
        raise ValueError("horizontal_fov_deg must be between 0 and 180.")
    if not (0.0 < half_fov_y < math.pi * 0.5):
        raise ValueError("vertical_fov_deg must be between 0 and 180.")

    image_error_x = (box[:, 0] - 0.5) / 0.5
    box_size = torch.sqrt((box[:, 2] * box[:, 3]).clamp_min(1.0e-8))
    center_velocity_x = torch.zeros_like(image_error_x)
    if target_box_history is not None:
        history = target_box_history.to(device=device, dtype=torch.float32)
        if history.ndim != 3 or history.size(0) != batch_size or history.size(-1) < 4:
            raise ValueError("Target box history must have shape [B,K,>=4].")
        if target_box_history_valid is None or target_box_history_valid.shape != history.shape[:2]:
            raise ValueError("Target box history validity shape is invalid.")
        valid = target_box_history_valid.to(device=device, dtype=torch.bool)
        if history.size(1) > 0:
            positions = torch.arange(history.size(1), device=device)[None].expand_as(valid)
            last_index = positions.masked_fill(~valid, -1).max(dim=1).values
            has_history = last_index >= 0
            safe_index = last_index.clamp_min(0)
            previous_box = history[
                torch.arange(batch_size, device=device), safe_index, :4
            ].clamp(0.0, 1.0)
            previous_error_x = (previous_box[:, 0] - 0.5) / 0.5
            center_velocity_x = torch.where(
                has_history, image_error_x - previous_error_x, center_velocity_x
            )

    active_horizon = min(max(int(short_horizon), 1), horizon)
    # Calibrated on the frozen Tracker outputs and expert actions from
    # City_1/2/3 trajectories 1-450 (82,402 frames). Online expert yaw has a
    # much steeper instantaneous correction, but imposing that on a policy
    # trained with the dataset labels caused catastrophic sample switching.
    desired_lateral = (
        0.88010705 * image_error_x - 0.43097605 * center_velocity_x
    )[:, None, None].expand(
        -1, candidate_count, active_horizon
    )
    desired_yaw = (
        0.12978940 * image_error_x - 1.74331236 * center_velocity_x
    )[:, None, None].expand(
        -1, candidate_count, active_horizon
    )
    lateral_error = normalized[:, :, :active_horizon, 1] - desired_lateral
    yaw_error = normalized[:, :, :active_horizon, 3] - desired_yaw
    active_center_error = torch.stack(
        [yaw_error, lateral_error], dim=-1
    )
    center_error = active_center_error[:, :, -1:, :].expand(
        -1, -1, horizon, -1
    ).clone()
    center_error[:, :, :active_horizon] = active_center_error
    future_box_size = box_size[:, None, None].expand(
        -1, candidate_count, horizon
    )
    gamma = float(discount)
    if not (0.0 < gamma <= 1.0):
        raise ValueError("discount must be in (0,1].")
    weights = torch.pow(
        torch.full((active_horizon,), gamma, device=device, dtype=torch.float32),
        torch.arange(active_horizon, device=device, dtype=torch.float32),
    )
    weight_sum = weights.sum().clamp_min(1.0e-6)

    center_huber = (
        F.smooth_l1_loss(
            lateral_error / max(float(recenter_sigma), 1.0e-6),
            torch.zeros_like(lateral_error),
            reduction="none",
        )
        + F.smooth_l1_loss(
            yaw_error / max(float(recenter_sigma), 1.0e-6),
            torch.zeros_like(yaw_error),
            reduction="none",
        )
    )
    recenter_cost = (center_huber * weights).sum(dim=2) / weight_sum
    recenter_cost = (recenter_cost / (1.0 + recenter_cost)).clamp(0.0, 1.0)

    # Preserve the learned pursuit behavior. A current box has no target-speed
    # information, so it must not invent an absolute vx target from bbox size.
    base_forward = normalized[:, :1, :active_horizon, 0]
    slower_than_base = F.relu(
        base_forward - normalized[:, :, :active_horizon, 0]
    )
    pursuit_scale = max(float(pursuit_center_sigma), 1.0e-3)
    pursuit_cost = (
        (slower_than_base / pursuit_scale).square().clamp(max=1.0) * weights
    ).sum(dim=2) / weight_sum

    if previous_action is None:
        previous = torch.zeros(
            batch_size, 4, device=device, dtype=torch.float32
        )
    else:
        if previous_action.shape != (batch_size, 4):
            raise ValueError("Previous action must have shape [B,4].")
        _, previous = _normalized_actions_after_physical_clipping(
            previous_action.to(device=device, dtype=torch.float32),
            max_vel=max_vel,
            max_yaw_rate=max_yaw_rate,
            max_speed_norm=max_speed_norm,
        )
    first_change = (
        normalized[:, :, 0] - previous[:, None]
    ).square().mean(dim=-1)
    if horizon > 1:
        transition = (normalized[:, :, 1:] - normalized[:, :, :-1]).square().mean(dim=-1)
        temporal = transition.mean(dim=2)
    else:
        temporal = torch.zeros_like(first_change)
    smooth_normalizer = max(
        float(first_action_smooth_weight) + float(temporal_smooth_weight),
        1.0e-6,
    )
    smooth_cost = (
        float(first_action_smooth_weight) * first_change
        + float(temporal_smooth_weight) * temporal
    ) / smooth_normalizer
    smooth_cost = (smooth_cost / 0.05).clamp(0.0, 1.0)

    # Prefer a representative diffusion sample instead of an outlier. Median
    # consensus is robust when one of four samples is poor.
    consensus_steps = normalized[:, :, :active_horizon]
    consensus_center = consensus_steps.median(dim=1, keepdim=True).values
    consensus_delta = consensus_steps - consensus_center
    consensus_scale = torch.tensor(
        [0.20, 0.20, 0.20, 0.25], device=device, dtype=torch.float32
    )
    consensus_cost = (
        (consensus_delta / consensus_scale).square().mean(dim=(2, 3))
    ).clamp(0.0, 1.0)

    total_weight = max(
        float(recenter_weight)
        + float(pursuit_weight)
        + float(smooth_weight)
        + float(consensus_weight),
        1.0e-6,
    )
    score = -(
        float(recenter_weight) * recenter_cost
        + float(pursuit_weight) * pursuit_cost
        + float(smooth_weight) * smooth_cost
        + float(consensus_weight) * consensus_cost
    ) / total_weight
    return {
        "score": score,
        "recenter_cost": recenter_cost,
        "pursuit_cost": pursuit_cost,
        "smooth_cost": smooth_cost,
        "consensus_cost": consensus_cost,
        "predicted_center_error": center_error,
        "predicted_box_size": future_box_size,
        "physical_candidates": physical,
        "observed_center_velocity": center_velocity_x,
    }


def select_geometric_candidate(
    scores: torch.Tensor,
    selection_margin: float,
    allow_switch: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """Keep candidate zero unless another candidate wins by a real margin."""
    if scores.ndim != 2 or scores.size(1) < 1:
        raise ValueError("Candidate scores must have shape [B,N].")
    raw_index = scores.argmax(dim=1)
    raw_score = scores.gather(1, raw_index[:, None]).squeeze(1)
    advantage = raw_score - scores[:, 0]
    if allow_switch is None:
        allow_switch = torch.ones_like(raw_index, dtype=torch.bool)
    elif allow_switch.shape != raw_index.shape:
        raise ValueError("allow_switch must have shape [B].")
    else:
        allow_switch = allow_switch.to(device=scores.device, dtype=torch.bool)
    used_fallback = (raw_index != 0) & (
        (advantage < float(selection_margin)) | ~allow_switch
    )
    selected_index = torch.where(
        used_fallback, torch.zeros_like(raw_index), raw_index
    )
    return {
        "selected_index": selected_index,
        "raw_selected_index": raw_index,
        "score_advantage": advantage,
        "used_fallback": used_fallback,
    }


def build_capture_value_candidates(
    predicted_action: torch.Tensor,
    expert_action: torch.Tensor,
    *,
    candidate_count: int,
    noise_std: float,
) -> torch.Tensor:
    """Build policy-like positive and hard-negative action sequences."""
    if predicted_action.shape != expert_action.shape or predicted_action.ndim != 3:
        raise ValueError("Predicted and expert actions must match [B,H,A].")
    candidate_count = int(candidate_count)
    if candidate_count < 2:
        raise ValueError("Capture-value training requires at least two candidates.")

    predicted = predicted_action.detach().float().clamp(-1.0, 1.0)
    expert = expert_action.detach().float().clamp(-1.0, 1.0)
    candidates = [predicted, expert]
    if candidate_count >= 3:
        sign_error = expert.clone()
        if sign_error.size(-1) >= 2:
            sign_error[..., 1] = -sign_error[..., 1]
        if sign_error.size(-1) >= 4:
            sign_error[..., 3] = -sign_error[..., 3]
        candidates.append(sign_error)
    if candidate_count >= 4:
        mismatched = expert.roll(1, dims=0) if expert.size(0) > 1 else expert.flip(1)
        candidates.append(mismatched)
    while len(candidates) < candidate_count:
        scale = max(float(noise_std), 0.0) * (1.0 + 0.25 * len(candidates))
        candidates.append(expert + torch.randn_like(expert) * scale)

    stacked = torch.stack(candidates[:candidate_count], dim=1)
    if float(noise_std) > 0.0:
        # Keep the exact expert candidate stable and broaden the remaining
        # candidate distribution around policy-like trajectories.
        noise = torch.randn_like(stacked) * float(noise_std)
        noise[:, 1] = 0.0
        stacked = stacked + noise
    return stacked.clamp(-1.0, 1.0)


def approximate_capture_outcomes(
    candidates: torch.Tensor,
    expert_action: torch.Tensor,
    target_relative: torch.Tensor,
    valid_mask: Optional[torch.Tensor],
    *,
    max_vel: float,
    max_yaw_rate: float,
    capture_distance: float,
    camera_half_fov_deg: float = 45.0,
) -> Dict[str, torch.Tensor]:
    """Approximate counterfactual outcomes around the recorded expert rollout.

    The dataset supplies future target positions along the expert trajectory.
    Candidate translation and yaw deltas are applied relative to that trajectory,
    producing training-only capture, distance, and visibility targets.
    """
    if candidates.ndim != 4:
        raise ValueError("Candidates must have shape [B,N,H,A].")
    batch_size, candidate_count, horizon, action_dim = candidates.shape
    if expert_action.shape != (batch_size, horizon, action_dim):
        raise ValueError("Expert actions must match candidate [B,H,A].")
    if target_relative.ndim != 3 or target_relative.size(0) != batch_size:
        raise ValueError("Target-relative states must have shape [B,T,3].")
    if target_relative.size(1) < horizon + 1 or target_relative.size(-1) < 3:
        raise ValueError("Capture-value targets require H+1 future relative states.")

    if valid_mask is None:
        valid = torch.ones(
            batch_size, horizon, device=candidates.device, dtype=torch.bool
        )
    else:
        if valid_mask.shape != (batch_size, horizon):
            raise ValueError("Capture-value validity must have shape [B,H].")
        valid = valid_mask.to(device=candidates.device, dtype=torch.bool)

    candidate = candidates.float()
    expert = expert_action[:, None].float()
    translation_delta = (
        expert[..., :3] - candidate[..., :3]
    ) * float(max_vel)
    translation_delta = translation_delta.cumsum(dim=2)
    relative = target_relative[:, None, 1 : horizon + 1, :3].float()
    relative = relative + translation_delta

    if action_dim >= 4:
        yaw_delta = (
            candidate[..., 3] - expert[..., 3]
        ).cumsum(dim=2)
        yaw_delta = yaw_delta * (float(max_yaw_rate) * math.pi / 180.0)
        cos_yaw = yaw_delta.cos()
        sin_yaw = yaw_delta.sin()
        x, y, z = relative.unbind(dim=-1)
        relative = torch.stack(
            [cos_yaw * x + sin_yaw * y, -sin_yaw * x + cos_yaw * y, z],
            dim=-1,
        )

    distance = relative.norm(dim=-1)
    valid_expanded = valid[:, None].expand(-1, candidate_count, -1)
    valid_count = valid.sum(dim=1).clamp_min(1)
    final_index = (valid_count - 1)[:, None, None].expand(
        -1, candidate_count, 1
    )
    final_distance = distance.gather(2, final_index).squeeze(-1)

    x, y, z = relative.unbind(dim=-1)
    horizontal = torch.sqrt(x.square() + y.square()).clamp_min(1.0e-6)
    half_fov = float(camera_half_fov_deg) * math.pi / 180.0
    visible = (
        (x > 0.0)
        & (torch.atan2(y.abs(), x.clamp_min(1.0e-6)) <= half_fov)
        & (torch.atan2(z.abs(), horizontal) <= half_fov)
        & valid_expanded
    )
    visibility = visible.float().sum(dim=2) / valid_count[:, None].float()

    distance_scale = max(float(capture_distance), 1.0e-6)
    capture_probability = torch.sigmoid(
        (distance_scale - final_distance) / max(0.15 * distance_scale, 1.0e-6)
    )
    return {
        "capture_probability": capture_probability.clamp(0.0, 1.0),
        "final_distance_normalized": final_distance / distance_scale,
        "visibility_probability": visibility.clamp(0.0, 1.0),
    }


class CaptureValueHead(nn.Module):
    """Score candidate action sequences from cached video and target context."""

    def __init__(
        self,
        *,
        video_context_dim: int,
        target_context_dim: int,
        action_dim: int,
        horizon: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_heads: int = 8,
        distance_score_weight: float = 1.0,
        visibility_score_weight: float = 0.25,
    ) -> None:
        super().__init__()
        self.action_dim = int(action_dim)
        self.horizon = int(horizon)
        self.hidden_dim = int(hidden_dim)
        self.distance_score_weight = float(distance_score_weight)
        self.visibility_score_weight = float(visibility_score_weight)
        if self.horizon <= 0 or self.hidden_dim <= 0:
            raise ValueError("Capture-value dimensions must be positive.")
        if self.hidden_dim % int(num_heads) != 0:
            raise ValueError("Capture-value hidden_dim must be divisible by num_heads.")

        self.video_encoder = nn.Sequential(
            nn.LayerNorm(int(video_context_dim)),
            nn.Linear(int(video_context_dim), self.hidden_dim),
        )
        self.target_encoder = nn.Sequential(
            nn.LayerNorm(int(target_context_dim)),
            nn.Linear(int(target_context_dim), self.hidden_dim),
        )
        self.action_encoder = nn.Linear(self.action_dim, self.hidden_dim)
        self.horizon_embedding = nn.Parameter(
            torch.zeros(1, 1, self.horizon, self.hidden_dim)
        )
        layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=int(num_heads),
            dim_feedforward=self.hidden_dim * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.sequence_encoder = nn.TransformerEncoder(
            layer, num_layers=int(num_layers), norm=nn.LayerNorm(self.hidden_dim)
        )
        self.output = nn.Linear(self.hidden_dim, 3)
        nn.init.normal_(self.horizon_embedding, std=0.02)

    def forward(
        self,
        candidates: torch.Tensor,
        *,
        video_context: torch.Tensor,
        target_context: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if candidates.ndim != 4:
            raise ValueError("Capture-value candidates must be [B,N,H,A].")
        batch_size, candidate_count, horizon, action_dim = candidates.shape
        if (horizon, action_dim) != (self.horizon, self.action_dim):
            raise ValueError("Capture-value candidate horizon/action dimensions mismatch.")
        if video_context.size(0) != batch_size or target_context.size(0) != batch_size:
            raise ValueError("Capture-value context batch must match candidates.")

        parameter = self.action_encoder.weight
        candidates = candidates.to(device=parameter.device, dtype=parameter.dtype)
        video_context = video_context.to(device=parameter.device, dtype=parameter.dtype)
        target_context = target_context.to(device=parameter.device, dtype=parameter.dtype)
        context = self.video_encoder(video_context) + self.target_encoder(target_context)
        tokens = self.action_encoder(candidates)
        tokens = tokens + self.horizon_embedding.to(tokens)
        tokens = tokens + context[:, None, None]
        tokens = tokens.flatten(0, 1)

        padding_mask = None
        mask_float = None
        if valid_mask is not None:
            if valid_mask.shape != (batch_size, horizon):
                raise ValueError("Capture-value valid_mask must be [B,H].")
            valid = valid_mask.to(device=tokens.device, dtype=torch.bool)
            attention_valid = valid.clone()
            attention_valid[~attention_valid.any(dim=1), 0] = True
            padding_mask = (~attention_valid)[:, None].expand(
                -1, candidate_count, -1
            )
            padding_mask = padding_mask.flatten(0, 1)
            mask_float = valid[:, None, :, None].to(tokens.dtype)
        encoded = self.sequence_encoder(tokens, src_key_padding_mask=padding_mask)
        encoded = encoded.view(batch_size, candidate_count, horizon, self.hidden_dim)
        if mask_float is None:
            pooled = encoded.mean(dim=2)
        else:
            pooled = (encoded * mask_float).sum(dim=2) / mask_float.sum(dim=2).clamp_min(1.0)
        raw = self.output(pooled).float()
        capture_logit = raw[..., 0]
        distance_normalized = F.softplus(raw[..., 1])
        visibility_logit = raw[..., 2]
        score = (
            capture_logit
            - self.distance_score_weight * distance_normalized
            + self.visibility_score_weight * visibility_logit
        )
        return {
            "score": score,
            "capture_logit": capture_logit,
            "final_distance_normalized": distance_normalized,
            "visibility_logit": visibility_logit,
        }


def capture_value_loss(
    prediction: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor],
    *,
    distance_score_weight: float = 1.0,
    visibility_score_weight: float = 0.25,
) -> Dict[str, torch.Tensor]:
    capture = F.binary_cross_entropy_with_logits(
        prediction["capture_logit"], targets["capture_probability"]
    )
    distance = F.smooth_l1_loss(
        prediction["final_distance_normalized"],
        targets["final_distance_normalized"],
    )
    visibility = F.binary_cross_entropy_with_logits(
        prediction["visibility_logit"], targets["visibility_probability"]
    )
    target_score = (
        torch.logit(targets["capture_probability"].clamp(1.0e-4, 1.0 - 1.0e-4))
        - float(distance_score_weight) * targets["final_distance_normalized"]
        + float(visibility_score_weight)
        * torch.logit(
            targets["visibility_probability"].clamp(1.0e-4, 1.0 - 1.0e-4)
        )
    )
    target_index = target_score.argmax(dim=1)
    ranking = F.cross_entropy(prediction["score"], target_index)
    total = capture + distance + visibility + ranking
    accuracy = (prediction["score"].argmax(dim=1) == target_index).float().mean()
    return {
        "loss": total,
        "capture": capture,
        "distance": distance,
        "visibility": visibility,
        "ranking": ranking,
        "accuracy": accuracy,
        "target_capture": targets["capture_probability"].mean(),
    }
