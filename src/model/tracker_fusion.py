from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class ActionTrackerCrossAttention(nn.Module):
    """Independent action-to-tracker attention with a zero-initialized gate."""

    def __init__(
        self, action_dim: int, num_heads: int, head_dim: int, gate_init: float = 0.0
    ) -> None:
        super().__init__()
        attention_dim = int(num_heads) * int(head_dim)
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        self.query_norm = nn.LayerNorm(action_dim)
        self.query = nn.Linear(action_dim, attention_dim)
        self.key = nn.Linear(action_dim, attention_dim)
        self.value = nn.Linear(action_dim, attention_dim)
        self.output = nn.Linear(attention_dim, action_dim)
        self.gate = nn.Parameter(torch.tensor(float(gate_init)))

    def forward(
        self,
        action_hidden: torch.Tensor,
        condition: torch.Tensor,
        *,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        batch, action_len, _ = action_hidden.shape
        tracker_len = condition.size(1)
        q = self.query(self.query_norm(action_hidden)).view(
            batch, action_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        k = self.key(condition).view(
            batch, tracker_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        v = self.value(condition).view(
            batch, tracker_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        if return_attention:
            scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) / (self.head_dim ** 0.5)
            attention = torch.softmax(scores, dim=-1).to(dtype=q.dtype)
            delta = torch.matmul(attention, v)
        else:
            attention = None
            delta = F.scaled_dot_product_attention(q, k, v)
        delta = delta.transpose(1, 2).reshape(batch, action_len, -1)
        delta = torch.tanh(self.gate) * self.output(delta)
        if return_attention:
            assert attention is not None
            return delta, attention
        return delta


class _FutureReadoutLayer(nn.Module):
    def __init__(self, action_dim: int) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(action_dim)
        self.query = nn.Linear(action_dim, action_dim)
        self.key = nn.Linear(action_dim, action_dim)
        self.value = nn.Linear(action_dim, action_dim)
        self.output = nn.Linear(action_dim, action_dim)
        self.gate = nn.Parameter(torch.tensor(0.0))


class BoxFreeFutureTargetReadout(nn.Module):
    """Template-guided state transitions aligned to a short action sequence."""

    def __init__(
        self,
        tracker_dim: int,
        action_dim: int,
        video_dim: int,
        num_layers: int,
        start_layer: int,
        action_horizon: int,
    ) -> None:
        super().__init__()
        self.start_layer = int(start_layer)
        self.action_horizon = int(action_horizon)
        self.video_projection = nn.Sequential(
            nn.LayerNorm(video_dim), nn.Linear(video_dim, action_dim), nn.GELU(), nn.LayerNorm(action_dim)
        )
        self.template_projection = nn.Sequential(
            nn.LayerNorm(tracker_dim), nn.Linear(tracker_dim, action_dim), nn.GELU(), nn.LayerNorm(action_dim)
        )
        self.target_query = nn.Parameter(torch.zeros(1, 1, action_dim))
        self.template_pool = nn.MultiheadAttention(action_dim, num_heads=8, batch_first=True)
        self.target_search = nn.MultiheadAttention(action_dim, num_heads=8, batch_first=True)
        self.horizon_embedding = nn.Parameter(torch.zeros(1, self.action_horizon, action_dim))
        self.timestep_projection = nn.Sequential(nn.Linear(1, action_dim), nn.SiLU(), nn.Linear(action_dim, action_dim))
        self.query_mlp = nn.Sequential(
            nn.LayerNorm(action_dim * 4), nn.Linear(action_dim * 4, action_dim), nn.GELU(), nn.Linear(action_dim, action_dim)
        )
        self.dynamic_gate = nn.Sequential(
            nn.LayerNorm(action_dim * 3), nn.Linear(action_dim * 3, action_dim), nn.SiLU(), nn.Linear(action_dim, 1)
        )
        size_hidden = max(action_dim // 4, 1)
        self.current_size_head = nn.Sequential(
            nn.LayerNorm(action_dim),
            nn.Linear(action_dim, size_hidden),
            nn.GELU(),
            nn.Linear(size_hidden, 2),
        )
        self.box_offset_head = nn.Sequential(
            nn.LayerNorm(action_dim),
            nn.Linear(action_dim, size_hidden),
            nn.GELU(),
            nn.Linear(size_hidden, 4),
        )
        self.layers = nn.ModuleDict({
            str(index): _FutureReadoutLayer(action_dim)
            for index in range(self.start_layer, int(num_layers))
        })
        nn.init.trunc_normal_(self.target_query, std=0.02)
        nn.init.trunc_normal_(self.horizon_embedding, std=0.02)
        nn.init.zeros_(self.box_offset_head[-1].weight)
        nn.init.zeros_(self.box_offset_head[-1].bias)
        nn.init.constant_(self.current_size_head[-1].bias, -3.0)

    def make_target_state(
        self,
        template_features: torch.Tensor,
        spatial_tokens: torch.Tensor,
        full_xy: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if template_features.ndim != 3 or spatial_tokens.ndim != 3:
            raise ValueError("Template features and spatial Tracker tokens must be rank-3.")
        if template_features.size(0) != spatial_tokens.size(0):
            raise ValueError("Template and Search batch sizes must match.")
        template_tokens = self.template_projection(template_features)
        query = self.target_query.expand(template_tokens.size(0), -1, -1)
        template_identity, _ = self.template_pool(query, template_tokens, template_tokens, need_weights=False)
        current_token, weights = self.target_search(
            template_identity, spatial_tokens, spatial_tokens,
            need_weights=True, average_attn_weights=True,
        )
        # [B, 1, N]; it is an internal content-derived reference, never a box label.
        attention = weights.clamp_min(1.0e-8)
        soft_center = torch.bmm(attention, full_xy).squeeze(1)
        current_size = torch.sigmoid(self.current_size_head(current_token.squeeze(1))).clamp_min(1.0e-4)
        return {
            "current_token": current_token,
            "attention": attention,
            "soft_center": soft_center,
            "current_size": current_size,
        }

    def _future_queries(
        self,
        action_hidden: torch.Tensor,
        target_state: dict[str, torch.Tensor],
        action_timestep: torch.Tensor,
    ) -> torch.Tensor:
        """Build one horizon-specific target query for every Action token."""
        if action_hidden.size(1) != self.action_horizon:
            raise ValueError(
                f"Expected {self.action_horizon} Action queries, got {action_hidden.size(1)}."
            )
        batch = action_hidden.size(0)
        current = target_state["current_token"].expand(-1, self.action_horizon, -1)
        horizon = self.horizon_embedding.expand(batch, -1, -1)
        # DeepSpeed converts this module to the model compute dtype (bf16 in
        # training). Keep the scalar timestep aligned with its projection
        # weights instead of forcing float32.
        timestep_input = action_timestep.reshape(batch, 1).to(
            device=action_hidden.device, dtype=action_hidden.dtype
        )
        timestep = self.timestep_projection(timestep_input).unsqueeze(1)
        return self.query_mlp(
            torch.cat(
                [
                    action_hidden,
                    current,
                    horizon,
                    timestep.expand(-1, self.action_horizon, -1),
                ],
                dim=-1,
            )
        )

    def future_tokens(
        self,
        layer_index: int,
        action_hidden: torch.Tensor,
        video_hidden: torch.Tensor,
        target_state: dict[str, torch.Tensor],
        action_timestep: torch.Tensor,
    ) -> torch.Tensor:
        """Read horizon-aligned future target tokens from Video Expert memory."""
        if layer_index < self.start_layer:
            return torch.zeros_like(action_hidden)
        layer = self.layers[str(layer_index)]
        queries = self._future_queries(action_hidden, target_state, action_timestep)
        q = layer.query(layer.query_norm(queries))
        video_memory = self.video_projection(video_hidden)
        k = layer.key(video_memory)
        v = layer.value(video_memory)
        scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) / (q.size(-1) ** 0.5)
        return torch.matmul(torch.softmax(scores, dim=-1).to(dtype=v.dtype), v)

    def box_offsets(self, future_target_tokens: torch.Tensor) -> torch.Tensor:
        """Predict direct horizon offsets [dCX,dCY,dLogW,dLogH] from s0."""
        if future_target_tokens.ndim != 3 or future_target_tokens.size(1) != self.action_horizon:
            raise ValueError(
                f"future_target_tokens must be [B,{self.action_horizon},D], got {tuple(future_target_tokens.shape)}."
            )
        return self.box_offset_head(future_target_tokens)

    def action_aligned_state_tokens(
        self,
        future_target_tokens: torch.Tensor,
        target_state: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Return pre-action states s0..s(H-1) for action tokens a0..a(H-1)."""
        if future_target_tokens.ndim != 3 or future_target_tokens.size(1) != self.action_horizon:
            raise ValueError(
                f"future_target_tokens must be [B,{self.action_horizon},D], "
                f"got {tuple(future_target_tokens.shape)}."
            )
        current = target_state["current_token"]
        if current.shape != future_target_tokens[:, :1].shape:
            raise ValueError(
                "current target token must match one future target token; "
                f"got current={tuple(current.shape)} future={tuple(future_target_tokens.shape)}."
            )
        # Transition token i predicts s(i+1). Action i must instead consume its
        # pre-action state si, so prepend s0 and leave terminal sH unfused.
        return torch.cat([current, future_target_tokens[:, :-1]], dim=1)

    def state_boxes(
        self,
        future_target_tokens: torch.Tensor,
        target_state: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return s0..sH boxes, adjacent center flows, and direct s0 offsets."""
        offsets = self.box_offsets(future_target_tokens)
        current_center = target_state["soft_center"].unsqueeze(1)
        current_size_value = target_state.get("current_size")
        if current_size_value is None:
            raise RuntimeError("Future Target box prediction requires current_size.")
        current_size = current_size_value.unsqueeze(1).clamp(1.0e-4, 1.0)
        future_centers = current_center + offsets[..., :2]
        future_sizes = torch.exp(
            current_size.log() + offsets[..., 2:].clamp(-6.0, 6.0)
        ).clamp(1.0e-4, 1.0)
        state_centers = torch.cat([current_center, future_centers], dim=1)
        state_sizes = torch.cat([current_size, future_sizes], dim=1)
        center_flow = state_centers[:, 1:] - state_centers[:, :-1]
        return torch.cat([state_centers, state_sizes], dim=-1), center_flow, offsets

    def state_centers(
        self,
        future_target_tokens: torch.Tensor,
        target_state: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compatibility view returning s0..sH centers and adjacent flows."""
        state_boxes, center_flow, _ = self.state_boxes(future_target_tokens, target_state)
        return state_boxes[..., :2], center_flow

    @staticmethod
    def _cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
        center, size = boxes[..., :2], boxes[..., 2:].clamp_min(1.0e-6)
        half = 0.5 * size
        return torch.cat([center - half, center + half], dim=-1)

    @classmethod
    def _aligned_generalized_iou(cls, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_xyxy = cls._cxcywh_to_xyxy(pred)
        target_xyxy = cls._cxcywh_to_xyxy(target)
        inter_lt = torch.maximum(pred_xyxy[..., :2], target_xyxy[..., :2])
        inter_rb = torch.minimum(pred_xyxy[..., 2:], target_xyxy[..., 2:])
        inter_wh = (inter_rb - inter_lt).clamp_min(0.0)
        intersection = inter_wh.prod(dim=-1)
        pred_area = (pred_xyxy[..., 2:] - pred_xyxy[..., :2]).clamp_min(0.0).prod(dim=-1)
        target_area = (target_xyxy[..., 2:] - target_xyxy[..., :2]).clamp_min(0.0).prod(dim=-1)
        union = (pred_area + target_area - intersection).clamp_min(1.0e-8)
        iou = intersection / union
        enclosing_lt = torch.minimum(pred_xyxy[..., :2], target_xyxy[..., :2])
        enclosing_rb = torch.maximum(pred_xyxy[..., 2:], target_xyxy[..., 2:])
        enclosing_area = (enclosing_rb - enclosing_lt).clamp_min(0.0).prod(dim=-1).clamp_min(1.0e-8)
        return iou - (enclosing_area - union) / enclosing_area

    def box_state_losses(
        self,
        pred_state_boxes: torch.Tensor,
        target_boxes: torch.Tensor,
        target_box_valid: torch.Tensor,
        sequence_valid: torch.Tensor | None,
        horizon_discount: float,
    ) -> dict[str, torch.Tensor]:
        """Supervise normalized cxcywh boxes with Smooth L1 and aligned GIoU."""
        required_states = self.action_horizon + 1
        if pred_state_boxes.shape[1:] != (required_states, 4):
            raise ValueError(
                f"pred_state_boxes must be [B,{required_states},4], got {tuple(pred_state_boxes.shape)}."
            )
        if target_boxes.ndim != 3 or target_boxes.size(1) < required_states or target_boxes.size(2) != 4:
            raise ValueError(f"target_boxes must cover {required_states} normalized cxcywh states.")
        pred = pred_state_boxes.float()
        target = target_boxes[:, :required_states].to(device=pred.device, dtype=pred.dtype)
        valid = target_box_valid.to(device=pred.device, dtype=pred.dtype)
        if valid.ndim == 3:
            valid = valid.squeeze(-1)
        valid = valid[:, :required_states]
        if sequence_valid is not None:
            valid = valid * sequence_valid[:, :required_states].to(device=pred.device, dtype=pred.dtype)
        future_weights = float(horizon_discount) ** torch.arange(
            self.action_horizon, device=pred.device, dtype=pred.dtype
        )
        weights = torch.cat([torch.ones(1, device=pred.device, dtype=pred.dtype), future_weights])
        weights = valid * weights.unsqueeze(0)
        denominator = weights.sum().clamp_min(1.0)
        l1_error = F.smooth_l1_loss(pred, target, reduction="none").mean(dim=-1)
        giou_error = 1.0 - self._aligned_generalized_iou(pred, target)
        return {
            "l1": (l1_error * weights).sum() / denominator,
            "giou": (giou_error * weights).sum() / denominator,
        }

    def center_state_losses(
        self,
        pred_state_centers: torch.Tensor,
        pred_center_flow: torch.Tensor,
        target_centers: torch.Tensor,
        target_center_valid: torch.Tensor,
        sequence_valid: torch.Tensor | None,
        horizon_discount: float,
    ) -> dict[str, torch.Tensor]:
        """Supervise s0..sH and transitions while masking invalid sequence tails."""
        horizon = self.action_horizon
        required_states = horizon + 1
        if pred_state_centers.shape[1:] != (required_states, 2):
            raise ValueError(
                f"pred_state_centers must be [B,{required_states},2], "
                f"got {tuple(pred_state_centers.shape)}."
            )
        if pred_center_flow.shape[1:] != (horizon, 2):
            raise ValueError(
                f"pred_center_flow must be [B,{horizon},2], got {tuple(pred_center_flow.shape)}."
            )
        if target_centers.ndim != 3 or target_centers.size(1) < required_states:
            available = target_centers.size(1) if target_centers.ndim >= 2 else 0
            raise ValueError(
                "State-action alignment requires current state plus one post-action state "
                f"per action; need {required_states} centers, got {available}."
            )
        if not 0.0 < float(horizon_discount) <= 1.0:
            raise ValueError("tracker_future_horizon_discount must be in (0, 1].")

        pred_states = pred_state_centers.float()
        pred_flow = pred_center_flow.float()
        gt_states = target_centers[:, :required_states].to(
            device=pred_states.device, dtype=pred_states.dtype
        )
        state_valid = target_center_valid.to(device=pred_states.device, dtype=pred_states.dtype)
        if state_valid.ndim == 3:
            state_valid = state_valid.squeeze(-1)
        if state_valid.ndim != 2 or state_valid.size(1) < required_states:
            raise ValueError(
                f"target_center_valid must cover {required_states} states, "
                f"got {tuple(state_valid.shape)}."
            )
        state_valid = state_valid[:, :required_states]
        if sequence_valid is not None:
            if sequence_valid.ndim != 2 or sequence_valid.size(1) < required_states:
                raise ValueError(
                    f"sequence_valid must cover {required_states} states, "
                    f"got {tuple(sequence_valid.shape)}."
                )
            state_valid = state_valid * sequence_valid[:, :required_states].to(
                device=state_valid.device, dtype=state_valid.dtype
            )

        future_weights = float(horizon_discount) ** torch.arange(
            horizon, device=pred_states.device, dtype=pred_states.dtype
        )
        current_error = F.smooth_l1_loss(
            pred_states[:, 0], gt_states[:, 0], reduction="none"
        ).mean(dim=-1)
        current_valid = state_valid[:, 0]
        current = (current_error * current_valid).sum() / current_valid.sum().clamp_min(1.0)

        future_error = F.smooth_l1_loss(
            pred_states[:, 1:], gt_states[:, 1:], reduction="none"
        ).mean(dim=-1)
        future_valid = state_valid[:, 1:] * future_weights.unsqueeze(0)
        future = (future_error * future_valid).sum() / future_valid.sum().clamp_min(1.0)

        gt_flow = gt_states[:, 1:] - gt_states[:, :-1]
        transition_error = F.smooth_l1_loss(
            pred_flow, gt_flow, reduction="none"
        ).mean(dim=-1)
        transition_valid = (
            state_valid[:, 1:] * state_valid[:, :-1] * future_weights.unsqueeze(0)
        )
        transition = (
            (transition_error * transition_valid).sum()
            / transition_valid.sum().clamp_min(1.0)
        )
        return {"current": current, "future": future, "transition": transition}

    def delta(
        self,
        layer_index: int,
        action_hidden: torch.Tensor,
        video_hidden: torch.Tensor,
        target_state: dict[str, torch.Tensor],
        action_timestep: torch.Tensor,
        *,
        return_tokens: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if layer_index < self.start_layer:
            delta = torch.zeros_like(action_hidden)
            return (delta, delta) if return_tokens else delta
        layer = self.layers[str(layer_index)]
        future_tokens = self.future_tokens(
            layer_index, action_hidden, video_hidden, target_state, action_timestep
        )
        current = target_state["current_token"].expand(-1, self.action_horizon, -1)
        action_state_tokens = self.action_aligned_state_tokens(future_tokens, target_state)
        reliability_input = torch.cat(
            [action_hidden, action_state_tokens, current], dim=-1
        )
        reliability = torch.sigmoid(self.dynamic_gate(reliability_input))
        delta = torch.tanh(layer.gate) * reliability * layer.output(action_state_tokens)
        return (delta, future_tokens) if return_tokens else delta


class FutureStateConditioner(nn.Module):
    """Build current s0 and 320-token Tracker memory for the StateDiT expert."""

    def __init__(self, tracker_dim: int, state_hidden_dim: int) -> None:
        super().__init__()
        self.tracker_dim = int(tracker_dim)
        self.template_projection = nn.Sequential(
            nn.LayerNorm(tracker_dim),
            nn.Linear(tracker_dim, state_hidden_dim),
            nn.GELU(),
            nn.LayerNorm(state_hidden_dim),
        )
        self.target_query = nn.Parameter(torch.zeros(1, 1, state_hidden_dim))
        self.template_pool = nn.MultiheadAttention(state_hidden_dim, num_heads=8, batch_first=True)
        self.target_search = nn.MultiheadAttention(state_hidden_dim, num_heads=8, batch_first=True)
        size_hidden = max(state_hidden_dim // 4, 1)
        self.current_size_head = nn.Sequential(
            nn.LayerNorm(state_hidden_dim),
            nn.Linear(state_hidden_dim, size_hidden),
            nn.GELU(),
            nn.Linear(size_hidden, 2),
        )
        self.box_embedding = nn.Sequential(
            nn.Linear(4, state_hidden_dim),
            nn.SiLU(),
            nn.Linear(state_hidden_dim, state_hidden_dim),
            nn.LayerNorm(state_hidden_dim),
        )
        nn.init.trunc_normal_(self.target_query, std=0.02)
        nn.init.constant_(self.current_size_head[-1].bias, -3.0)

    def forward(
        self,
        template_features: torch.Tensor,
        search_tokens: torch.Tensor,
        full_xy: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        expected_template = (64, self.tracker_dim)
        if template_features.ndim != 3 or tuple(template_features.shape[1:]) != expected_template:
            raise ValueError(
                f"template_features must be [B,{expected_template[0]},{expected_template[1]}], "
                f"got {tuple(template_features.shape)}."
            )
        if search_tokens.ndim != 3 or search_tokens.size(1) != 256:
            raise ValueError(f"search_tokens must be [B,256,D], got {tuple(search_tokens.shape)}.")
        if full_xy.shape != (search_tokens.size(0), 256, 2):
            raise ValueError(f"full_xy must be [B,256,2], got {tuple(full_xy.shape)}.")
        template_tokens = self.template_projection(template_features.to(search_tokens))
        query = self.target_query.to(search_tokens).expand(search_tokens.size(0), -1, -1)
        identity, _ = self.template_pool(query, template_tokens, template_tokens, need_weights=False)
        current_token, attention = self.target_search(
            identity, search_tokens, search_tokens, need_weights=True, average_attn_weights=True
        )
        attention = attention.clamp_min(1.0e-8)
        center = torch.bmm(attention, full_xy.to(search_tokens)).squeeze(1)
        size = torch.sigmoid(self.current_size_head(current_token.squeeze(1))).clamp(1.0e-4, 1.0)
        current_box = torch.cat([center.clamp(0.0, 1.0), size], dim=-1)
        box_features = torch.cat(
            [
                current_box[:, :2].mul(2.0).sub(1.0),
                current_box[:, 2:].clamp_min(1.0e-4).log(),
            ],
            dim=-1,
        )
        current_condition = current_token.squeeze(1) + self.box_embedding(box_features)
        tracker_memory = torch.cat([template_tokens, search_tokens], dim=1)
        if not torch.isfinite(current_box).all() or not torch.isfinite(tracker_memory).all():
            raise FloatingPointError("Future State conditioning produced NaN/Inf values.")
        return {
            "current_box": current_box,
            "current_condition": current_condition,
            "tracker_memory": tracker_memory,
            "current_attention": attention,
            "full_xy": full_xy,
        }

    @staticmethod
    def current_tracking_losses(
        current_box: torch.Tensor,
        current_attention: torch.Tensor,
        full_xy: torch.Tensor,
        target_box: torch.Tensor,
        valid: torch.Tensor,
        image_size: torch.Tensor,
        attention_sigma: float,
    ) -> dict[str, torch.Tensor]:
        """Supervise s0 in Search-grid coordinates and its Template attention."""
        if current_box.ndim != 2 or current_box.size(-1) != 4:
            raise ValueError("current_box must be [B,4].")
        batch = current_box.size(0)
        if current_attention.shape != (batch, 1, 256):
            raise ValueError("current_attention must be [B,1,256].")
        if full_xy.shape != (batch, 256, 2):
            raise ValueError("full_xy must be [B,256,2].")
        if target_box.shape != (batch, 4):
            raise ValueError("target_box must be [B,4].")
        if image_size.shape != (batch, 2):
            raise ValueError("image_size must be [B,2] as [height,width].")

        pred = current_box.float()
        target = target_box.to(device=pred.device, dtype=pred.dtype)
        coords = full_xy.to(device=pred.device, dtype=pred.dtype)
        sample_valid = valid.to(device=pred.device, dtype=pred.dtype).reshape(batch)
        denominator = sample_valid.sum().clamp_min(1.0)

        grid_size = 16
        grid_span = (coords.amax(dim=1) - coords.amin(dim=1)).clamp_min(1.0e-6)
        crop_span = grid_span * (float(grid_size) / float(grid_size - 1))
        center_delta = (pred[:, :2] - target[:, :2]) / crop_span
        center_error = F.smooth_l1_loss(
            center_delta, torch.zeros_like(center_delta), reduction="none"
        ).mean(dim=-1)

        giou = BoxFreeFutureTargetReadout._aligned_generalized_iou(pred, target)
        giou_error = 1.0 - giou

        cell_span = crop_span / float(grid_size)
        offset_cells = (coords - target[:, None, :2]) / cell_span[:, None, :]
        sigma = max(float(attention_sigma), 1.0e-3)
        target_attention = torch.softmax(
            -0.5 * offset_cells.square().sum(dim=-1) / (sigma * sigma), dim=-1
        )
        predicted_attention = current_attention[:, 0].float().clamp_min(1.0e-8)
        predicted_attention = predicted_attention / predicted_attention.sum(
            dim=-1, keepdim=True
        ).clamp_min(1.0e-8)
        attention_kl = (
            target_attention
            * (
                target_attention.clamp_min(1.0e-8).log()
                - predicted_attention.log()
            )
        ).sum(dim=-1)

        image_hw = image_size.to(device=pred.device, dtype=pred.dtype)
        image_wh = image_hw[:, [1, 0]]
        center_error_pixels = (
            (pred[:, :2] - target[:, :2]) * image_wh
        ).square().sum(dim=-1).sqrt()

        return {
            "center": (center_error * sample_valid).sum() / denominator,
            "giou": (giou_error * sample_valid).sum() / denominator,
            "attention": (attention_kl * sample_valid).sum() / denominator,
            "center_error_pixels": (
                center_error_pixels * sample_valid
            ).sum() / denominator,
        }

    @staticmethod
    def relative_states(target_boxes: torch.Tensor, eps: float = 1.0e-4) -> torch.Tensor:
        if target_boxes.ndim != 3 or target_boxes.size(-1) != 4:
            raise ValueError("target_boxes must be [B,T,4] normalized cxcywh.")
        current = target_boxes[:, :1].float()
        future = target_boxes[:, 1:].float()
        relative = torch.cat(
            [
                future[..., :2] - current[..., :2],
                (future[..., 2:].clamp_min(eps) / current[..., 2:].clamp_min(eps)).log(),
            ],
            dim=-1,
        )
        if not torch.isfinite(relative).all():
            raise FloatingPointError("Relative future states contain NaN/Inf.")
        return relative

    @staticmethod
    def decode_relative_states(
        current_box: torch.Tensor,
        relative_states: torch.Tensor,
    ) -> torch.Tensor:
        if current_box.ndim != 2 or current_box.size(-1) != 4:
            raise ValueError("current_box must be [B,4].")
        if relative_states.ndim != 3 or relative_states.size(-1) != 4:
            raise ValueError("relative_states must be [B,H,4].")
        current = current_box.float().unsqueeze(1)
        centers = (current[..., :2] + relative_states.float()[..., :2]).clamp(0.0, 1.0)
        sizes = (
            current[..., 2:].clamp_min(1.0e-4)
            * relative_states.float()[..., 2:].clamp(-6.0, 6.0).exp()
        ).clamp(1.0e-4, 1.0)
        future = torch.cat([centers, sizes], dim=-1)
        boxes = torch.cat([current.clamp(1.0e-4, 1.0), future], dim=1)
        if not torch.isfinite(boxes).all():
            raise FloatingPointError("Decoded state boxes contain NaN/Inf.")
        return boxes


class LocalFeatureTrackerConditionFusion(nn.Module):
    """Build 256 geometry-aware Tracker search tokens plus one bbox token."""

    def __init__(
        self,
        tracker_dim: int = 192,
        action_dim: int = 1024,
        num_heads: int = 24,
        head_dim: int = 128,
        num_layers: int = 30,
        start_layer: int = 18,
        grid_size: int = 16,
        use_local_position_embedding: bool = False,
        include_box_token: bool = True,
        gate_init: float = 0.0,
        detach_tracker_inputs: bool = True,
        enable_cross_attention: bool = True,
    ) -> None:
        super().__init__()
        self.detach_tracker_inputs = bool(detach_tracker_inputs)
        self.start_layer = int(start_layer)
        self.grid_size = max(int(grid_size), 1)
        self.num_spatial_tokens = self.grid_size**2
        self.use_local_position_embedding = bool(use_local_position_embedding)
        self.include_box_token = bool(include_box_token)
        self.enable_cross_attention = bool(enable_cross_attention)
        self.feature_projection = nn.Sequential(
            nn.LayerNorm(tracker_dim),
            nn.Linear(tracker_dim, action_dim),
            nn.GELU(),
            nn.Linear(action_dim, action_dim),
            nn.LayerNorm(action_dim),
        )
        if self.use_local_position_embedding:
            self.local_position_embedding = nn.Parameter(
                torch.zeros(1, self.num_spatial_tokens, action_dim)
            )
        else:
            self.register_parameter("local_position_embedding", None)
        self.full_image_geometry_embedding = nn.Sequential(
            nn.Linear(3, 256),
            nn.GELU(),
            nn.Linear(256, action_dim),
            nn.LayerNorm(action_dim),
        )
        self.tracker_modality_embedding = nn.Parameter(
            torch.zeros(1, 1, action_dim)
        )
        if self.include_box_token:
            self.box_mlp = nn.Sequential(
                nn.Linear(4, 256),
                nn.GELU(),
                nn.Linear(256, action_dim),
                nn.LayerNorm(action_dim),
            )
        else:
            self.box_mlp = None
        self.condition_norm = nn.LayerNorm(action_dim)
        if self.local_position_embedding is not None:
            nn.init.trunc_normal_(self.local_position_embedding, std=0.02)
        nn.init.trunc_normal_(self.tracker_modality_embedding, std=0.02)

        local = (torch.arange(self.grid_size, dtype=torch.float32) + 0.5) / float(
            self.grid_size
        )
        grid_y, grid_x = torch.meshgrid(local, local, indexing="ij")
        self.register_buffer(
            "local_grid_coordinates",
            torch.stack([grid_x, grid_y], dim=-1).reshape(1, -1, 2),
            persistent=False,
        )
        self.layers = nn.ModuleDict(
            {
                str(index): ActionTrackerCrossAttention(
                    action_dim, num_heads, head_dim, gate_init=gate_init
                )
                for index in range(self.start_layer, int(num_layers))
            }
            if self.enable_cross_attention else {}
        )

    def full_image_coordinates(
        self,
        search_geometry: torch.Tensor,
        image_size: torch.Tensor,
    ) -> torch.Tensor:
        """Map search-grid cell centers to normalized full-image x/y coordinates."""
        if search_geometry.ndim != 2 or search_geometry.size(-1) != 3:
            raise ValueError("search_geometry must be [B,3] as [x1,y1,side].")
        if image_size.ndim != 2 or image_size.size(-1) != 2:
            raise ValueError("image_size must be [B,2] as [height,width].")
        if search_geometry.size(0) != image_size.size(0):
            raise ValueError("search_geometry and image_size batch sizes must match.")
        geometry = torch.nan_to_num(
            search_geometry.detach().float(), nan=0.0, posinf=0.0, neginf=0.0
        )
        size = torch.nan_to_num(
            image_size.detach().float(), nan=1.0, posinf=1.0, neginf=1.0
        ).clamp_min(1.0)
        local = self.local_grid_coordinates.to(
            device=geometry.device, dtype=geometry.dtype
        )
        x1, y1, side = geometry.unbind(dim=-1)
        image_h, image_w = size.unbind(dim=-1)
        full_x = (x1[:, None] + local[..., 0] * side[:, None]) / image_w[:, None]
        full_y = (y1[:, None] + local[..., 1] * side[:, None]) / image_h[:, None]
        return torch.stack([full_x, full_y], dim=-1)

    def full_image_geometry(
        self,
        search_geometry: torch.Tensor,
        image_size: torch.Tensor,
    ) -> torch.Tensor:
        """Return full-image token coordinates and normalized square-crop scale."""
        full_xy = self.full_image_coordinates(search_geometry, image_size)
        geometry = torch.nan_to_num(
            search_geometry.detach().float(), nan=0.0, posinf=0.0, neginf=0.0
        )
        size = torch.nan_to_num(
            image_size.detach().float(), nan=1.0, posinf=1.0, neginf=1.0
        ).clamp_min(1.0)
        crop_scale = geometry[:, 2].clamp_min(0.0) / (
            size[:, 0] * size[:, 1]
        ).sqrt().clamp_min(1.0)
        crop_scale = crop_scale[:, None, None].expand(-1, self.num_spatial_tokens, -1)
        return torch.cat([full_xy, crop_scale], dim=-1)

    def make_condition(
        self,
        tracker_features: torch.Tensor,
        tracker_bbox: torch.Tensor | None,
        tracker_search_geometry: torch.Tensor,
        tracker_image_size: torch.Tensor,
    ) -> torch.Tensor:
        if tracker_features.ndim != 3:
            raise ValueError("tracker_features must be [B,256,192].")
        batch, token_count, _ = tracker_features.shape
        if token_count != self.num_spatial_tokens:
            raise ValueError(
                f"Expected {self.num_spatial_tokens} Tracker tokens, got {token_count}."
            )
        if self.include_box_token and (
            tracker_bbox is None or tracker_bbox.shape != (batch, 4)
        ):
            raise ValueError("tracker_bbox must be [B,4] as normalized cxcywh when using a box token.")

        param = next(self.parameters())
        device, dtype = param.device, param.dtype
        source_features = tracker_features.detach() if self.detach_tracker_inputs else tracker_features
        source_bbox = None if tracker_bbox is None else (
            tracker_bbox.detach() if self.detach_tracker_inputs else tracker_bbox
        )
        features = torch.nan_to_num(
            source_features, nan=0.0, posinf=1.0e4, neginf=-1.0e4
        ).to(device=device, dtype=dtype)
        bbox = None if source_bbox is None else torch.nan_to_num(
            source_bbox.float(), nan=0.0, posinf=1.0, neginf=0.0
        ).clamp(0.0, 1.0).to(device=device, dtype=dtype)
        full_geometry = self.full_image_geometry(
            tracker_search_geometry.to(device=device),
            tracker_image_size.to(device=device),
        ).to(dtype=dtype)
        spatial_geometry = torch.cat(
            [
                full_geometry[..., :2].mul(2.0).sub(1.0).clamp(-4.0, 4.0),
                full_geometry[..., 2:].clamp(0.0, 4.0),
            ],
            dim=-1,
        )

        spatial_tokens = (
            self.feature_projection(features)
            + self.full_image_geometry_embedding(spatial_geometry)
            + self.tracker_modality_embedding.to(device=device, dtype=dtype)
        )
        if self.local_position_embedding is not None:
            spatial_tokens = spatial_tokens + self.local_position_embedding.to(
                device=device, dtype=dtype
            )
        spatial_tokens = self.condition_norm(spatial_tokens)
        if not self.include_box_token:
            return spatial_tokens
        assert self.box_mlp is not None and bbox is not None
        box_token = self.box_mlp(bbox).unsqueeze(1)
        return torch.cat([spatial_tokens, box_token], dim=1)

    def delta(
        self,
        layer_index: int,
        action_hidden: torch.Tensor,
        condition: torch.Tensor,
        *,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor | None]:
        if not self.enable_cross_attention or layer_index < self.start_layer:
            delta = torch.zeros_like(action_hidden)
            return (delta, None) if return_attention else delta
        return self.layers[str(layer_index)](
            action_hidden, condition, return_attention=return_attention
        )


class FrozenTrackerConditionFusion(nn.Module):
    """Projects frozen DeiT search features and bbox center for late action fusion."""

    def __init__(
        self,
        tracker_dim: int = 192,
        action_dim: int = 1024,
        num_heads: int = 24,
        head_dim: int = 128,
        num_layers: int = 30,
        start_layer: int = 18,
        condition_mode: str = "center_features",
        response_grid_size: int = 7,
        gate_init: float = 0.0,
    ) -> None:
        super().__init__()
        self.start_layer = int(start_layer)
        self.condition_mode = str(condition_mode).strip().lower()
        valid_modes = {
            "none",
            "center",
            "bbox",
            "features",
            "response",
            "center_features",
            "bbox_response",
            "bbox_response_features",
        }
        if self.condition_mode not in valid_modes:
            raise ValueError(
                f"Unsupported tracker_condition_mode={self.condition_mode!r}; "
                f"expected one of {sorted(valid_modes)}."
            )
        self.response_grid_size = max(int(response_grid_size), 1)
        self.feature_projection = nn.Sequential(
            nn.LayerNorm(tracker_dim),
            nn.Linear(tracker_dim, action_dim),
            nn.GELU(),
            nn.Linear(action_dim, action_dim),
            nn.LayerNorm(action_dim),
        )
        self.center_embedding = nn.Sequential(
            nn.Linear(2, 256),
            nn.GELU(),
            nn.Linear(256, action_dim),
            nn.LayerNorm(action_dim),
        )
        self.bbox_embedding = nn.Sequential(
            nn.Linear(4, 256),
            nn.GELU(),
            nn.Linear(256, action_dim),
            nn.LayerNorm(action_dim),
        )
        self.response_embedding = nn.Sequential(
            nn.Linear(3, 256),
            nn.GELU(),
            nn.Linear(256, action_dim),
            nn.LayerNorm(action_dim),
        )
        coords = torch.linspace(-1.0, 1.0, self.response_grid_size)
        grid_y, grid_x = torch.meshgrid(coords, coords, indexing="ij")
        self.register_buffer(
            "response_coordinates",
            torch.stack([grid_x, grid_y], dim=-1).reshape(1, -1, 2),
            persistent=False,
        )
        self.layers = nn.ModuleDict(
            {
                str(index): ActionTrackerCrossAttention(
                    action_dim, num_heads, head_dim, gate_init=gate_init
                )
                for index in range(self.start_layer, int(num_layers))
            }
        )

    def make_condition(
        self,
        tracker_features: torch.Tensor | None = None,
        tracker_center: torch.Tensor | None = None,
        tracker_bbox: torch.Tensor | None = None,
        tracker_response: torch.Tensor | None = None,
    ) -> torch.Tensor:
        param = next(self.parameters())
        tokens: list[torch.Tensor] = []

        if "features" in self.condition_mode:
            if tracker_features is None or tracker_features.ndim != 3:
                raise ValueError("Feature condition requires tracker_features [B,N,C].")
            features = tracker_features.detach().to(device=param.device, dtype=param.dtype)
            tokens.append(self.feature_projection(features))

        if self.condition_mode in {"center", "center_features"}:
            if tracker_center is None:
                raise ValueError("Center condition requires tracker_center.")
            if tracker_center.ndim == 3:
                tracker_center = tracker_center[:, 0]
            if tracker_center.ndim != 2 or tracker_center.size(-1) != 2:
                raise ValueError("tracker_center must have shape [B,2] or [B,T,2].")
            center = tracker_center.detach().to(device=param.device, dtype=param.dtype)
            center = center.clamp(0.0, 1.0).mul(2.0).sub(1.0)
            tokens.append(self.center_embedding(center).unsqueeze(1))

        if self.condition_mode in {"bbox", "bbox_response", "bbox_response_features"}:
            if tracker_bbox is None:
                raise ValueError("BBox condition requires tracker_bbox.")
            if tracker_bbox.ndim == 3:
                tracker_bbox = tracker_bbox[:, 0]
            if tracker_bbox.ndim != 2 or tracker_bbox.size(-1) != 4:
                raise ValueError("tracker_bbox must have shape [B,4] or [B,T,4].")
            bbox = tracker_bbox.detach().to(device=param.device, dtype=param.dtype)
            bbox = bbox.clamp(0.0, 1.0)
            bbox_xy = bbox[:, :2].mul(2.0).sub(1.0)
            bbox_wh = bbox[:, 2:].clamp_min(1.0e-4).log()
            tokens.append(self.bbox_embedding(torch.cat([bbox_xy, bbox_wh], dim=-1)).unsqueeze(1))

        if "response" in self.condition_mode:
            if tracker_response is None:
                raise ValueError("Response condition requires tracker_response.")
            if tracker_response.ndim == 4:
                tracker_response = tracker_response[:, 0]
            expected = (self.response_grid_size, self.response_grid_size)
            if tracker_response.ndim != 3 or tuple(tracker_response.shape[-2:]) != expected:
                raise ValueError(
                    f"tracker_response must have shape [B,{expected[0]},{expected[1]}] "
                    f"or [B,T,{expected[0]},{expected[1]}]."
                )
            response = tracker_response.detach().to(device=param.device, dtype=param.dtype)
            response = response.clamp_min(0.0)
            response = response / response.sum(dim=(-2, -1), keepdim=True).clamp_min(1.0e-8)
            response = response.reshape(response.size(0), -1, 1)
            coords = self.response_coordinates.to(device=param.device, dtype=param.dtype)
            coords = coords.expand(response.size(0), -1, -1)
            tokens.append(self.response_embedding(torch.cat([response, coords], dim=-1)))

        if not tokens:
            raise RuntimeError("tracker_condition_mode='none' does not produce condition tokens.")
        return torch.cat(tokens, dim=1)

    def delta(
        self, layer_index: int, action_hidden: torch.Tensor, condition: torch.Tensor
    ) -> torch.Tensor:
        if layer_index < self.start_layer:
            return torch.zeros_like(action_hidden)
        return self.layers[str(layer_index)](action_hidden, condition)
