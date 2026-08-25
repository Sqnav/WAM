from __future__ import annotations

import unittest

import torch

from model.capture_value_reranker import (
    CaptureValueHead,
    build_structured_recenter_candidates,
    approximate_capture_outcomes,
    build_capture_value_candidates,
    capture_value_loss,
    score_geometric_capture_trajectories,
    select_geometric_candidate,
    sample_grouped_candidate_noise,
)
from model.config import ModelConfig
from model.losses import world_model_dit_loss


class CaptureValueRerankerTests(unittest.TestCase):
    def test_structured_candidates_preserve_base_and_recenter(self) -> None:
        candidates = torch.randn(1, 4, 8, 4)
        base = candidates[:, 0].clone()
        box = torch.tensor([[0.25, 0.10, 0.05, 0.05]])
        output = build_structured_recenter_candidates(candidates, box)
        self.assertTrue(torch.equal(output[:, 0], base))
        self.assertLess(float(output[0, 3, 0, 1]), 0.0)
        self.assertLess(float(output[0, 3, 0, 2]), 0.0)
        self.assertLess(float(output[0, 3, 0, 3]), 0.0)
        translation_norm = torch.linalg.vector_norm(output[0, 1:, 0, :3], dim=-1)
        self.assertTrue(torch.allclose(translation_norm, torch.ones_like(translation_norm)))

    @staticmethod
    def _geometric_score(
        candidates: torch.Tensor, box: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        return score_geometric_capture_trajectories(
            candidates,
            box,
            torch.zeros(candidates.size(0), 4),
            max_vel=1.0,
            max_yaw_rate=15.0,
            max_speed_norm=1.0,
        )

    def test_geometric_recenter_prefers_yaw_toward_target(self) -> None:
        candidates = torch.zeros(1, 2, 8, 4)
        candidates[0, 0, 0, 3] = 0.5
        candidates[0, 1, 0, 3] = -0.5
        box = torch.tensor([[0.75, 0.5, 0.06, 0.06]])

        result = self._geometric_score(candidates, box)

        self.assertLess(
            result["recenter_cost"][0, 0].item(),
            result["recenter_cost"][0, 1].item(),
        )
        self.assertGreater(result["score"][0, 0].item(), result["score"][0, 1].item())

    def test_extra_candidates_do_not_advance_base_rng_stream(self) -> None:
        torch.manual_seed(7)
        expected_base = torch.randn(2, 8, 4)
        expected_next = torch.randn(2, 8, 4)

        torch.manual_seed(7)
        grouped = sample_grouped_candidate_noise(
            2,
            4,
            8,
            4,
            device=torch.device("cpu"),
            dtype=torch.float32,
        ).reshape(2, 4, 8, 4)
        actual_next = torch.randn(2, 8, 4)

        torch.testing.assert_close(grouped[:, 0], expected_base)
        torch.testing.assert_close(actual_next, expected_next)
        self.assertFalse(torch.equal(grouped[:, 0], grouped[:, 1]))

    def test_geometric_pursuit_preserves_base_forward_speed(self) -> None:
        candidates = torch.zeros(1, 2, 8, 4)
        candidates[0, 0, :, 0] = 0.8
        candidates[0, 1, :, 0] = 0.2
        box = torch.tensor([[0.5, 0.5, 0.10, 0.10]])

        result = self._geometric_score(candidates, box)

        self.assertLess(
            result["pursuit_cost"][0, 0].item(),
            result["pursuit_cost"][0, 1].item(),
        )
        self.assertGreater(result["score"][0, 0].item(), result["score"][0, 1].item())

    def test_geometric_recenter_does_not_invent_vertical_control(self) -> None:
        candidates = torch.zeros(1, 2, 8, 4)
        candidates[0, 0, :2, 2] = -0.5
        candidates[0, 1, :2, 2] = 0.5
        box = torch.tensor([[0.5, 0.25, 0.06, 0.06]])

        result = self._geometric_score(candidates, box)

        torch.testing.assert_close(
            result["recenter_cost"][0, 0],
            result["recenter_cost"][0, 1],
        )

    def test_geometric_components_are_normalized(self) -> None:
        result = self._geometric_score(
            torch.randn(2, 4, 8, 4),
            torch.tensor([[0.1, 0.9, 0.05, 0.05], [0.5, 0.5, 0.1, 0.1]]),
        )

        for name in (
            "recenter_cost",
            "pursuit_cost",
            "smooth_cost",
            "consensus_cost",
        ):
            self.assertTrue(torch.all(result[name] >= 0.0), name)
            self.assertTrue(torch.all(result[name] <= 1.0), name)

    def test_recent_box_motion_changes_dynamic_action_prior(self) -> None:
        candidates = torch.zeros(1, 2, 8, 4)
        candidates[0, 0, 0, 3] = -0.2
        candidates[0, 1, 0, 3] = 0.2
        current = torch.tensor([[0.6, 0.5, 0.06, 0.06]])
        history = torch.zeros(1, 2, 5)
        history[0, -1, :4] = torch.tensor([0.5, 0.5, 0.06, 0.06])
        valid = torch.tensor([[False, True]])

        result = score_geometric_capture_trajectories(
            candidates,
            current,
            torch.zeros(1, 4),
            history,
            valid,
            max_vel=1.0,
            max_yaw_rate=15.0,
            max_speed_norm=1.0,
        )

        self.assertLess(
            result["recenter_cost"][0, 0].item(),
            result["recenter_cost"][0, 1].item(),
        )
        self.assertGreater(result["observed_center_velocity"][0].item(), 0.0)

    def test_tail_changes_only_smoothness(self) -> None:
        candidates = torch.zeros(1, 2, 8, 4)
        candidates[0, 1, 2:, 3] = torch.tensor([1.0, -1.0] * 3)
        result = self._geometric_score(
            candidates, torch.tensor([[0.5, 0.5, 0.06, 0.06]])
        )

        for name in ("recenter_cost", "pursuit_cost", "consensus_cost"):
            torch.testing.assert_close(result[name][0, 0], result[name][0, 1])
        self.assertLess(
            result["smooth_cost"][0, 0].item(),
            result["smooth_cost"][0, 1].item(),
        )

    def test_geometric_smoothness_penalizes_oscillation(self) -> None:
        candidates = torch.zeros(1, 2, 8, 4)
        candidates[0, 1, :, 3] = torch.tensor([1.0, -1.0] * 4)
        box = torch.tensor([[0.5, 0.5, 0.061, 0.061]])

        result = self._geometric_score(candidates, box)

        self.assertLess(
            result["smooth_cost"][0, 0].item(),
            result["smooth_cost"][0, 1].item(),
        )

    def test_geometric_runtime_translation_clipping_matches_limit(self) -> None:
        candidates = torch.ones(1, 1, 8, 4)
        box = torch.tensor([[0.5, 0.5, 0.061, 0.061]])

        result = self._geometric_score(candidates, box)

        translation_norm = torch.linalg.vector_norm(
            result["physical_candidates"][..., :3], dim=-1
        )
        self.assertTrue(torch.all(translation_norm <= 1.0 + 1.0e-6))

    def test_candidate_zero_is_kept_below_selection_margin(self) -> None:
        selection = select_geometric_candidate(
            torch.tensor([[1.0, 1.02, 0.9, 0.8], [1.0, 1.05, 0.9, 0.8]]),
            selection_margin=0.03,
        )

        self.assertEqual(selection["selected_index"].tolist(), [0, 1])
        self.assertEqual(selection["used_fallback"].tolist(), [True, False])

    def test_candidate_zero_is_kept_when_recentering_is_not_needed(self) -> None:
        selection = select_geometric_candidate(
            torch.tensor([[1.0, 1.5], [1.0, 1.5]]),
            selection_margin=0.1,
            allow_switch=torch.tensor([False, True]),
        )

        self.assertEqual(selection["raw_selected_index"].tolist(), [1, 1])
        self.assertEqual(selection["selected_index"].tolist(), [0, 1])
        self.assertEqual(selection["used_fallback"].tolist(), [True, False])

    def test_candidate_builder_keeps_exact_expert_candidate(self) -> None:
        torch.manual_seed(3)
        predicted = torch.zeros(2, 8, 4)
        expert = torch.rand(2, 8, 4) * 2.0 - 1.0

        candidates = build_capture_value_candidates(
            predicted,
            expert,
            candidate_count=4,
            noise_std=0.2,
        )

        self.assertEqual(candidates.shape, (2, 4, 8, 4))
        torch.testing.assert_close(candidates[:, 1], expert)
        self.assertTrue(torch.all(candidates <= 1.0))
        self.assertTrue(torch.all(candidates >= -1.0))

    def test_bad_translation_has_worse_capture_outcome(self) -> None:
        expert = torch.zeros(1, 8, 4)
        bad = expert.clone()
        bad[..., 0] = -1.0
        candidates = torch.stack([expert, bad], dim=1)
        target_relative = torch.zeros(1, 9, 3)
        target_relative[..., 0] = 5.0

        outcomes = approximate_capture_outcomes(
            candidates,
            expert,
            target_relative,
            torch.ones(1, 8, dtype=torch.bool),
            max_vel=5.0,
            max_yaw_rate=30.0,
            capture_distance=10.0,
        )

        self.assertEqual(outcomes["capture_probability"].shape, (1, 2))
        self.assertTrue(
            torch.isfinite(outcomes["final_distance_normalized"]).all()
        )
        self.assertGreater(
            outcomes["capture_probability"][0, 0].item(),
            outcomes["capture_probability"][0, 1].item(),
        )
        self.assertLess(
            outcomes["final_distance_normalized"][0, 0].item(),
            outcomes["final_distance_normalized"][0, 1].item(),
        )

    def test_value_head_accepts_fp32_inputs_with_bfloat16_parameters(self) -> None:
        head = CaptureValueHead(
            video_context_dim=16,
            target_context_dim=8,
            action_dim=4,
            horizon=8,
            hidden_dim=32,
            num_layers=1,
            num_heads=4,
        ).bfloat16()
        candidates = torch.randn(2, 4, 8, 4)
        prediction = head(
            candidates,
            video_context=torch.randn(2, 16),
            target_context=torch.randn(2, 8),
            valid_mask=torch.ones(2, 8, dtype=torch.bool),
        )

        for key in (
            "score",
            "capture_logit",
            "final_distance_normalized",
            "visibility_logit",
        ):
            self.assertEqual(prediction[key].shape, (2, 4))
            self.assertTrue(torch.isfinite(prediction[key]).all())

        targets = {
            "capture_probability": torch.rand(2, 4),
            "final_distance_normalized": torch.rand(2, 4),
            "visibility_probability": torch.rand(2, 4),
        }
        losses = capture_value_loss(prediction, targets)
        self.assertTrue(torch.isfinite(losses["loss"]))

    def test_adapter_total_uses_only_capture_value_objective(self) -> None:
        cfg = ModelConfig(
            use_fastwam_mot=True,
            use_capture_value_reranking=True,
            capture_value_adapter_only=True,
            capture_value_loss_weight=0.2,
            x0_action_loss_weight=0.0,
        )
        outputs = {
            "feat": torch.zeros(1, 1),
            "policy_flow_loss": torch.tensor(9.0),
            "video_flow_loss": torch.tensor(7.0),
            "capture_value_loss": torch.tensor(2.5, requires_grad=True),
        }
        batch = {"expert_action": torch.zeros(1, 2, 4)}

        losses = world_model_dit_loss(outputs, batch, cfg)

        torch.testing.assert_close(losses["total"], torch.tensor(0.5))


if __name__ == "__main__":
    unittest.main()
