from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch
from torch import nn

from model.target_action_conditioning import (
    HistoricalTargetMemory,
    TargetActionConditioning,
    augment_target_box_history,
    make_online_target_box_history,
)
from train.train_teacher import _freeze_for_target_conditioning_adapter_training


class TargetActionConditioningTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(17)
        self.batch_size = 2
        self.horizon = 8
        self.previous_length = 7
        self.history = torch.rand(self.batch_size, self.previous_length, 5)
        self.valid = torch.ones(
            self.batch_size, self.previous_length, dtype=torch.bool
        )
        self.current_box = torch.rand(self.batch_size, 4)
        self.current_confidence = torch.rand(self.batch_size, 1)

    def make_conditioner(self) -> TargetActionConditioning:
        return TargetActionConditioning(
            action_hidden_dim=16,
            history_length=8,
            horizon=self.horizon,
            memory_dim=8,
            memory_layers=2,
            memory_heads=2,
        )

    def test_online_history_is_right_aligned_and_excludes_current_box(self) -> None:
        boxes = [torch.full((4,), float(index) / 10.0) for index in range(3)]
        confidence = [0.1, 0.2, 0.3]
        history, valid = make_online_target_box_history(
            boxes,
            confidence,
            previous_length=self.previous_length,
            device=torch.device("cpu"),
        )

        self.assertEqual(history.shape, (1, self.previous_length, 5))
        self.assertEqual(valid.shape, (1, self.previous_length))
        self.assertEqual(int(valid.sum()), 3)
        torch.testing.assert_close(history[0, -1, :4], boxes[-1])
        torch.testing.assert_close(history[0, -1, 4], torch.tensor(0.3))
        self.assertTrue(torch.equal(history[0, :4], torch.zeros(4, 5)))

    def test_invalid_history_boundary_does_not_create_fake_velocity(self) -> None:
        memory = HistoricalTargetMemory(
            action_hidden_dim=16,
            history_length=8,
            horizon=8,
            memory_dim=8,
            num_layers=1,
            num_heads=2,
        )
        previous = torch.zeros(1, 7, 5)
        previous[0, -1] = torch.tensor([0.4, 0.6, 0.2, 0.3, 0.9])
        previous_valid = torch.zeros(1, 7, dtype=torch.bool)
        previous_valid[0, -1] = True
        features, valid = memory._features(
            previous,
            previous_valid,
            torch.tensor([[0.5, 0.7, 0.2, 0.3]]),
            torch.tensor([[0.8]]),
        )

        self.assertTrue(valid[0, -2:].all())
        self.assertTrue(torch.equal(features[0, -2, 4:8], torch.zeros(4)))
        self.assertGreater(float(features[0, -1, 4:6].abs().sum()), 0.0)

    def test_v1_builds_one_condition_per_action_horizon(self) -> None:
        conditioner = self.make_conditioner()
        conditions = conditioner.build_conditions(
            previous_history=self.history,
            previous_valid=self.valid,
            current_box=self.current_box,
            current_confidence=self.current_confidence,
        )

        self.assertEqual(conditions["history"].shape, (self.batch_size, 8, 16))
        hidden = torch.randn(self.batch_size, 8, 16)
        torch.testing.assert_close(conditioner.residual(hidden, conditions), torch.zeros_like(hidden))

    def test_history_memory_accepts_float32_inputs_with_bfloat16_parameters(self) -> None:
        conditioner = self.make_conditioner().to(dtype=torch.bfloat16)
        conditions = conditioner.build_conditions(
            previous_history=self.history,
            previous_valid=self.valid,
            current_box=self.current_box,
            current_confidence=self.current_confidence,
        )
        hidden = torch.randn(self.batch_size, self.horizon, 16)
        residual = conditioner.residual(hidden, conditions)

        self.assertEqual(conditions["history"].dtype, torch.bfloat16)
        self.assertEqual(residual.dtype, hidden.dtype)
        self.assertTrue(torch.isfinite(residual).all())

    def test_history_augmentation_preserves_padding(self) -> None:
        history = self.history.clone()
        valid = self.valid.clone()
        valid[:, :3] = False
        augmented, augmented_valid = augment_target_box_history(
            history,
            valid,
            center_jitter_std=0.01,
            log_size_jitter_std=0.02,
            confidence_dropout_probability=0.5,
        )
        self.assertTrue(torch.equal(augmented_valid, valid))
        self.assertTrue(torch.equal(augmented[:, :3], torch.zeros_like(augmented[:, :3])))

    def test_incremental_freeze_policy_trains_only_history_memory(self) -> None:
        model = nn.Module()
        model.fastwam = nn.Module()
        model.fastwam.target_action_conditioning = self.make_conditioner()
        model.inherited_policy = nn.Linear(3, 3)
        inherited = {name for name, _ in model.named_parameters()}
        _freeze_for_target_conditioning_adapter_training(
            model, inherited, SimpleNamespace()
        )
        trainable = {
            name for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        self.assertTrue(trainable)
        self.assertTrue(
            all(
                "history_memory." in name or "history_adapter." in name
                for name in trainable
            )
        )
        self.assertTrue(
            all(
                not parameter.requires_grad
                for name, parameter in model.named_parameters()
                if name.startswith("inherited_policy.")
            )
        )

if __name__ == "__main__":
    unittest.main()
