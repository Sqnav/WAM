from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch

from model.tracker_fusion import (
    BoxFreeFutureTargetReadout,
    FrozenTrackerConditionFusion,
    FutureStateConditioner,
    LocalFeatureTrackerConditionFusion,
)
from model.config import ModelConfig, migrate_legacy_config
from model.losses import world_model_dit_loss
from model.model import TeacherWorldModelDiT
from model.state_dit import FutureStateDiT
from tracking.model import UAVTracker


class FutureStateDiTTests(unittest.TestCase):
    def test_current_tracking_losses_penalize_uniform_attention(self) -> None:
        grid = torch.linspace(0.2, 0.8, 16)
        grid_y, grid_x = torch.meshgrid(grid, grid, indexing="ij")
        full_xy = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=-1).unsqueeze(0)
        attention = torch.full((1, 1, 256), 1.0 / 256.0, requires_grad=True)
        current_box = torch.tensor([[0.5, 0.5, 0.1, 0.1]], requires_grad=True)
        target_box = torch.tensor([[0.3, 0.4, 0.1, 0.1]])

        losses = FutureStateConditioner.current_tracking_losses(
            current_box=current_box,
            current_attention=attention,
            full_xy=full_xy,
            target_box=target_box,
            valid=torch.ones(1),
            image_size=torch.tensor([[640.0, 640.0]]),
            attention_sigma=1.5,
        )

        self.assertGreater(losses["center"].item(), 0.0)
        self.assertGreater(losses["giou"].item(), 0.0)
        self.assertGreater(losses["attention"].item(), 0.0)
        self.assertGreater(losses["center_error_pixels"].item(), 100.0)
        (losses["center"] + losses["giou"] + losses["attention"]).backward()
        self.assertGreater(current_box.grad.abs().sum().item(), 0.0)
        self.assertGreater(attention.grad.abs().sum().item(), 0.0)

    def test_state_dit_forward_shape_and_gradients(self) -> None:
        model = FutureStateDiT(
            state_dim=4,
            hidden_dim=16,
            ffn_dim=32,
            freq_dim=8,
            num_heads=2,
            attn_head_dim=8,
            num_layers=2,
            horizon=8,
            eps=1.0e-6,
            use_gradient_checkpointing=False,
        )
        noisy = torch.randn(2, 8, 4, requires_grad=True)
        current = torch.randn(2, 16, requires_grad=True)
        memory = torch.randn(2, 320, 16, requires_grad=True)
        pre = model.pre_dit(
            noisy_future_states=noisy,
            current_condition=current,
            timestep=torch.tensor([250.0, 750.0]),
            tracker_memory=memory,
            state_valid_mask=torch.ones(2, 9, dtype=torch.bool),
        )
        hidden = pre["tokens"]
        self.assertEqual(tuple(hidden.shape), (2, 9, 16))
        self_mask = torch.ones(2, 1, 9, 9, dtype=torch.bool)
        for block in model.blocks:
            hidden = block(
                hidden,
                pre["context"],
                pre["t_mod"],
                pre["freqs"],
                context_mask=pre["context_mask"],
                self_attn_mask=self_mask,
            )
        flow = model.post_dit(hidden, pre)
        self.assertEqual(tuple(flow.shape), (2, 8, 4))
        self.assertTrue(torch.isfinite(flow).all())
        flow.square().mean().backward()
        self.assertGreater(noisy.grad.abs().sum().item(), 0.0)
        self.assertGreater(current.grad.abs().sum().item(), 0.0)
        self.assertGreater(memory.grad.abs().sum().item(), 0.0)

    def test_relative_state_round_trip(self) -> None:
        current = torch.tensor([[0.4, 0.5, 0.2, 0.1]])
        future = torch.tensor(
            [[[0.42, 0.48, 0.22, 0.09], [0.45, 0.46, 0.18, 0.12]]]
        )
        boxes = torch.cat([current.unsqueeze(1), future], dim=1)
        relative = FutureStateConditioner.relative_states(boxes)
        decoded = FutureStateConditioner.decode_relative_states(current, relative)
        torch.testing.assert_close(decoded, boxes)

    def test_v4_total_uses_exactly_four_losses(self) -> None:
        cfg = ModelConfig(
            use_future_state_dit=True,
            use_fastwam_mot=True,
            train_direct_action=True,
            fastwam_lambda_action=1.0,
            fastwam_lambda_video=1.0,
            future_state_flow_weight=1.0,
            current_box_weight=1.0,
            tracker_center_flow_supervision=True,
            tracker_center_flow_loss_weight=100.0,
        )
        outputs = {
            "policy_flow_loss": torch.tensor(1.0),
            "video_flow_loss": torch.tensor(2.0),
            "state_flow_loss": torch.tensor(3.0),
            "current_box_loss": torch.tensor(4.0),
            "current_center_spatial_loss": torch.tensor(0.0),
            "current_box_giou_loss": torch.tensor(0.0),
            "current_attention_loss": torch.tensor(0.0),
            "center_flow_loss": torch.tensor(1000.0),
        }
        batch = {"expert_action": torch.zeros(1, 9, 4)}
        losses = world_model_dit_loss(
            outputs, batch, cfg, valid_mask=torch.ones(1, 9)
        )
        torch.testing.assert_close(losses["total"], torch.tensor(10.0))
        torch.testing.assert_close(losses["total_loss"], torch.tensor(10.0))
        torch.testing.assert_close(losses["action_flow_loss"], torch.tensor(1.0))
        torch.testing.assert_close(losses["video_flow_loss"], torch.tensor(2.0))
        torch.testing.assert_close(losses["state_flow_loss"], torch.tensor(3.0))
        torch.testing.assert_close(losses["current_box_loss"], torch.tensor(4.0))

    def test_v4_spatial_losses_and_localization_warmup(self) -> None:
        cfg = ModelConfig(
            use_future_state_dit=True,
            current_box_weight=5.0,
            current_center_weight=5.0,
            current_box_giou_weight=2.0,
            current_attention_weight=1.0,
        )
        outputs = {
            "policy_flow_loss": torch.tensor(1.0),
            "video_flow_loss": torch.tensor(2.0),
            "state_flow_loss": torch.tensor(3.0),
            "current_box_loss": torch.tensor(4.0),
            "current_center_spatial_loss": torch.tensor(5.0),
            "current_box_giou_loss": torch.tensor(6.0),
            "current_attention_loss": torch.tensor(7.0),
        }
        batch = {"expert_action": torch.zeros(1, 9, 4)}

        joint = world_model_dit_loss(outputs, batch, cfg)
        warmup = world_model_dit_loss(
            outputs, batch, cfg, localization_only=True
        )

        torch.testing.assert_close(joint["localization_total"], torch.tensor(64.0))
        torch.testing.assert_close(joint["total"], torch.tensor(70.0))
        torch.testing.assert_close(warmup["total"], torch.tensor(64.0))
        torch.testing.assert_close(
            warmup["localization_warmup_active"], torch.tensor(1.0)
        )


class FrozenTrackerFusionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fusion = FrozenTrackerConditionFusion(
            tracker_dim=6,
            action_dim=8,
            num_heads=2,
            head_dim=4,
            num_layers=4,
            start_layer=2,
        )

    def test_condition_has_feature_and_center_tokens(self) -> None:
        features = torch.randn(3, 49, 6)
        center = torch.tensor([[0.0, 0.5], [0.5, 0.5], [1.0, 1.0]])
        condition = self.fusion.make_condition(features, center)
        self.assertEqual(tuple(condition.shape), (3, 50, 8))

    def test_zero_gates_preserve_base_attention(self) -> None:
        action = torch.randn(2, 5, 8)
        condition = self.fusion.make_condition(
            torch.randn(2, 49, 6), torch.rand(2, 2)
        )
        for layer_index in range(4):
            torch.testing.assert_close(
                self.fusion.delta(layer_index, action, condition),
                torch.zeros_like(action),
            )

    def test_only_late_layers_can_add_tracker_delta(self) -> None:
        action = torch.randn(2, 5, 8)
        condition = self.fusion.make_condition(
            torch.randn(2, 49, 6), torch.rand(2, 2)
        )
        self.fusion.layers["2"].gate.data.fill_(1.0)
        torch.testing.assert_close(
            self.fusion.delta(1, action, condition), torch.zeros_like(action)
        )
        self.assertGreater(self.fusion.delta(2, action, condition).abs().sum().item(), 0.0)

    def test_tracker_inputs_are_detached(self) -> None:
        features = torch.randn(2, 49, 6, requires_grad=True)
        center = torch.rand(2, 2, requires_grad=True)
        condition = self.fusion.make_condition(features, center)
        condition.sum().backward()
        self.assertIsNone(features.grad)
        self.assertIsNone(center.grad)

    def test_all_condition_modes_have_expected_token_count(self) -> None:
        expected_tokens = {
            "center": 1,
            "bbox": 1,
            "features": 49,
            "response": 49,
            "center_features": 50,
            "bbox_response": 50,
            "bbox_response_features": 99,
        }
        inputs = {
            "tracker_features": torch.randn(2, 49, 6),
            "tracker_center": torch.rand(2, 2),
            "tracker_bbox": torch.rand(2, 4),
            "tracker_response": torch.rand(2, 7, 7),
        }
        for mode, count in expected_tokens.items():
            with self.subTest(mode=mode):
                fusion = FrozenTrackerConditionFusion(
                    tracker_dim=6,
                    action_dim=8,
                    num_heads=2,
                    head_dim=4,
                    num_layers=4,
                    start_layer=2,
                    condition_mode=mode,
                )
                condition = fusion.make_condition(**inputs)
                self.assertEqual(tuple(condition.shape), (2, count, 8))

    def test_gate_initialization_is_applied_to_every_late_layer(self) -> None:
        fusion = FrozenTrackerConditionFusion(
            tracker_dim=6,
            action_dim=8,
            num_heads=2,
            head_dim=4,
            num_layers=5,
            start_layer=2,
            gate_init=0.05,
        )
        self.assertEqual(len(fusion.layers), 3)
        for layer in fusion.layers.values():
            self.assertAlmostEqual(float(layer.gate.detach()), 0.05, places=6)

    def test_bbox_and_response_are_detached_and_normalized(self) -> None:
        fusion = FrozenTrackerConditionFusion(
            tracker_dim=6,
            action_dim=8,
            num_heads=2,
            head_dim=4,
            num_layers=4,
            start_layer=2,
            condition_mode="bbox_response",
        )
        bbox = torch.rand(2, 4, requires_grad=True)
        response = torch.rand(2, 7, 7, requires_grad=True)
        condition = fusion.make_condition(
            tracker_bbox=bbox, tracker_response=response
        )
        self.assertTrue(torch.isfinite(condition).all())
        condition.sum().backward()
        self.assertIsNone(bbox.grad)
        self.assertIsNone(response.grad)

    def test_required_inputs_are_validated(self) -> None:
        fusion = FrozenTrackerConditionFusion(
            tracker_dim=6,
            action_dim=8,
            num_heads=2,
            head_dim=4,
            num_layers=4,
            start_layer=2,
            condition_mode="response",
        )
        with self.assertRaisesRegex(ValueError, "requires tracker_response"):
            fusion.make_condition()
        with self.assertRaisesRegex(ValueError, "tracker_response must have shape"):
            fusion.make_condition(tracker_response=torch.rand(2, 8, 8))


class LocalFeatureTrackerFusionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fusion = LocalFeatureTrackerConditionFusion(
            tracker_dim=6,
            action_dim=8,
            num_heads=2,
            head_dim=4,
            num_layers=4,
            start_layer=2,
            grid_size=16,
            gate_init=0.25,
        )

    def _inputs(self, batch: int = 2) -> tuple[torch.Tensor, ...]:
        return (
            torch.randn(batch, 256, 6),
            torch.rand(batch, 4),
            torch.tensor([[10.0, 20.0, 120.0]]).expand(batch, -1).clone(),
            torch.tensor([[640.0, 640.0]]).expand(batch, -1).clone(),
        )

    def test_condition_shape_and_finite_values(self) -> None:
        features, bbox, geometry, image_size = self._inputs()
        features[0, 0, 0] = float("nan")
        condition = self.fusion.make_condition(
            features, bbox, geometry, image_size
        )
        self.assertEqual(tuple(condition.shape), (2, 257, 8))
        self.assertTrue(torch.isfinite(condition).all())

    def test_joint_finetune_inputs_retain_gradients(self) -> None:
        fusion = LocalFeatureTrackerConditionFusion(
            tracker_dim=6, action_dim=8, num_heads=2, head_dim=4,
            num_layers=4, start_layer=2, grid_size=16,
            detach_tracker_inputs=False,
        )
        features, bbox, geometry, image_size = self._inputs()
        features.requires_grad_()
        bbox.requires_grad_()
        fusion.make_condition(features, bbox, geometry, image_size).square().sum().backward()
        self.assertIsNotNone(features.grad)
        self.assertGreater(features.grad.abs().sum().item(), 0.0)
        self.assertIsNotNone(bbox.grad)
        self.assertGreater(bbox.grad.abs().sum().item(), 0.0)

    def test_full_image_coordinate_mapping(self) -> None:
        fusion = LocalFeatureTrackerConditionFusion(
            tracker_dim=3,
            action_dim=4,
            num_heads=1,
            head_dim=4,
            num_layers=2,
            start_layer=1,
            grid_size=2,
        )
        coordinates = fusion.full_image_coordinates(
            torch.tensor([[10.0, 20.0, 40.0]]),
            torch.tensor([[100.0, 200.0]]),
        )
        expected = torch.tensor(
            [[[0.10, 0.30], [0.20, 0.30], [0.10, 0.50], [0.20, 0.50]]]
        )
        torch.testing.assert_close(coordinates, expected)
        spatial_geometry = fusion.full_image_geometry(
            torch.tensor([[10.0, 20.0, 40.0]]),
            torch.tensor([[100.0, 200.0]]),
        )
        torch.testing.assert_close(spatial_geometry[..., :2], expected)
        torch.testing.assert_close(
            spatial_geometry[..., 2],
            torch.full((1, 4), 40.0 / (100.0 * 200.0) ** 0.5),
        )

    def test_local_position_embedding_is_optional(self) -> None:
        fusion = LocalFeatureTrackerConditionFusion(
            tracker_dim=6,
            action_dim=8,
            num_heads=2,
            head_dim=4,
            num_layers=4,
            start_layer=2,
            grid_size=16,
            use_local_position_embedding=False,
        )
        self.assertIsNone(fusion.local_position_embedding)
        features, bbox, geometry, image_size = self._inputs()
        self.assertEqual(
            tuple(fusion.make_condition(features, bbox, geometry, image_size).shape),
            (2, 257, 8),
        )

    def test_spatial_only_condition_has_no_bbox_token(self) -> None:
        fusion = LocalFeatureTrackerConditionFusion(
            tracker_dim=6, action_dim=8, num_heads=2, head_dim=4,
            num_layers=4, start_layer=0, grid_size=16, include_box_token=False,
        )
        features, _, geometry, image_size = self._inputs()
        condition = fusion.make_condition(features, None, geometry, image_size)
        self.assertEqual(tuple(condition.shape), (2, 256, 8))
        condition.sum().backward()
        self.assertIsNotNone(fusion.feature_projection[1].weight.grad)

    def test_zero_initialized_gates_preserve_original_mot_output(self) -> None:
        fusion = LocalFeatureTrackerConditionFusion(
            tracker_dim=6,
            action_dim=8,
            num_heads=2,
            head_dim=4,
            num_layers=4,
            start_layer=2,
            grid_size=16,
        )
        features, bbox, geometry, image_size = self._inputs()
        condition = fusion.make_condition(features, bbox, geometry, image_size)
        action = torch.randn(2, 8, 8)
        for layer_index in range(4):
            torch.testing.assert_close(
                fusion.delta(layer_index, action, condition), torch.zeros_like(action)
            )


    def test_only_layers_18_to_29_equivalent_range_receive_fusion(self) -> None:
        self.assertEqual(set(self.fusion.layers), {"2", "3"})
        features, bbox, geometry, image_size = self._inputs()
        condition = self.fusion.make_condition(features, bbox, geometry, image_size)
        action = torch.randn(2, 8, 8)
        torch.testing.assert_close(
            self.fusion.delta(1, action, condition), torch.zeros_like(action)
        )
        self.assertGreater(self.fusion.delta(2, action, condition).abs().sum().item(), 0.0)

    def test_backward_updates_new_modules_but_not_tracker_inputs(self) -> None:
        features, bbox, geometry, image_size = self._inputs()
        features.requires_grad_()
        bbox.requires_grad_()
        geometry.requires_grad_()
        image_size.requires_grad_()
        action = torch.randn(2, 8, 8, requires_grad=True)
        condition = self.fusion.make_condition(features, bbox, geometry, image_size)
        output = action + self.fusion.delta(2, action, condition)
        output.square().mean().backward()

        self.assertIsNone(features.grad)
        self.assertIsNone(bbox.grad)
        self.assertIsNone(geometry.grad)
        self.assertIsNone(image_size.grad)
        self.assertGreater(
            self.fusion.feature_projection[1].weight.grad.abs().sum().item(), 0.0
        )
        self.assertGreater(
            self.fusion.full_image_geometry_embedding[0].weight.grad.abs().sum().item(),
            0.0,
        )
        self.assertGreater(self.fusion.box_mlp[0].weight.grad.abs().sum().item(), 0.0)
        self.assertGreater(self.fusion.layers["2"].query.weight.grad.abs().sum().item(), 0.0)
        self.assertGreater(self.fusion.layers["2"].gate.grad.abs().item(), 0.0)

    def test_all_action_queries_read_the_shared_tracker_condition(self) -> None:
        features, bbox, geometry, image_size = self._inputs(batch=1)
        condition = self.fusion.make_condition(features, bbox, geometry, image_size)
        action = torch.randn(1, 8, 8)
        baseline = self.fusion.delta(2, action, condition)
        changed = condition.clone()
        changed[:, 0] += 1.0
        perturbed = self.fusion.delta(2, action, changed)
        per_query_change = (perturbed - baseline).abs().sum(dim=-1)
        self.assertTrue((per_query_change > 0).all())

    def test_training_window_uses_only_current_tracker_frame(self) -> None:
        holder = SimpleNamespace(
            cfg=SimpleNamespace(
                tracker_mot_integration="frozen_deit_tracker_local_feature",
                tracker_condition_mode="none",
                tracker_feature_grid_size=16,
                tracker_feature_dim=6,
            )
        )
        features = torch.zeros(2, 9, 256, 6)
        features[:, 0].fill_(1.0)
        features[:, 1:].fill_(9.0)
        selected, _ = TeacherWorldModelDiT._first_tracker_features(
            holder, features, None
        )
        torch.testing.assert_close(selected, torch.ones(2, 256, 6))

        geometry = torch.zeros(2, 9, 3)
        geometry[:, 0] = torch.tensor([10.0, 20.0, 30.0])
        geometry[:, 1:] = 999.0
        selected_geometry = TeacherWorldModelDiT._first_tracker_tensor(
            geometry, unbatched_ndim=1, name="tracker_search_geometry"
        )
        torch.testing.assert_close(
            selected_geometry,
            torch.tensor([[10.0, 20.0, 30.0]]).expand(2, -1),
        )


class JointTrackerSpatialOnlyTests(unittest.TestCase):
    def test_disabled_detection_head_registers_no_head_parameters(self) -> None:
        tracker = UAVTracker(
            pretrained=False,
            template_size=128,
            search_size=256,
            square_boxes=True,
            enable_head=False,
        )
        self.assertIsNone(tracker.head)
        self.assertFalse(any(name.startswith("head.") for name, _ in tracker.named_parameters()))
        output = tracker(
            torch.randn(1, 3, 128, 128),
            torch.randn(1, 3, 256, 256),
            return_head=False,
        )
        self.assertNotIn("center_logits", output)
        with self.assertRaisesRegex(RuntimeError, "detection head is disabled"):
            tracker(
                torch.randn(1, 3, 128, 128),
                torch.randn(1, 3, 256, 256),
                return_head=True,
            )

    def test_spatial_only_tracker_forward_skips_detection_head(self) -> None:
        tracker = UAVTracker(
            pretrained=False,
            template_size=128,
            search_size=256,
            square_boxes=True,
        )
        output = tracker(
            torch.randn(1, 3, 128, 128),
            torch.randn(1, 3, 256, 256),
            return_head=False,
        )
        self.assertEqual(tuple(output["search_features"].shape), (1, 192, 16, 16))
        self.assertEqual(tuple(output["template_features"].shape), (1, 64, 192))
        self.assertNotIn("center_logits", output)
        output["search_features"].square().mean().backward()
        self.assertIsNotNone(tracker.patch_embed.proj.weight.grad)
        self.assertIsNone(tracker.head.center.weight.grad)


class BoxFreeFutureTargetReadoutTests(unittest.TestCase):
    def test_legacy_future_checkpoint_is_marked_as_alignment_v1(self) -> None:
        migrated = migrate_legacy_config({"tracker_future_target_alignment": True})
        self.assertEqual(migrated["tracker_state_action_alignment_version"], 1)
        current = migrate_legacy_config({
            "tracker_future_target_alignment": True,
            "tracker_state_action_alignment_version": 3,
        })
        self.assertEqual(current["tracker_state_action_alignment_version"], 3)

    def test_box_free_mode_has_no_action_to_spatial_layers(self) -> None:
        fusion = LocalFeatureTrackerConditionFusion(
            tracker_dim=6, action_dim=8, num_heads=2, head_dim=4,
            num_layers=4, start_layer=0, grid_size=16,
            include_box_token=False, enable_cross_attention=False,
        )
        self.assertEqual(len(fusion.layers), 0)
        features = torch.randn(1, 256, 6)
        condition = fusion.make_condition(
            features, None, torch.tensor([[0.0, 0.0, 128.0]]), torch.tensor([[640.0, 640.0]])
        )
        torch.testing.assert_close(
            fusion.delta(0, torch.randn(1, 8, 8), condition),
            torch.zeros(1, 8, 8),
        )

    def test_template_guided_future_readout_has_expected_shapes_and_gradients(self) -> None:
        spatial = LocalFeatureTrackerConditionFusion(
            tracker_dim=6, action_dim=8, num_heads=2, head_dim=4,
            num_layers=4, start_layer=0, grid_size=16,
            include_box_token=False, detach_tracker_inputs=False,
        )
        readout = BoxFreeFutureTargetReadout(
            tracker_dim=6, action_dim=8, video_dim=12,
            num_layers=4, start_layer=2, action_horizon=8,
        )
        features = torch.randn(2, 256, 6, requires_grad=True)
        template = torch.randn(2, 64, 6, requires_grad=True)
        geometry = torch.tensor([[10.0, 20.0, 120.0], [5.0, 8.0, 160.0]])
        image_size = torch.full((2, 2), 640.0)
        condition = spatial.make_condition(features, None, geometry, image_size)
        state = readout.make_target_state(
            template, condition, spatial.full_image_coordinates(geometry, image_size)
        )
        action = torch.randn(2, 8, 8, requires_grad=True)
        video = torch.randn(2, 12, 12, requires_grad=True)
        timestep = torch.rand(2)
        delta, future_tokens = readout.delta(
            2, action, video, state, timestep, return_tokens=True
        )
        action_states = readout.action_aligned_state_tokens(future_tokens, state)
        boxes, center_flow, box_offsets = readout.state_boxes(future_tokens, state)
        centers = boxes[..., :2]
        self.assertEqual(tuple(state["attention"].shape), (2, 1, 256))
        self.assertEqual(tuple(delta.shape), (2, 8, 8))
        self.assertEqual(tuple(future_tokens.shape), (2, 8, 8))
        self.assertEqual(tuple(action_states.shape), (2, 8, 8))
        self.assertEqual(tuple(center_flow.shape), (2, 8, 2))
        self.assertEqual(tuple(centers.shape), (2, 9, 2))
        self.assertEqual(tuple(boxes.shape), (2, 9, 4))
        self.assertEqual(tuple(box_offsets.shape), (2, 8, 4))
        torch.testing.assert_close(action_states[:, :1], state["current_token"])
        torch.testing.assert_close(action_states[:, 1:], future_tokens[:, :-1])
        self.assertTrue(torch.isfinite(delta).all())
        (delta.square().mean() + boxes.square().mean() + state["current_token"].square().mean()).backward()
        self.assertGreater(features.grad.abs().sum().item(), 0.0)
        self.assertGreater(template.grad.abs().sum().item(), 0.0)
        self.assertIsNotNone(readout.layers["2"].gate.grad)
        self.assertIsNotNone(readout.current_size_head[-1].weight.grad)
        self.assertIsNotNone(readout.box_offset_head[-1].weight.grad)

    def test_state_sequence_contains_current_and_eight_post_action_states(self) -> None:
        readout = BoxFreeFutureTargetReadout(
            tracker_dim=8, action_dim=8, video_dim=8,
            num_layers=1, start_layer=0, action_horizon=8,
        )
        readout.box_offset_head[-1].weight.data.zero_()
        readout.box_offset_head[-1].bias.data.copy_(torch.tensor([0.1, -0.05, 0.0, 0.0]))
        transition_tokens = torch.randn(1, 8, 8)
        state = {
            "current_token": torch.full((1, 1, 8), 7.0),
            "soft_center": torch.tensor([[0.2, 0.8]]),
            "current_size": torch.tensor([[0.1, 0.2]]),
        }

        action_states = readout.action_aligned_state_tokens(transition_tokens, state)
        state_centers, center_flow = readout.state_centers(transition_tokens, state)

        torch.testing.assert_close(action_states[:, 0], state["current_token"][:, 0])
        torch.testing.assert_close(action_states[:, 1:], transition_tokens[:, :-1])
        self.assertFalse(torch.equal(action_states[:, -1], transition_tokens[:, -1]))
        expected = torch.cat([
            state["soft_center"].unsqueeze(1),
            state["soft_center"].unsqueeze(1)
            + torch.tensor([0.1, -0.05]).view(1, 1, 2).expand(1, 8, 2),
        ], dim=1)
        torch.testing.assert_close(state_centers, expected)
        expected_flow = torch.zeros(1, 8, 2)
        expected_flow[:, 0] = torch.tensor([0.1, -0.05])
        torch.testing.assert_close(center_flow, expected_flow)

    def test_box_losses_are_zero_for_matching_boxes(self) -> None:
        readout = BoxFreeFutureTargetReadout(
            tracker_dim=8, action_dim=8, video_dim=8,
            num_layers=1, start_layer=0, action_horizon=2,
        )
        boxes = torch.tensor([[[0.2, 0.3, 0.1, 0.2], [0.3, 0.4, 0.1, 0.2], [0.4, 0.5, 0.2, 0.1]]])
        losses = readout.box_state_losses(
            boxes,
            boxes.clone(),
            target_box_valid=torch.ones(1, 3, 1),
            sequence_valid=torch.ones(1, 3),
            horizon_discount=0.9,
        )
        torch.testing.assert_close(losses["l1"], torch.tensor(0.0))
        torch.testing.assert_close(losses["giou"], torch.tensor(0.0), atol=1.0e-6, rtol=0.0)

    def test_state_losses_ignore_invalid_padded_tail(self) -> None:
        readout = BoxFreeFutureTargetReadout(
            tracker_dim=8, action_dim=8, video_dim=8,
            num_layers=1, start_layer=0, action_horizon=2,
        )
        target = torch.tensor([[[0.1, 0.2], [0.2, 0.3], [0.3, 0.4]]])
        pred_states = target.clone()
        pred_states[:, 2] = 100.0
        pred_flow = torch.tensor([[[0.1, 0.1], [100.0, 100.0]]])
        losses = readout.center_state_losses(
            pred_states,
            pred_flow,
            target,
            target_center_valid=torch.ones(1, 3, 1),
            sequence_valid=torch.tensor([[1.0, 1.0, 0.0]]),
            horizon_discount=0.9,
        )
        torch.testing.assert_close(losses["current"], torch.tensor(0.0))
        torch.testing.assert_close(losses["future"], torch.tensor(0.0))
        torch.testing.assert_close(losses["transition"], torch.tensor(0.0))


if __name__ == "__main__":
    unittest.main()
