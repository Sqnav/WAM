import argparse
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from eval.online_eval_teacher import (
    _current_state_center_for_next_search,
    _load_companion_s0_state,
    _model_driven_initial_bbox,
    _predicted_video_assets_complete,
    _predicted_video_enabled_for_key,
    save_target_crop_action_overlay,
)
from eval.postprocess_online_eval_visuals import postprocess_trajectory


class ModelDrivenSearchTests(unittest.TestCase):
    def test_main_trainable_only_checkpoint_loads_companion_s0_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            s0_path = root / "s0.pt"
            expected = {
                "tracker.weight": torch.ones(1),
                "fastwam.tracker_fusion.weight": torch.ones(1),
                "fastwam.current_target_localizer.weight": torch.ones(1),
            }
            torch.save({"model": expected}, s0_path)
            checkpoint = {
                "model_state_format": "trainable_only",
                "run_args": {
                    "training_stage": "main",
                    "s0_localizer_checkpoint": str(s0_path),
                },
            }

            state, resolved = _load_companion_s0_state(root / "main.pt", checkpoint)

            self.assertEqual(set(state), set(expected))
            self.assertEqual(resolved, s0_path.resolve())

    def test_companion_s0_rejects_non_s0_parameters(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            s0_path = root / "s0.pt"
            torch.save({"model": {"fastwam.video_expert.weight": torch.ones(1)}}, s0_path)
            checkpoint = {
                "model_state_format": "trainable_only",
                "run_args": {
                    "training_stage": "main",
                    "s0_localizer_checkpoint": str(s0_path),
                },
            }

            with self.assertRaisesRegex(ValueError, "non-S0 parameters"):
                _load_companion_s0_state(root / "main.pt", checkpoint)

    def test_main_checkpoint_loads_standalone_tracker_as_frozen_s0(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tracker_path = root / "tracker.pt"
            tracker_state = {
                "template_pos": torch.ones(1),
                "search_pos": torch.ones(1),
                "segment_embed": torch.ones(1),
                "head.weight": torch.ones(1),
            }
            torch.save({"model": tracker_state, "args": {"backbone": "tiny"}}, tracker_path)
            checkpoint = {
                "model_state_format": "trainable_only",
                "run_args": {
                    "training_stage": "main",
                    "s0_localizer_checkpoint": str(tracker_path),
                },
            }

            state, resolved = _load_companion_s0_state(root / "main.pt", checkpoint)

            self.assertEqual(set(state), {f"tracker.{key}" for key in tracker_state})
            self.assertEqual(resolved, tracker_path.resolve())

    def test_initial_bbox_uses_current_online_projection(self) -> None:
        relative = np.array([10.0, 0.0, 0.0], dtype=np.float32)
        actual = _model_driven_initial_bbox(
            relative,
            (640, 640),
            90.0,
            (0.46, 0.0, 0.0),
            0.1,
        )

        self.assertAlmostEqual(actual[0], 287.5)
        self.assertAlmostEqual(actual[1], 287.5)
        self.assertAlmostEqual(actual[2], 64.0)
        self.assertAlmostEqual(actual[3], 64.0)

    def test_next_search_uses_current_state_instead_of_future_state(self) -> None:
        state_centers = torch.tensor([[[0.25, 0.40], [0.80, 0.90]]])

        center = _current_state_center_for_next_search(state_centers)

        np.testing.assert_allclose(center, np.array([0.25, 0.40], dtype=np.float32))

    def test_overlay_draws_state_trajectory_without_search_anchor(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "overlay.png"
            metadata = save_target_crop_action_overlay(
                output,
                np.zeros((640, 640, 3), dtype=np.uint8),
                np.array([10.0, 0.0, 0.0], dtype=np.float32),
                np.zeros((1, 4), dtype=np.float32),
                90.0,
                (0.46, 0.0, 0.0),
                1.0,
                ortrack_bbox_xywh=(288.0, 288.0, 64.0, 64.0),
                ortrack_confidence=1.0,
                model_driven_search_geometry=(192.0, 192.0, 256.0),
                current_state_box_cxcywh=(0.4, 0.5, 0.1, 0.1),
                future_state_boxes_cxcywh=(
                    (0.3, 0.5, 0.1, 0.1),
                    (0.2, 0.4, 0.1, 0.1),
                ),
            )

            target_crop = metadata["target_crop"]
            self.assertTrue(output.is_file())
            self.assertEqual(target_crop["search_crop_xyxy"], [192, 192, 448, 448])
            self.assertEqual(target_crop["current_state_box_xyxy"], [224, 288, 288, 352])
            self.assertEqual(target_crop["future_state_box_xyxy"], [160, 288, 224, 352])
            self.assertEqual(
                target_crop["future_state_boxes_xyxy"],
                [[160, 288, 224, 352], [96, 224, 160, 288]],
            )
            self.assertIsNone(target_crop["ortrack_box_xyxy"])
            self.assertEqual(
                metadata["state_trajectory"]["pixel_points"],
                [[256, 320], [192, 320], [128, 256]],
            )
            self.assertEqual(metadata["state_trajectory"]["num_future_states"], 2)


class PredictedVideoVisualizationTests(unittest.TestCase):
    def test_predicted_video_trajectory_filter_is_independent(self) -> None:
        args = argparse.Namespace(
            visualize_trajectory_keys="all",
            predicted_video_trajectory_keys="City_1/trajectory_0451",
        )

        self.assertTrue(
            _predicted_video_enabled_for_key(args, "City_1", "trajectory_0451")
        )
        self.assertFalse(
            _predicted_video_enabled_for_key(args, "City_1", "trajectory_0452")
        )
        self.assertFalse(
            _predicted_video_enabled_for_key(args, "City_2", "trajectory_0451")
        )

    def test_completion_requires_predicted_rgb_for_every_online_step(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            frame_paths = [
                "predicted_video/frame_00000/pred_000.png",
                "predicted_video/frame_00001/pred_000.png",
            ]
            for relative_path in frame_paths:
                path = out_dir / relative_path
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"png")
            rollout = {
                "steps": [
                    {"predicted_video_frames": [frame_paths[0]]},
                    {"predicted_video_frames": [frame_paths[1]]},
                ]
            }
            (out_dir / "online_rollout.json").write_text(
                json.dumps(rollout), encoding="utf-8"
            )

            self.assertTrue(_predicted_video_assets_complete(out_dir))
            (out_dir / frame_paths[1]).unlink()
            self.assertFalse(_predicted_video_assets_complete(out_dir))

    def test_postprocess_persists_state_trajectory_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trajectory_dir = Path(tmp)
            rgb_dir = trajectory_dir / "rgb"
            rgb_dir.mkdir()
            Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8)).save(
                rgb_dir / "frame_00000.png"
            )
            rollout_path = trajectory_dir / "online_rollout.json"
            rollout_path.write_text(
                json.dumps(
                    {
                        "steps": [
                            {
                                "step": 0,
                                "relative_target_body": [10.0, 0.0, 0.0],
                                "action_sequence_physical": [[0.0, 0.0, 0.0, 0.0]],
                                "current_target_box_cxcywh": [0.5, 0.5, 0.1, 0.1],
                                "future_target_boxes_cxcywh": [
                                    [0.6, 0.5, 0.1, 0.1],
                                    [0.7, 0.5, 0.1, 0.1],
                                ],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            generated, skipped = postprocess_trajectory(
                trajectory_dir,
                output_name="overlays",
                fov_deg=90.0,
                camera_offset_body=[0.46, 0.0, 0.0],
                dt=1.0,
                overwrite=True,
            )

            saved = json.loads(rollout_path.read_text(encoding="utf-8"))
            metadata = saved["steps"][0]["target_crop_action_overlay"]
            self.assertEqual((generated, skipped), (1, 0))
            self.assertEqual(metadata["state_trajectory"]["num_future_states"], 2)
            self.assertTrue(
                (trajectory_dir / metadata["overlay"]).is_file()
            )

    def test_postprocess_adds_predicted_boxes_when_action_overlay_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trajectory_dir = Path(tmp)
            rgb_dir = trajectory_dir / "rgb"
            rgb_dir.mkdir()
            Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8)).save(
                rgb_dir / "frame_00000.png"
            )
            predicted_dir = trajectory_dir / "predicted_video" / "frame_00000"
            predicted_dir.mkdir(parents=True)
            predicted_frames = []
            for index in range(3):
                path = predicted_dir / f"pred_{index:03d}.png"
                Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8)).save(path)
                predicted_frames.append(str(path.relative_to(trajectory_dir)))
            overlay_dir = trajectory_dir / "overlays"
            overlay_dir.mkdir()
            (overlay_dir / "frame_00000_target_crop_action_traj.png").write_bytes(
                b"existing"
            )
            rollout_path = trajectory_dir / "online_rollout.json"
            rollout_path.write_text(
                json.dumps(
                    {
                        "steps": [
                            {
                                "step": 0,
                                "relative_target_body": [10.0, 0.0, 0.0],
                                "action_sequence_physical": [[0.0, 0.0, 0.0, 0.0]],
                                "current_target_box_cxcywh": [0.5, 0.5, 0.2, 0.2],
                                "future_target_boxes_cxcywh": [
                                    [0.6, 0.5, 0.2, 0.2],
                                    [0.7, 0.5, 0.2, 0.2],
                                ],
                                "predicted_video_frames": predicted_frames,
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            generated, skipped = postprocess_trajectory(
                trajectory_dir,
                output_name="overlays",
                fov_deg=90.0,
                camera_offset_body=[0.46, 0.0, 0.0],
                dt=1.0,
                overwrite=False,
            )

            saved = json.loads(rollout_path.read_text(encoding="utf-8"))
            overlay_paths = saved["steps"][0]["predicted_video_state_overlays"]
            self.assertEqual((generated, skipped), (1, 0))
            self.assertEqual(len(overlay_paths), 3)
            self.assertTrue(all((trajectory_dir / path).is_file() for path in overlay_paths))


if __name__ == "__main__":
    unittest.main()
