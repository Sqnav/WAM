import torch

from train.train_teacher import TrajectoryDataset, collate_fn


def _sample(value: float, video_latents=None):
    sample = {
        "images": torch.full((2, 3, 4, 4), value),
        "target_relative": torch.full((2, 3), value),
    }
    if video_latents is not None:
        sample["video_latents"] = video_latents
    return sample


def test_collate_mixed_wan_cache_hits_falls_back_to_real_rgb():
    cached = _sample(1.0, torch.ones(2, 4, 2, 2))
    uncached = _sample(2.0)

    batch = collate_fn([cached, uncached])

    assert batch["video_latents"] is None
    assert torch.equal(batch["images"][0], cached["images"])
    assert torch.equal(batch["images"][1], uncached["images"])


def test_collate_all_wan_cache_hits_stacks_latents():
    first = _sample(1.0, torch.ones(2, 4, 2, 2))
    second = _sample(2.0, torch.full((2, 4, 2, 2), 2.0))

    batch = collate_fn([first, second])

    assert batch["video_latents"].shape == (2, 2, 4, 2, 2)
    assert torch.equal(batch["video_latents"][0], first["video_latents"])


def _model_driven_dataset(seq_len: int = 3):
    return TrajectoryDataset(
        records=[],
        image_size=32,
        seq_len=seq_len,
        target_relative_dim=3,
        action_dim=4,
        random_crop=False,
        use_model_driven_tracker_crops=True,
    )


def test_model_driven_tracker_starts_at_first_valid_box():
    dataset = _model_driven_dataset()
    record = {
        "target_bboxes_xywh": [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [10.0, 12.0, 20.0, 18.0],
            [11.0, 13.0, 20.0, 18.0],
            [12.0, 14.0, 20.0, 18.0],
        ],
        "target_bbox_valid": [0.0, 0.0, 1.0, 1.0, 1.0],
    }

    init_index = dataset._first_valid_tracker_frame(record, 5, 0)

    assert init_index == 2
    assert dataset._select_window(5, init_index) == (2, 5)


def test_model_driven_tracker_rejects_trajectory_without_valid_box():
    dataset = _model_driven_dataset()
    record = {
        "target_bboxes_xywh": [[0.0, 0.0, 0.0, 0.0]] * 4,
        "target_bbox_valid": [0.0] * 4,
    }

    try:
        dataset._first_valid_tracker_frame(record, 4, 7)
    except ValueError as exc:
        assert "no valid Tracker initialization box" in str(exc)
    else:
        raise AssertionError("Expected an all-invalid trajectory to be rejected.")
