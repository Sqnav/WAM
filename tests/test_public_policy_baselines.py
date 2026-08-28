import numpy as np

from eval.public_policy_baselines import RandomActionPolicy


def test_random_action_policy_is_step_deterministic_and_bounded():
    policy = RandomActionPolicy(seed=42, horizon=8)
    kwargs = {
        "scene_id": "City_3",
        "trajectory_name": "trajectory_0451",
        "step": 7,
    }
    first = policy.infer(**kwargs)
    second = policy.infer(**kwargs)

    assert first.shape == (8, 4)
    np.testing.assert_array_equal(first, second)
    assert np.all(np.linalg.norm(first[:, :3], axis=-1) <= 1.0 + 1.0e-6)
    assert np.all(np.abs(first[:, 3]) <= 1.0)


def test_random_action_policy_changes_across_trajectory_and_step():
    policy = RandomActionPolicy(seed=42, horizon=8)
    reference = policy.infer(scene_id="City_1", trajectory_name="trajectory_0451", step=0)
    next_step = policy.infer(scene_id="City_1", trajectory_name="trajectory_0451", step=1)
    next_trajectory = policy.infer(scene_id="City_1", trajectory_name="trajectory_0452", step=0)

    assert not np.array_equal(reference, next_step)
    assert not np.array_equal(reference, next_trajectory)
