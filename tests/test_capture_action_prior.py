import unittest
import tempfile

import torch

from model.capture_action_prior import (
    CAPTURE_ACTION_PRIOR_VERSION,
    CaptureActionPrior,
    featurize_capture_action_context,
    load_capture_action_prior,
    score_candidates_with_action_prior,
)


class CaptureActionPriorTests(unittest.TestCase):
    def test_featurizer_and_prior_shapes(self) -> None:
        history = torch.rand(2, 7, 5)
        valid = torch.ones(2, 7, dtype=torch.bool)
        current = torch.rand(2, 4)
        confidence = torch.rand(2)
        previous_action = torch.rand(2, 4)
        features = featurize_capture_action_context(
            history, valid, current, confidence, previous_action
        )
        self.assertEqual(features.shape, (2, 80))
        model = CaptureActionPrior(history_length=8, hidden_dim=32, dropout=0.0)
        prediction = model(features)
        self.assertEqual(prediction["mean"].shape, (2, 4))
        self.assertTrue(torch.all(prediction["std"] >= 0.05))

    def test_candidate_score_prefers_prior_mean(self) -> None:
        candidates = torch.zeros(1, 3, 8, 4)
        candidates[0, 0, 0] = torch.tensor([0.8, 0.2, -0.1, 0.3])
        candidates[0, 1, 0] = torch.tensor([0.8, 0.2, -0.1, -0.3])
        candidates[0, 2, 0] = torch.tensor([0.4, 0.2, -0.1, 0.3])
        prediction = {
            "mean": torch.tensor([[0.8, 0.2, -0.1, 0.3]]),
            "std": torch.full((1, 4), 0.2),
        }
        scores = score_candidates_with_action_prior(candidates, prediction)
        self.assertEqual(scores.argmax(dim=1).item(), 0)

    def test_checkpoint_load_preserves_global_rng(self) -> None:
        model = CaptureActionPrior(history_length=8, hidden_dim=32, dropout=0.0)
        checkpoint = {
            "metadata": {
                "version": CAPTURE_ACTION_PRIOR_VERSION,
                "history_length": 8,
                "hidden_dim": 32,
            },
            "model": model.state_dict(),
        }
        with tempfile.NamedTemporaryFile(suffix=".pt") as handle:
            torch.save(checkpoint, handle.name)
            torch.manual_seed(123)
            expected = torch.randn(8)
            torch.manual_seed(123)
            load_capture_action_prior(
                handle.name, device=torch.device("cpu"), dtype=torch.float32
            )
            actual = torch.randn(8)
        self.assertTrue(torch.equal(actual, expected))


if __name__ == "__main__":
    unittest.main()
