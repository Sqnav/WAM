import unittest
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "model/FastWAM/src"))

from model.fasterwam_dot import FasterWAMActionHead, FasterWAMDoT


class FasterWAMDoTConditioningTests(unittest.TestCase):
    def _modules(self):
        action_head = FasterWAMActionHead(
            action_dim=4,
            hidden_dim=8,
            ffn_dim=16,
            freq_dim=8,
            num_heads=2,
            attn_head_dim=4,
            use_gradient_checkpointing=True,
        )
        dot = FasterWAMDoT(
            video_num_layers=2,
            num_action_layers=1,
            num_heads=2,
            attn_head_dim=4,
            use_gradient_checkpointing=True,
        )
        return action_head, dot

    def test_condition_residual_changes_action_and_receives_gradient(self):
        torch.manual_seed(7)
        action_head, dot = self._modules()
        action_head.train()
        dot.train()
        action = torch.randn(2, 8, 4)
        timestep = torch.full((2,), 0.5)
        action_pre = action_head.pre_dit(action, timestep)
        fused_video_cache = {
            "canonical_key": torch.randn(1, 2, 3, 8),
            "value": torch.randn(1, 2, 3, 8),
        }
        scale = torch.nn.Parameter(torch.tensor(0.25))

        conditioned = dot.forward_action(
            action_expert=action_head,
            action_pre=action_pre,
            fused_video_cache=fused_video_cache,
            condition_residual=lambda hidden: scale * torch.tanh(hidden),
        )
        baseline = dot.forward_action(
            action_expert=action_head,
            action_pre=action_pre,
            fused_video_cache=fused_video_cache,
        )
        self.assertEqual(conditioned.shape, baseline.shape)
        self.assertFalse(torch.allclose(conditioned, baseline))

        conditioned.square().mean().backward()
        self.assertIsNotNone(scale.grad)
        self.assertGreater(float(scale.grad.abs()), 0.0)

    def test_action_candidates_share_smaller_video_cache_batch(self):
        torch.manual_seed(11)
        action_head, dot = self._modules()
        action_head.eval()
        dot.eval()
        batch_size = 2
        candidate_count = 4
        action = torch.randn(batch_size * candidate_count, 8, 4)
        timestep = torch.full((batch_size * candidate_count,), 0.5)
        action_pre = action_head.pre_dit(action, timestep)
        fused_video_cache = {
            "canonical_key": torch.randn(1, batch_size, 3, 8),
            "value": torch.randn(1, batch_size, 3, 8),
        }

        output = dot.forward_action(
            action_expert=action_head,
            action_pre=action_pre,
            fused_video_cache=fused_video_cache,
        )
        output_with_attention, attention = dot.forward_action(
            action_expert=action_head,
            action_pre=action_pre,
            fused_video_cache=fused_video_cache,
            return_attention=True,
        )

        self.assertEqual(output.shape, (batch_size * candidate_count, 8, 8))
        self.assertTrue(torch.equal(output_with_attention, output))
        self.assertEqual(
            attention.shape,
            (batch_size * candidate_count, 2, 8, 3),
        )

    def test_attention_export_matches_action_output(self):
        torch.manual_seed(17)
        action_head, dot = self._modules()
        action_head.eval()
        dot.eval()
        action = torch.randn(2, 8, 4)
        timestep = torch.full((2,), 0.5)
        action_pre = action_head.pre_dit(action, timestep)
        fused_video_cache = {
            "canonical_key": torch.randn(1, 2, 3, 8),
            "value": torch.randn(1, 2, 3, 8),
        }

        baseline = dot.forward_action(
            action_expert=action_head,
            action_pre=action_pre,
            fused_video_cache=fused_video_cache,
        )
        output, attention = dot.forward_action(
            action_expert=action_head,
            action_pre=action_pre,
            fused_video_cache=fused_video_cache,
            return_attention=True,
        )

        self.assertTrue(torch.equal(output, baseline))
        self.assertEqual(attention.shape, (2, 2, 8, 3))
        self.assertTrue(torch.isfinite(attention).all())
        self.assertTrue((attention >= 0).all())
        self.assertTrue((attention.sum(dim=-1) <= 1.0 + 1.0e-6).all())


if __name__ == "__main__":
    unittest.main()
