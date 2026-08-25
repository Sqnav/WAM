from __future__ import annotations

import unittest

import torch
from torch import nn

from model.current_box_action_conditioner import CurrentBoxActionConditioner
from model.heads import FastWAMHead
from model.target_action_conditioning import TargetActionConditioning


class _FakeMoT:
    def __init__(self, action_expert: nn.Module, hidden_dim: int) -> None:
        self.num_layers = 2
        self.num_heads = 2
        self.mixtures = {"action": action_expert}
        self.hidden_dim = hidden_dim
        self.observed_video_batches: list[int] = []

    def _build_expert_attention_io(
        self,
        expert: nn.Module,
        block: nn.Module,
        hidden: torch.Tensor,
        freqs: torch.Tensor,
        t_mod: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        del expert, block, freqs, t_mod
        zeros = torch.zeros_like(hidden)
        return hidden, hidden, hidden, hidden, zeros, zeros, zeros, zeros, False

    def _mixed_attention(
        self,
        q_cat: torch.Tensor,
        k_cat: torch.Tensor,
        v_cat: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        del v_cat, attention_mask
        self.observed_video_batches.append(int(k_cat.size(0)))
        return q_cat


class CurrentBoxActionConditionerTests(unittest.TestCase):
    def test_box_residual_is_applied_independently_of_legacy_gate(self) -> None:
        conditioner = CurrentBoxActionConditioner(
            hidden_dim=16,
            layers=(1, 3),
            gate_init=0.0,
        )
        action_hidden = torch.randn(2, 8, 16)
        box_feature = conditioner.encode_box(torch.rand(2, 4))

        zero_gate_delta = conditioner.delta(1, action_hidden, box_feature)
        conditioner.gates["1"].data.fill_(10.0)
        changed_gate_delta = conditioner.delta(1, action_hidden, box_feature)

        self.assertGreater(zero_gate_delta.abs().max().item(), 0.0)
        torch.testing.assert_close(zero_gate_delta, changed_gate_delta)
        self.assertFalse(conditioner.gates["1"].requires_grad)
        torch.testing.assert_close(conditioner.gate_mean(), torch.tensor(1.0))

    def test_box_is_detached_but_conditioner_and_action_receive_gradients(self) -> None:
        conditioner = CurrentBoxActionConditioner(
            hidden_dim=16,
            layers=(0,),
            gate_init=0.5,
        )
        current_box = torch.rand(2, 4, requires_grad=True)
        action_hidden = torch.randn(2, 8, 16, requires_grad=True)

        box_feature = conditioner.encode_box(current_box)
        conditioner.delta(0, action_hidden, box_feature).square().mean().backward()

        self.assertIsNone(current_box.grad)
        self.assertIsNotNone(action_hidden.grad)
        self.assertTrue(
            all(parameter.grad is not None for parameter in conditioner.box_encoder.parameters())
        )
        self.assertTrue(
            all(
                parameter.grad is not None
                for parameter in conditioner.fusion_layers["0"].parameters()
            )
        )
        self.assertIsNone(conditioner.gates["0"].grad)

    def test_complete_tracker_head_bypasses_spatial_tracker_memory(self) -> None:
        head = FastWAMHead.__new__(FastWAMHead)
        nn.Module.__init__(head)
        head.current_box_action_conditioner = CurrentBoxActionConditioner(
            hidden_dim=16,
            layers=(0,),
        )
        head.tracker_integration = "mot_tracker_finetune_local_feature"
        head.tracker_condition_mode = "none"
        head.tracker_fusion = None
        head.cfg = type("Cfg", (), {"tracker_include_box_token": True})()

        condition = head._make_tracker_condition(
            tracker_features=None,
            tracker_center=None,
            tracker_bbox=torch.rand(2, 4),
            tracker_response=None,
            tracker_search_geometry=None,
            tracker_image_size=None,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )

        self.assertIsNone(condition)

    def test_cached_fastwam_candidates_repeat_video_kv_and_history(self) -> None:
        batch_size = 2
        extra_candidates = 3
        horizon = 8
        hidden_dim = 8
        video_tokens = 5

        head = FastWAMHead.__new__(FastWAMHead)
        nn.Module.__init__(head)
        action_expert = nn.Module()
        action_expert.blocks = nn.ModuleList([nn.Identity(), nn.Identity()])
        head.mot = _FakeMoT(action_expert, hidden_dim)
        head.current_box_action_conditioner = CurrentBoxActionConditioner(
            hidden_dim=hidden_dim,
            layers=(0, 1),
        )
        head.target_action_conditioning = TargetActionConditioning(
            action_hidden_dim=hidden_dim,
            history_length=8,
            horizon=horizon,
            memory_dim=8,
            memory_layers=1,
            memory_heads=2,
        )
        head._mot_attention_context_without_ffn = (
            lambda block, attention_io, mixed, context: mixed
        )
        head._mot_ffn = lambda block, hidden, attention_io: hidden

        candidate_batch = batch_size * extra_candidates
        action_tokens = torch.randn(candidate_batch, horizon, hidden_dim)
        action_pre = {
            "tokens": action_tokens,
            "freqs": torch.empty(0),
            "t_mod": torch.empty(0),
            "context": None,
            "context_mask": None,
        }
        video_kv_cache = [
            {
                "k": torch.randn(batch_size, video_tokens, hidden_dim),
                "v": torch.randn(batch_size, video_tokens, hidden_dim),
            }
            for _ in range(2)
        ]
        box_feature = head.current_box_action_conditioner.encode_box(
            torch.rand(candidate_batch, 4)
        )
        target_conditions = {
            "history": torch.randn(candidate_batch, horizon, hidden_dim)
        }
        attention_mask = torch.ones(
            video_tokens + horizon,
            video_tokens + horizon,
            dtype=torch.bool,
        )

        output = head._forward_action_with_video_cache_and_current_box(
            action_tokens=action_tokens,
            action_pre=action_pre,
            video_kv_cache=video_kv_cache,
            attention_mask=attention_mask,
            video_seq_len=video_tokens,
            box_feature=box_feature,
            target_conditions=target_conditions,
        )

        self.assertEqual(output.shape, action_tokens.shape)
        self.assertEqual(head.mot.observed_video_batches, [candidate_batch] * 2)
        self.assertTrue(torch.isfinite(output).all())


if __name__ == "__main__":
    unittest.main()
