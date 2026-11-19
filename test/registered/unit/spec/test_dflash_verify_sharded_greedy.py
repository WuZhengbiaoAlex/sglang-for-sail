import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import torch
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.graph_shared_output import GraphSharedOutput
from sglang.srt.runtime_context import get_parallel
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.srt.speculative.dflash_worker_v2 import (
    DFlashWorkerV2,
    _can_use_sharded_greedy,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=4, suite="base-a-test-cpu")


def greedy_batch():
    info = SamplingBatchInfo(
        temperatures=torch.ones((1, 1)),
        top_ps=torch.ones(1),
        top_ks=torch.ones(1, dtype=torch.int32),
        min_ps=torch.zeros(1),
        is_all_greedy=True,
        is_any_greedy=True,
        need_top_p_sampling=False,
        need_top_k_sampling=False,
        need_min_p_sampling=False,
        vocab_size=32,
    )
    return SimpleNamespace(
        sampling_info=info, has_grammar=False, return_logprob=False, beam_tail=None
    )


class TestDFlashVerifyShardedGreedy(CustomTestCase):
    def test_batch_eligibility_and_modifiers(self):
        batch = greedy_batch()
        self.assertTrue(_can_use_sharded_greedy(batch, None))
        self.assertFalse(_can_use_sharded_greedy(batch, (torch.tensor([1]), None)))
        for field in ("has_grammar", "return_logprob"):
            with self.subTest(field=field):
                batch = greedy_batch()
                setattr(batch, field, True)
                self.assertFalse(_can_use_sharded_greedy(batch, None))

        for field, value in (
            ("is_all_greedy", False),
            ("has_custom_logit_processor", True),
            ("grammar_mask", object()),
            ("logit_bias", torch.zeros(1, 32)),
            ("acc_additive_penalties", torch.zeros(1, 32)),
            ("acc_scaling_penalties", torch.ones(1, 32)),
            ("penalizer_orchestrator", SimpleNamespace(is_required=True)),
            ("sampling_mask_batch_indices", torch.tensor([0])),
            ("sampling_support_logprobs_capture_indices", torch.tensor([0])),
        ):
            with self.subTest(field=field):
                batch = greedy_batch()
                setattr(batch.sampling_info, field, value)
                self.assertFalse(_can_use_sharded_greedy(batch, None))

    def test_alternating_requests_share_the_captured_shard(self):
        worker = DFlashWorkerV2.__new__(DFlashWorkerV2)
        worker._selector_sample = None
        storage = GraphSharedOutput(device=torch.device("cpu"), max_rows=5)
        worker.model_runner = SimpleNamespace(graph_shared_output=storage)
        shard = SimpleNamespace(org_vocab_start_index=0, num_org_elements=4)
        head = SimpleNamespace(shard_indices=shard, org_vocab_size=7)
        local = torch.tensor([[1, 7, 0, -1], [0, 1, 2, 3]], dtype=torch.bfloat16)
        peer = torch.tensor([[7, 0, 2, 31], [6, 1, 2, 31]], dtype=torch.bfloat16)
        group = Mock(world_size=2)
        group.all_gather.return_value = torch.cat([local, peer], dim=1)

        def gather_small(output, values):
            other_values, other_ids = peer[:, :3].max(dim=1)
            other = other_values if values.is_floating_point() else other_ids + 4
            output.copy_(torch.cat([values, other]))

        group.all_gather_into_tensor.side_effect = gather_small
        captured = LogitsProcessorOutput(
            next_token_logits=None, dflash_local_logits=local
        )
        reference = group.all_gather.return_value[:, :7].float()
        batch = greedy_batch()
        with get_parallel().override(attn_tp_group=group):
            for requires_full, graph in (
                (False, True),
                (True, True),
                (True, False),
                (False, True),
            ):
                with self.subTest(requires_full=requires_full, graph=graph):
                    batch.return_logprob = requires_full
                    replay = replace(captured)
                    ids = worker._resolve_verify_vocab_output(
                        batch, replay, head, can_run_cuda_graph=graph
                    )
                    self.assertIsNone(replay.dflash_local_logits)
                    self.assertIs(captured.dflash_local_logits, local)
                    if requires_full:
                        self.assertIsNone(ids)
                        torch.testing.assert_close(replay.next_token_logits, reference)
                        if graph:
                            self.assertEqual(
                                replay.next_token_logits.data_ptr(),
                                storage.get_logits_buffer(7, rows=2).data_ptr(),
                            )
                    else:
                        self.assertIsNone(replay.next_token_logits)
                        torch.testing.assert_close(ids, reference.argmax(-1))
            self.assertEqual(group.all_gather.call_count, 2)
            self.assertEqual(group.all_gather_into_tensor.call_count, 4)

    def test_accept_commits_match_dense_for_every_bonus_position(self):
        worker = DFlashWorkerV2.__new__(DFlashWorkerV2)
        worker.block_size = 5
        worker._selector_sample = None
        worker._use_triton_accept_bonus = False
        worker._tp_sync = Mock()
        top1 = torch.tensor([[1, 2, 3, 4, 5]] * 5)
        logits = torch.zeros(25, 32).scatter_(1, top1.reshape(-1, 1), 1)
        proposals = torch.tensor([[11, 1, 2, 3, 4]] * 5)
        for position in range(4):
            proposals[position, 1 + position] = 20
        kwargs = dict(
            candidates=proposals,
            sampling_info=greedy_batch().sampling_info,
            draft_input=None,
            prefix_lens=torch.arange(5),
            bs=5,
        )
        old = worker._accept_block(next_token_logits=logits, **kwargs)
        new = worker._accept_block(
            next_token_logits=None, target_predict=top1.flatten(), **kwargs
        )
        for a, b in zip(old, new):
            if a is None:
                self.assertIsNone(b)
            else:
                torch.testing.assert_close(a, b)
        self.assertEqual(new[0].tolist(), [0, 1, 2, 3, 4])
        self.assertEqual(new[1].tolist(), [1, 2, 3, 4, 5])
        self.assertEqual(new[2].tolist(), [1, 2, 3, 4, 5])


if __name__ == "__main__":
    unittest.main()
