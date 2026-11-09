"""DFlash CUDA/PPU idle graph and scheduler admission contracts, CPU only."""

import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.runtime_context import reset_context
from sglang.srt.speculative import dflash_info as info_mod
from sglang.srt.speculative import dflash_worker_v2 as mod
from sglang.srt.speculative.dflash_info_v2 import DFlashDraftInputV2
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestDFlashIdleGraph(CustomTestCase):
    def tearDown(self):
        reset_context()
        super().tearDown()

    def test_idle_graph_decision_and_empty_publish(self):
        for npu, dp, prepared, expected in (
            (False, True, True, True),
            (False, True, False, False),
            (True, True, True, False),
            (False, False, True, False),
        ):
            with self.subTest(npu=npu, dp=dp, prepared=prepared):
                fb = SimpleNamespace(can_run_decode_cuda_graph=prepared)
                target = SimpleNamespace(forward_batch_generation=mock.Mock())
                worker = SimpleNamespace(
                    device="cpu",
                    block_size=5,
                    _target_worker=target,
                    _validate_phase1_sampling_support=mock.Mock(),
                )
                worker._make_next_draft_input_decode = (
                    mod.DFlashWorkerV2._make_next_draft_input_decode.__get__(worker)
                )
                batch = SimpleNamespace(
                    forward_mode=ForwardMode.IDLE, is_extend_in_batch=False, spec_info=None
                )
                publish = mock.Mock()
                with (
                    mock.patch.object(mod, "_is_npu", npu),
                    mock.patch.object(mod, "get_parallel", return_value=SimpleNamespace(enable_dp_attention=dp)),
                    mock.patch.object(mod.DFlashVerifyInput, "prepare_for_verify", return_value=(fb, prepared)) as prepare,
                ):
                    result = mod.DFlashWorkerV2.forward_batch_generation(
                        worker, batch, on_publish=publish
                    )
                self.assertEqual(result.can_run_cuda_graph, expected)
                self.assertEqual(result.next_token_ids.numel(), 0)
                self.assertEqual(result.accept_lens.numel(), 0)
                self.assertEqual(result.new_seq_lens.numel(), 0)
                self.assertIsNone(result.logits_output)
                publish.assert_called_once_with(result.new_seq_lens)
                if dp:
                    prepare.assert_called_once()
                    self.assertEqual(fb.can_run_decode_cuda_graph, expected)
                    target.forward_batch_generation.assert_called_once_with(
                        batch=None, forward_batch=fb, is_verify=True, skip_attn_backend_init=True
                    )
                else:
                    prepare.assert_not_called()
                    target.forward_batch_generation.assert_not_called()

    def test_active_verify_keeps_cuda_idle_graph_and_rejects_mixed_round(self):
        for npu, dp, counts, mixed, initial, expected in (
            (False, True, [32, 0, 0, 0], False, True, True),
            (True, True, [32, 0, 0, 0], False, True, False),
            (True, True, [32, 32, 32, 32], False, True, True),
            (False, True, [32, 0, 0, 0], True, True, False),
            (True, True, [32, 32, 32, 32], True, True, False),
            (False, True, [32, 32, 32, 32], False, False, False),
            (False, False, None, True, True, True),
        ):
            with self.subTest(npu=npu, dp=dp, counts=counts, mixed=mixed):
                fb = SimpleNamespace(
                    original_global_num_tokens_cpu=counts,
                    can_run_decode_cuda_graph=initial,
                )
                with (
                    mock.patch.object(mod, "_is_npu", npu),
                    mock.patch.object(mod, "get_parallel", return_value=SimpleNamespace(enable_dp_attention=dp)),
                ):
                    mod.DFlashWorkerV2._apply_target_verify_graph_guards(
                        fb, is_extend_in_batch=mixed
                    )
                self.assertEqual(fb.can_run_decode_cuda_graph, expected)

    def test_prepare_idle_preserves_mode_and_uses_graph_loader(self):
        for graph_allowed in (True, False):
            with self.subTest(graph_allowed=graph_allowed):
                graph = SimpleNamespace(
                    can_run_graph=mock.Mock(return_value=graph_allowed),
                    load_batch=mock.Mock(),
                )
                attention = SimpleNamespace(init_forward_metadata=mock.Mock())
                target = SimpleNamespace(model_runner=SimpleNamespace(
                    decode_cuda_graph_runner=graph, attn_backend=attention
                ))
                batch = SimpleNamespace(forward_mode=ForwardMode.IDLE)
                fb = SimpleNamespace(forward_mode=ForwardMode.IDLE, batch_size=0)
                verify = info_mod.DFlashVerifyInput(
                    draft_token=torch.empty(0, dtype=torch.long),
                    positions=torch.empty(0, dtype=torch.long), draft_token_num=5,
                )
                with (
                    mock.patch.object(info_mod, "_is_npu", False),
                    mock.patch.object(info_mod.ForwardBatch, "init_new", return_value=fb),
                ):
                    actual, can_graph = verify.prepare_for_verify(batch, target)
                self.assertIs(actual, fb)
                self.assertEqual(batch.forward_mode, ForwardMode.IDLE)
                self.assertEqual(can_graph, graph_allowed)
                self.assertEqual(graph.load_batch.call_count, int(graph_allowed))
                attention.init_forward_metadata.assert_not_called()

    def test_real_draft_batch_inherits_both_scheduler_vote_values(self):
        class CapturedDraft(Exception):
            pass

        for vote in (False, True):
            with self.subTest(vote=vote):
                captured = []

                def capture(batch):
                    captured.append(batch)
                    raise CapturedDraft

                target_model = SimpleNamespace(
                    get_input_embeddings=lambda: torch.nn.Embedding(16, 2),
                    lm_head=SimpleNamespace(weight=torch.zeros(16, 2)),
                )
                target = SimpleNamespace(model_runner=SimpleNamespace(model=target_model))
                pool = SimpleNamespace(req_to_token=torch.zeros(2, 100, dtype=torch.int64))
                worker = SimpleNamespace(
                    device="cpu", block_size=5, target_worker=target,
                    draft_model=SimpleNamespace(),
                    model_runner=SimpleNamespace(req_to_token_pool=pool),
                    _validate_phase1_sampling_support=mock.Mock(),
                    _ensure_draft_block_buffers=mock.Mock(),
                    _draft_block_ids_buf=torch.empty(2, 5, dtype=torch.int64),
                    _draft_block_positions_buf=torch.empty(2, 5, dtype=torch.int64),
                    _draft_block_tokens_buf=torch.empty(2, 5, dtype=torch.int64),
                    _draft_verify_out_cache_loc_buf=torch.empty(2, 5, dtype=torch.int64),
                    _draft_block_end_buf=torch.empty(2, dtype=torch.int64),
                    _draft_seq_lens_cpu_buf=torch.empty(2, dtype=torch.int64),
                    _use_triton_prepare_block=False, _mask_token_id=3,
                    _block_pos_offsets=torch.arange(5),
                    _full_embed_gpu=torch.zeros(16, 2), _noise_embed_scale=1.0,
                    use_compact_draft_cache=False,
                    _draft_block_spec_info=SimpleNamespace(num_tokens_per_req=5),
                    selector=None, lilicorr=None,
                    draft_owns_attention=True,
                    draft_tp_context=lambda *a, **kw: nullcontext(),
                    draft_model_runner=SimpleNamespace(tp_group=object(), forward=capture),
                )
                state = DFlashDraftInputV2.create_idle_input("cpu")
                state.bonus_tokens = torch.ones(2, dtype=torch.int64)
                batch = SimpleNamespace(
                    forward_mode=ForwardMode.DECODE, is_extend_in_batch=False,
                    spec_info=state, seq_lens=torch.tensor([3, 4]),
                    seq_lens_cpu=torch.tensor([3, 4]), req_pool_indices=torch.arange(2),
                    can_run_decode_cuda_graph=vote,
                )
                with (
                    mock.patch.object(torch.Tensor, "record_stream"),
                    mock.patch.object(torch, "get_device_module", return_value=SimpleNamespace(current_stream=lambda: None)),
                    mock.patch.object(mod, "enable_num_token_non_padded", return_value=False),
                    mock.patch.object(mod, "assign_extend_cache_locs_func", return_value=torch.arange(10)),
                    self.assertRaises(CapturedDraft),
                ):
                    mod.DFlashWorkerV2.forward_batch_generation(worker, batch)
                self.assertEqual(len(captured), 1)
                self.assertEqual(captured[0].can_run_decode_cuda_graph, vote)
                self.assertEqual(captured[0].batch_size, 2)
                self.assertIsNone(captured[0].original_global_num_tokens_cpu)


if __name__ == "__main__":
    unittest.main()
