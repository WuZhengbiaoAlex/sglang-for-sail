"""DFlash draft graph admission is local even when target DP peers are idle."""

import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.model_executor.runner import decode_cuda_graph_runner as graph_mod
from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
    DecodeCudaGraphRunner,
)
from sglang.srt.model_executor.runner_utils import buffers as buffers_mod
from sglang.srt.runtime_context import reset_context
from sglang.srt.speculative import dflash_worker_v2 as worker_mod
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _batch(bs=32):
    """The draft's hand-built batch has no target DP metadata/admission vote."""
    return ForwardBatch(
        forward_mode=ForwardMode.TARGET_VERIFY,
        batch_size=bs,
        input_ids=torch.zeros(bs * 5, dtype=torch.int64),
        req_pool_indices=torch.arange(bs),
        seq_lens=torch.full((bs,), 100, dtype=torch.int64),
        out_cache_loc=torch.arange(bs * 5),
        seq_lens_sum=bs * 100,
        positions=torch.arange(bs * 5),
        input_embeds=torch.arange(bs * 5 * 2).view(bs * 5, 2).float(),
        spec_info=SimpleNamespace(num_tokens_per_req=5),
    )


def _runner(*, draft=True, algorithm=SpeculativeAlgorithm.DFLASH):
    # Invoke production admission/load methods; mock only device/backend work.
    runner = object.__new__(DecodeCudaGraphRunner)
    runner.model_runner = SimpleNamespace(
        is_draft_worker=draft,
        spec_algorithm=algorithm,
        lora_manager=None,
        hisparse_coordinator=None,
    )
    runner.require_mlp_tp_gather = not runner._forward_is_dp_local(runner.model_runner)
    runner.require_attn_tp_gather = False
    # Deployment's old enable_dp_attention flag remains True in the draft scope.
    runner.require_mlp_sync = True
    runner.ragged_verify_mode = False
    runner.captured_req_width = 5
    runner.max_bs = 64
    runner.capture_bs = [1, 8, 32, 64]
    runner.dp_size = 4
    runner.disable_padding = False
    runner.enable_pdmux = False
    runner.is_encoder_decoder = False
    runner.enable_two_batch_overlap = False
    runner.record_nolora_graph = False
    runner.attention_graph_variants = None
    runner.dllm_uses_input_embeds = False
    runner.deepep_adapter = SimpleNamespace(replay=mock.Mock())
    runner.buffer_registry = SimpleNamespace(fill_from=mock.Mock())
    runner.buffers = SimpleNamespace(input_embeds=torch.zeros(64 * 5, 2))
    runner._validate_capture_hidden_mode = mock.Mock()
    runner._replay_attn_backend = mock.Mock(
        return_value=SimpleNamespace(init_forward_metadata_out_graph=mock.Mock())
    )
    runner.seq_len_fill_value = 5
    runner.capture_forward_mode = ForwardMode.TARGET_VERIFY
    runner._metadata_glue = None
    runner.backend = SimpleNamespace(can_run=mock.Mock(return_value=True))
    return runner


class TestDFlashDPLocalGraph(CustomTestCase):
    def tearDown(self):
        reset_context()
        super().tearDown()

    def test_draft_requires_scheduler_vote_without_global_counts(self):
        runner, batch = _runner(), _batch()
        self.assertIsNone(batch.original_global_num_tokens_cpu)
        self.assertFalse(batch.can_run_decode_cuda_graph)
        self.assertFalse(runner.can_run_graph(batch))
        batch.can_run_decode_cuda_graph = True
        self.assertTrue(runner.can_run_graph(batch))
        batch.spec_info.num_tokens_per_req = 4
        self.assertFalse(runner.can_run_graph(batch))
        batch.spec_info.num_tokens_per_req = 5
        batch.batch_size = 65
        self.assertFalse(runner.can_run_graph(batch))

    def test_draft_exact_bucket_backend_gate_still_applies(self):
        runner, batch = _runner(), _batch()
        runner.disable_padding = True
        runner.backend.can_run.return_value = False
        self.assertFalse(runner.can_run_graph(batch))
        self.assertEqual(runner.backend.can_run.call_args.args[1].size, 32)

    def test_draft_replay_uses_local_bucket_and_copies_embeddings(self):
        for bs, expected_bucket in ((32, 32), (3, 8), (1, 1)):
            with self.subTest(bs=bs):
                runner, batch = _runner(), _batch(bs)
                with mock.patch.object(graph_mod, "build_replay_fb_view") as view:
                    runner.load_batch(batch)
                self.assertEqual(runner.raw_bs, bs)
                self.assertEqual(runner.bs, expected_bucket)
                self.assertEqual(runner._replay_graph_key.size, expected_bucket)
                fill = runner.buffer_registry.fill_from.call_args.kwargs
                self.assertEqual(fill["raw_bs"], bs)
                self.assertEqual(fill["padded_bs"], expected_bucket)
                self.assertIsNone(view.call_args.kwargs["global_num_tokens_cpu"])
                torch.testing.assert_close(
                    runner.buffers.input_embeds[: bs * 5], batch.input_embeds
                )

    def test_target_still_requires_global_counts_and_vote(self):
        runner, batch = _runner(draft=False), _batch(3)
        with self.assertRaisesRegex(RuntimeError, "raw per-rank request counts"):
            runner.can_run_graph(batch)
        batch.original_global_num_tokens_cpu = [3, 48, 0, 1]
        self.assertFalse(runner.can_run_graph(batch))
        batch.can_run_decode_cuda_graph = True
        self.assertTrue(runner.can_run_graph(batch))
        with mock.patch.object(graph_mod, "build_replay_fb_view"):
            runner.load_batch(batch)
        self.assertEqual(runner.bs, 64)
        self.assertEqual(runner._replay_graph_key.size, 64)

    def test_attn_gather_keeps_single_local_metadata_slot(self):
        runner, batch = _runner(), _batch(3)
        runner.require_attn_tp_gather = True
        with mock.patch.object(
            buffers_mod, "enable_num_token_non_padded", return_value=False
        ):
            runner.buffers = buffers_mod.DecodeInputBuffers.create(
                device=torch.device("cpu"),
                max_bs=64,
                max_num_token=64 * 5,
                hidden_size=2,
                next_token_logits_buffer=torch.empty(64 * 5, 2),
                dtype=torch.float32,
                dp_size=4,
                pp_size=1,
                is_encoder_decoder=False,
                require_mlp_tp_gather=runner.require_mlp_tp_gather,
                seq_len_fill_value=5,
                encoder_len_fill_value=0,
                num_tokens_per_req=5,
                cache_loc_dtype=torch.int64,
                enable_mamba_track=False,
            )
        self.assertEqual(runner.buffers.global_num_tokens_gpu.numel(), 1)
        batch.can_run_decode_cuda_graph = True
        self.assertTrue(runner.can_run_graph(batch))
        with mock.patch.object(graph_mod, "build_replay_fb_view") as view:
            runner.load_batch(batch)
        self.assertEqual(runner.bs, 8)
        self.assertEqual(view.call_args.kwargs["global_num_tokens_cpu"], [40])

    def test_dspark_and_nextn_policy_preserved(self):
        with mock.patch(
            "sglang.srt.speculative.dspark_components.dspark_config.draft_is_deepseek_v4",
            return_value=False,
        ):
            runner = _runner(algorithm=SpeculativeAlgorithm.DSPARK)
            self.assertFalse(runner.require_mlp_tp_gather)
            batch = _batch()
            self.assertFalse(runner.can_run_graph(batch))
            batch.can_run_decode_cuda_graph = True
            self.assertTrue(runner.can_run_graph(batch))
        with mock.patch(
            "sglang.srt.speculative.dspark_components.dspark_config.draft_is_deepseek_v4",
            return_value=True,
        ):
            self.assertTrue(
                _runner(algorithm=SpeculativeAlgorithm.DSPARK).require_mlp_tp_gather
            )
        self.assertTrue(
            _runner(algorithm=SpeculativeAlgorithm.EAGLE).require_mlp_tp_gather
        )

    def test_cuda_dpa_capture_enabled_npu_still_eager(self):
        for npu, graph_backend, expected in (
            (False, "full", True),
            (True, "full", False),
            (False, "disabled", False),
        ):
            with self.subTest(npu=npu, backend=graph_backend):
                draft_worker = SimpleNamespace(init_cuda_graphs=mock.Mock())
                worker = SimpleNamespace(
                    draft_tp_context=mock.Mock(
                        side_effect=lambda *a, **kw: nullcontext()
                    ),
                    draft_model_runner=SimpleNamespace(
                        tp_group=object(), capture_tail_hooks=[]
                    ),
                    draft_owns_attention=True,
                    model_runner=SimpleNamespace(tp_rank=0),
                    _maybe_build_draft_sampler=mock.Mock(return_value=None),
                    _draft_worker=draft_worker,
                )
                config = SimpleNamespace(
                    graph=SimpleNamespace(
                        cuda_graph_config=SimpleNamespace(
                            decode=SimpleNamespace(backend=graph_backend)
                        )
                    )
                )
                with (
                    mock.patch.object(
                        worker_mod, "draft_pp_context", side_effect=nullcontext
                    ),
                    mock.patch.object(worker_mod, "get_exec", return_value=config),
                    mock.patch.object(
                        worker_mod,
                        "get_parallel",
                        return_value=SimpleNamespace(enable_dp_attention=True),
                    ),
                    mock.patch.object(worker_mod, "_is_npu", npu),
                    mock.patch.object(worker_mod, "is_cuda", return_value=False),
                    mock.patch.object(
                        worker_mod.current_platform,
                        "is_out_of_tree",
                        return_value=False,
                    ),
                ):
                    worker_mod.DFlashWorkerV2.init_cuda_graphs(worker)
                draft_worker.init_cuda_graphs.assert_called_once_with(
                    capture_decode_cuda_graph=expected
                )
                worker.draft_tp_context.assert_called_once_with(
                    worker.draft_model_runner.tp_group, owns_attention=True
                )


if __name__ == "__main__":
    unittest.main()
