import datetime
import socket
import time
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.environ import envs  # noqa: E402
from sglang.srt.managers.scheduler_components import dp_attn  # noqa: E402
from sglang.srt.model_executor.forward_batch_info import ForwardMode  # noqa: E402
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm  # noqa: E402

register_cpu_ci(est_time=45, suite="base-a-test-cpu")


class TestDPAttnSchedulerMetadata(CustomTestCase):
    def test_exchange_preserves_all_mlp_fields_and_elastic_path(self):
        rows = torch.tensor(
            [
                [11, 7, 1, 0, 0, ForwardMode.DECODE.value, 1, 10, 1],
                [23, 9, 0, 1, 1, ForwardMode.EXTEND.value, 0, 30, 0],
                [37, 5, 0, 1, 1, ForwardMode.EXTEND.value, 0, 90, 0],
                [49, 3, 1, 0, 0, ForwardMode.DECODE.value, 1, 50, 1],
            ],
            dtype=torch.int64,
        )
        active = torch.tensor([1, 1, 1, 0], dtype=torch.int64)
        # The CUDA case keeps tensors on CPU to test routing without needing a
        # device; only the external allocation/communication boundaries differ.
        cases = [
            ("cpu", False, "exchange"),
            ("cuda", False, "exchange"),
            ("cpu", True, "all_reduce"),
        ]
        for device, use_all_reduce, expected_path in cases:
            with self.subTest(device=device, elastic=use_all_reduce):
                info = dp_attn.MLPSyncBatchInfo(
                    dp_size=2,
                    tp_size=2,
                    cp_size=1,
                    num_tokens=11,
                    num_tokens_for_logprob=7,
                    can_run_decode_cuda_graph=True,
                    can_run_draft_cuda_graph=True,
                    can_run_prefill_cuda_graph=True,
                    is_extend_in_batch=False,
                    local_can_run_tbo=False,
                    local_forward_mode=ForwardMode.DECODE.value,
                    prefill_cuda_graph_max_prefix_len=10,
                )
                local = info._get_local_tensor("cpu")
                fallback = info._get_fallback_tensor("cpu")
                selected = []
                collected = []

                def write_rows(path):
                    def collect(output, input_tensor=None, **kwargs):
                        if input_tensor is not None:
                            torch.testing.assert_close(input_tensor, local)
                        output.view(4, 9).copy_(rows)
                        if use_all_reduce:
                            output.view(4, 9)[1].zero_()  # Absent elastic slot.
                        selected.append(path)
                        collected.append(output.view(4, 9))

                    return collect

                with (
                    patch.object(info, "_get_local_tensor", return_value=local),
                    patch.object(info, "_get_fallback_tensor", return_value=fallback),
                    patch.object(dp_attn, "_ENABLE_METRICS_DP_ATTENTION", False),
                    dp_attn.get_parallel().override(
                        tp_group=SimpleNamespace(
                            active_ranks_cpu=active, active_ranks=active
                        )
                    ),
                    patch.object(
                        dp_attn,
                        "exchange_sync_rows",
                        side_effect=write_rows("exchange"),
                    ),
                    patch.object(torch.distributed, "get_rank", return_value=0),
                    patch.object(
                        torch.distributed,
                        "all_reduce",
                        side_effect=write_rows("all_reduce"),
                    ),
                ):
                    info.all_gather(
                        device=device, group=object(), use_all_reduce=use_all_reduce
                    )

                expected = rows.clone()
                expected[3] = fallback
                if use_all_reduce:
                    expected[1] = fallback
                self.assertEqual(selected, [expected_path])
                torch.testing.assert_close(collected[0], expected, rtol=0, atol=0)
                torch.testing.assert_close(info.tp0_info_cpu, expected[[0, 2]])
                self.assertEqual(info.global_num_tokens, [11, 37])
                self.assertEqual(info.global_num_tokens_for_logprob, [7, 5])
                self.assertFalse(info.can_run_decode_cuda_graph)
                self.assertTrue(info.is_extend_in_batch)
                self.assertFalse(info.can_run_prefill_cuda_graph)
                self.assertEqual(info.prefill_cuda_graph_max_prefix_len, 90)
                self.assertFalse(info.can_run_draft_cuda_graph)

    def test_skip_all_gather_policy(self):
        with envs.SGLANG_SCHEDULER_SKIP_ALL_GATHER.override(False):
            self.assertTrue(dp_attn.should_skip_scheduler_all_gather(dp_size=1))
            self.assertFalse(dp_attn.should_skip_scheduler_all_gather(dp_size=2))
        with envs.SGLANG_SCHEDULER_SKIP_ALL_GATHER.override(True):
            self.assertTrue(dp_attn.should_skip_scheduler_all_gather(dp_size=2))

    def test_dp1_skip_preserves_local_tbo_metadata(self):
        batch = SimpleNamespace(
            forward_mode=ForwardMode.DECODE,
            batch_size=lambda: 4,
            spec_info=None,
        )
        tbo_preparer = Mock()
        tbo_preparer.prepare_all_gather.return_value = (
            True,
            ForwardMode.DECODE.value,
        )
        tbo_preparer.compute_output.return_value = (2, ForwardMode.DECODE)

        with (
            envs.SGLANG_SCHEDULER_SKIP_ALL_GATHER.override(False),
            patch.object(dp_attn, "TboDPAttentionPreparer", return_value=tbo_preparer),
            patch.object(dp_attn, "world_dp_gather_enabled", return_value=False),
            patch.object(
                dp_attn,
                "get_parallel",
                return_value=SimpleNamespace(
                    dp_size=1,
                    attn_tp_size=4,
                    attn_cp_size=1,
                    tp_group=SimpleNamespace(
                        device_group=object(), device="cpu", cpu_group=object()
                    ),
                ),
            ),
            patch.object(dp_attn, "check_cuda_graph_backend", return_value=False),
            patch.object(dp_attn.MLPSyncBatchInfo, "all_gather") as all_gather,
        ):
            result = dp_attn.prepare_mlp_sync_batch_raw(
                batch,
                model_runner=SimpleNamespace(
                    prefill_cuda_graph_runner=None,
                    spec_algorithm=SpeculativeAlgorithm.NONE,
                    model_config=object(),
                ),
                get_idle_batch=Mock(
                    side_effect=AssertionError("DP1 must not emit idle batch")
                ),
                disable_cuda_graph=False,
                require_mlp_tp_gather=False,
                disable_overlap_schedule=True,
                offload_tags=set(),
            )

        all_gather.assert_not_called()
        self.assertEqual(result.global_num_tokens, [4])
        self.assertEqual(result.tbo_split_seq_index, 2)
        self.assertEqual(result.global_forward_mode, ForwardMode.DECODE)
        self.assertEqual(result.recv_skipper_forward_mode, ForwardMode.DECODE)
        self.assertEqual(
            tbo_preparer.compute_output.call_args.args[0].tolist(),
            [[1, ForwardMode.DECODE.value]],
        )


class TestDecodeToExtendConversionVote(CustomTestCase):
    """A decode batch votes for the prefill graph only when its 1-token extend
    view can represent every row. Beam requests cannot: the converted batch
    takes the prefill result path, which commits them per-req instead of
    through the batch decode fold, and member rows carry no req."""

    def _vote(self, *, beam):
        runner = Mock(spec=dp_attn.PrefillCudaGraphRunner)
        runner.enable_lora = False
        runner.max_context_size = None
        runner.can_replay_locally.return_value = True
        batch = SimpleNamespace(
            forward_mode=ForwardMode.DECODE,
            batch_size=lambda: 2,
            return_logprob=False,
            has_grammar=False,
            reqs=[
                SimpleNamespace(beam_group=Mock() if beam else None),
                SimpleNamespace(beam_group=None),
            ],
        )
        with (
            patch.object(
                dp_attn, "get_moe_a2a_backend", return_value=Mock(is_none=lambda: True)
            ),
            patch.object(dp_attn, "uses_ssm_state", return_value=False),
            patch.object(
                dp_attn,
                "get_memory",
                return_value=SimpleNamespace(enable_hisparse=False),
            ),
            patch.object(
                dp_attn,
                "get_exec",
                return_value=SimpleNamespace(
                    overlap=SimpleNamespace(enable_two_batch_overlap=False)
                ),
            ),
            patch.object(dp_attn, "get_cp_strategy", return_value=None),
        ):
            return dp_attn._local_prefill_cuda_graph_vote(
                local_batch=batch,
                prefill_graph_runner=runner,
                coordinated_prefill=True,
                breakable_prefill=True,
                spec_algorithm=SpeculativeAlgorithm.NONE,
                model_config=object(),
            )

    def test_plain_decode_batch_votes_for_conversion(self):
        self.assertTrue(self._vote(beam=False))

    def test_beam_request_blocks_conversion(self):
        self.assertFalse(self._vote(beam=True))


class TestPrefillCudaGraphVote(CustomTestCase):
    def _vote(self, mode):
        runner = Mock(spec=dp_attn.PrefillCudaGraphRunner)
        runner.enable_lora = False
        runner.max_context_size = None
        runner.can_replay_locally.return_value = True
        batch = SimpleNamespace(
            forward_mode=mode,
            extend_num_tokens=4,
            input_embeds=None,
            replace_embeds=None,
            prefix_lens=[1, 1],
            return_logprob=False,
            batch_size=lambda: 2,
        )
        vote = dp_attn._local_prefill_cuda_graph_vote(
            local_batch=batch,
            prefill_graph_runner=runner,
            coordinated_prefill=True,
            breakable_prefill=True,
            spec_algorithm=SpeculativeAlgorithm.NONE,
            model_config=object(),
        )
        return vote, runner

    def test_extend_batch_votes_for_prefill_graph(self):
        vote, runner = self._vote(ForwardMode.EXTEND)

        self.assertTrue(vote)
        runner.can_replay_locally.assert_called_once()
        self.assertFalse(runner.can_replay_locally.call_args.kwargs["is_mixed"])

    def test_mixed_batch_delegates_to_runner_policy(self):
        vote, runner = self._vote(ForwardMode.MIXED)

        self.assertTrue(vote)
        runner.can_replay_locally.assert_called_once()
        self.assertTrue(runner.can_replay_locally.call_args.kwargs["is_mixed"])


# Keep the upstream prime/even/power-of-two coverage and add a subgroup whose
# group-rank 0 is global rank 1, plus the singleton all-gather fallback.
_EXCHANGE_WORLDS = (2, 3, 4, 5, 6, 8)
_EXCHANGE_GROUPS = tuple(tuple(range(sub)) for sub in _EXCHANGE_WORLDS) + (
    (1, 3, 5),
    (7,),
)
_EXCHANGE_ROUNDS = 4


def _exchange_worker(rank, world, port, out_q):
    """Compare the official exchange with ring all-gather under arrival skew."""
    import random

    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world,
        timeout=datetime.timedelta(seconds=60),
    )
    try:
        for members in _EXCHANGE_GROUPS:
            group = dist.new_group(list(members))
            if rank not in members:
                continue
            sub = len(members)
            group_rank = members.index(rank)
            rng = random.Random(1000 * sub + rank)
            for round_idx in range(_EXCHANGE_ROUNDS):
                is_extend = group_rank == (round_idx + 3) % sub
                info = dp_attn.MLPSyncBatchInfo(
                    dp_size=sub,
                    tp_size=1,
                    cp_size=1,
                    num_tokens=3 + group_rank + 100 * round_idx,
                    num_tokens_for_logprob=3 + group_rank,
                    can_run_decode_cuda_graph=group_rank != round_idx % sub,
                    can_run_draft_cuda_graph=group_rank != (round_idx + 1) % sub,
                    can_run_prefill_cuda_graph=group_rank != (round_idx + 2) % sub,
                    is_extend_in_batch=is_extend,
                    local_can_run_tbo=bool((group_rank + round_idx) % 2),
                    local_forward_mode=(
                        ForwardMode.EXTEND if is_extend else ForwardMode.DECODE
                    ).value,
                    prefill_cuda_graph_max_prefix_len=10 * (group_rank + 1) + round_idx,
                )
                local = info._get_local_tensor(device="cpu")
                reference = torch.zeros(sub * local.numel(), dtype=torch.int64)
                dp_attn.all_gather_single(reference, local, group=group)
                time.sleep(rng.uniform(0.0, 0.05))
                busy_until = time.perf_counter() + rng.uniform(0.0, 0.05)
                while time.perf_counter() < busy_until:
                    pass
                exchanged = torch.zeros_like(reference)
                dp_attn.exchange_sync_rows(exchanged, local, group=group)
                tp_group = SimpleNamespace(
                    active_ranks_cpu=torch.ones(sub, dtype=torch.int64)
                )
                with dp_attn.get_parallel().override(tp_group=tp_group):
                    info.all_gather(device="cpu", group=group)
                out_q.put(
                    (
                        members,
                        round_idx,
                        rank,
                        torch.equal(exchanged, reference),
                        torch.equal(info.tp0_info_cpu, reference.view(sub, 9)),
                        info.global_num_tokens,
                        info.global_num_tokens_for_logprob,
                        info.can_run_decode_cuda_graph,
                        info.can_run_draft_cuda_graph,
                        info.can_run_prefill_cuda_graph,
                        info.is_extend_in_batch,
                        info.prefill_cuda_graph_max_prefix_len,
                    )
                )
    finally:
        dist.destroy_process_group()


class TestSyncRowExchangeGloo(CustomTestCase):
    def test_device_and_small_groups_keep_all_gather(self):
        # A meta tensor exercises the device branch without accelerator access.
        for device, world in (("cpu", 1), ("cpu", 2), ("meta", 3)):
            with self.subTest(device=device, world=world):
                local = torch.empty(9, dtype=torch.int64, device=device)
                output = torch.full((world * 9,), -1, dtype=torch.int64)
                expected = torch.arange(world * 9, dtype=torch.int64)
                group = object()

                def all_gather(result, row, *, group):
                    self.assertIs(row, local)
                    result.copy_(expected)

                with (
                    patch.object(dist, "get_world_size", return_value=world),
                    patch.object(dp_attn, "all_gather_single", side_effect=all_gather),
                    patch.object(
                        dist, "gather", side_effect=AssertionError("unexpected gather")
                    ),
                    patch.object(
                        dist,
                        "broadcast",
                        side_effect=AssertionError("unexpected broadcast"),
                    ),
                ):
                    dp_attn.exchange_sync_rows(output, local, group)
                torch.testing.assert_close(output, expected, rtol=0, atol=0)

    def test_one_hop_exchange_matches_all_gather_on_every_rank(self):
        """Bitwise row equality and all nine fields across skewed Gloo groups."""
        world = max(_EXCHANGE_WORLDS)
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        sock.close()
        ctx = mp.get_context("spawn")
        out_q = ctx.Queue()
        procs = [
            ctx.Process(target=_exchange_worker, args=(r, world, port, out_q))
            for r in range(world)
        ]
        results = {}
        try:
            for proc in procs:
                proc.start()
            for _ in range(sum(map(len, _EXCHANGE_GROUPS)) * _EXCHANGE_ROUNDS):
                members, round_idx, rank, *result = out_q.get(timeout=240)
                results[(members, round_idx, rank)] = result
            for proc in procs:
                proc.join(timeout=60)
                self.assertEqual(proc.exitcode, 0)
        finally:
            for proc in procs:
                if proc.is_alive():
                    proc.kill()
                    proc.join(timeout=10)
            out_q.close()
            out_q.join_thread()
        for members in _EXCHANGE_GROUPS:
            sub = len(members)
            for round_idx in range(_EXCHANGE_ROUNDS):
                for rank in members:
                    (
                        same,
                        same_metadata,
                        tokens,
                        logprob_tokens,
                        decode_graph,
                        draft_graph,
                        prefill_graph,
                        is_extend,
                        prefix_len,
                    ) = results[(members, round_idx, rank)]
                    with self.subTest(members=members, round=round_idx, rank=rank):
                        self.assertTrue(same)
                        self.assertTrue(same_metadata)
                        self.assertEqual(
                            tokens, [3 + r + 100 * round_idx for r in range(sub)]
                        )
                        self.assertEqual(logprob_tokens, [3 + r for r in range(sub)])
                        self.assertFalse(decode_graph)
                        self.assertFalse(draft_graph)
                        self.assertFalse(prefill_graph)
                        self.assertTrue(is_extend)
                        self.assertEqual(prefix_len, 10 * sub + round_idx)


if __name__ == "__main__":
    unittest.main()
