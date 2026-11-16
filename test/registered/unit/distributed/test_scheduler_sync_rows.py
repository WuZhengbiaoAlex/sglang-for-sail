"""CPU tests for the scheduler's rank-ordered metadata exchange."""

import ast
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path
from unittest.mock import Mock, patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="base-a-test-cpu")


def load_exchange(all_gather=dist.all_gather_into_tensor):
    source = (
        Path(__file__).resolve().parents[4]
        / "python/sglang/srt/managers/scheduler_components/dp_attn.py"
    )
    tree = ast.parse(source.read_text())
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "exchange_sync_rows"
    )
    namespace = {"torch": torch, "all_gather_single": all_gather}
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"),
        namespace,
    )
    return namespace["exchange_sync_rows"]


def check_group(group):
    exchange = load_exchange()
    ranks = dist.get_process_group_ranks(group)
    for width in (1, 13, 9):
        for dtype in (torch.int32, torch.int64):
            row = torch.arange(width, dtype=dtype) + 100 * dist.get_rank()
            actual = torch.full((len(ranks) * width,), -1, dtype=dtype)
            expected = torch.empty_like(actual)
            dist.all_gather_into_tensor(expected, row, group=group)
            exchange(actual, row, group)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def worker(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=6,
        timeout=timedelta(seconds=60),
    )
    try:
        check_group(dist.group.WORLD)
        subgroup = dist.new_group([1, 2, 4, 5], backend="gloo")
        if rank in (1, 2, 4, 5):
            check_group(subgroup)
        pair = dist.new_group([1, 4], backend="gloo")
        if rank in (1, 4):
            check_group(pair)
        singleton = dist.new_group([3], backend="gloo")
        if rank == 3:
            check_group(singleton)
        dist.barrier()
    finally:
        dist.destroy_process_group()


@unittest.skipUnless(dist.is_gloo_available(), "requires Gloo")
class TestSchedulerSyncRows(unittest.TestCase):
    def test_world_and_subgroups(self):
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(
                worker, args=(str(Path(directory) / "store"),), nprocs=6, join=True
            )

    def test_non_cpu_uses_all_gather(self):
        gather = Mock()
        local = torch.empty(13, device="meta")
        output = torch.empty(52, device="meta")
        group = object()
        with (
            patch.object(dist, "get_world_size", return_value=4),
            patch.object(
                dist,
                "gather",
                side_effect=AssertionError("CPU gather used for GPU data"),
            ),
            patch.object(
                dist,
                "broadcast",
                side_effect=AssertionError("CPU broadcast used for GPU data"),
            ),
        ):
            load_exchange(gather)(output, local, group)
        gather.assert_called_once_with(output, local, group=group)


if __name__ == "__main__":
    unittest.main()
