"""PPU-only exact mapping, dynamic graph replay, and warmed graph benchmark."""

import json
import unittest

import torch

from sglang.srt.layers.attention.dsa.indexer_request_sharding import (
    map_logical_topk_to_physical,
)
from sglang.srt.layers.attention.dsa.ppu_topk_map import (
    map_logical_topk_to_physical_triton,
)
from sglang.srt.utils import is_ppu
from sglang.test.test_utils import CustomTestCase


def reference(indices, table, page_size):
    valid = (indices >= 0) & (indices < table.shape[1] * page_size)
    safe = torch.where(valid, indices, 0)
    pages = table.gather(1, (safe // page_size).long())
    return torch.where(valid, pages * page_size + safe % page_size, -1)


class TestPPUTopKMap(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available() or not is_ppu():
            raise unittest.SkipTest("Requires the PPU Triton runtime")
        torch.manual_seed(42)

    def test_edges_tails_and_strides(self):
        for rows, width in (
            (0, 17),
            (3, 0),
            (1, 1),
            (7, 259),
            (160, 2048),
            (320, 2048),
            (640, 2048),
        ):
            for page_size in (1, 17, 64, 128):
                for strided in (False, True):
                    with self.subTest(
                        rows=rows, width=width, page_size=page_size, strided=strided
                    ):
                        cols = 2082
                        table = torch.randint(
                            -1,
                            1000000,
                            (rows, cols * 2),
                            device="cuda",
                            dtype=torch.int32,
                        )
                        table = (
                            table[:, ::2] if strided else table[:, :cols].contiguous()
                        )
                        indices = torch.randint(
                            -100,
                            cols * page_size + 100,
                            (rows, width * 2),
                            device="cuda",
                            dtype=torch.int32,
                        )
                        indices = (
                            indices[:, ::2]
                            if strided
                            else indices[:, :width].contiguous()
                        )
                        if rows and width >= 8:
                            indices[:, :8] = torch.tensor(
                                [
                                    -2147483648,
                                    -1,
                                    0,
                                    page_size - 1,
                                    page_size,
                                    cols * page_size - 1,
                                    cols * page_size,
                                    2147483647,
                                ],
                                device="cuda",
                                dtype=torch.int32,
                            )
                        expected = reference(indices, table, page_size)
                        actual = map_logical_topk_to_physical(indices, table, page_size)
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_expanded_table_and_output_ownership(self):
        table = torch.arange(2082, device="cuda", dtype=torch.int32)[None, :].expand(
            160, -1
        )
        ids = torch.randint(-1, 133248, (160, 2048), device="cuda", dtype=torch.int32)
        first = map_logical_topk_to_physical(ids, table)
        snapshot = first.clone()
        second = map_logical_topk_to_physical(ids + 1, table)
        self.assertNotEqual(first.data_ptr(), second.data_ptr())
        torch.testing.assert_close(first, snapshot, rtol=0, atol=0)
        torch.testing.assert_close(
            second, reference(ids + 1, table, 64), rtol=0, atol=0
        )

    def test_graph_replay_updates_both_inputs(self):
        table = torch.randint(0, 200000, (160, 2082), device="cuda", dtype=torch.int32)
        ids = torch.randint(-1, 133248, (160, 2048), device="cuda", dtype=torch.int32)
        map_logical_topk_to_physical(ids, table)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = map_logical_topk_to_physical(ids, table)
        for seed in range(3):
            torch.manual_seed(seed)
            ids.copy_(
                torch.randint(-100, 134000, ids.shape, device="cuda", dtype=torch.int32)
            )
            table.copy_(
                torch.randint(0, 200000, table.shape, device="cuda", dtype=torch.int32)
            )
            graph.replay()
            torch.testing.assert_close(
                actual, reference(ids, table, 64), rtol=0, atol=0
            )

    def test_warmed_graph_benchmark(self):
        ids = torch.randint(-1, 133248, (160, 2048), device="cuda", dtype=torch.int32)
        table = torch.randint(0, 200000, (160, 2082), device="cuda", dtype=torch.int32)

        def measure(fn):
            for _ in range(3):
                fn(ids, table, 64)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for _ in range(100):
                    _result = fn(ids, table, 64)
            for _ in range(3):
                graph.replay()
            start, end = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            start.record()
            for _ in range(10):
                graph.replay()
            end.record()
            end.synchronize()
            return start.elapsed_time(end) * 1000 / 1000

        old = measure(reference)
        new = measure(map_logical_topk_to_physical_triton)
        print(
            "PPU_TOPK_MAP_BENCH="
            + json.dumps(
                {
                    "shape": [160, 2048],
                    "page_size": 64,
                    "torch_us": old,
                    "fused_us": new,
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    unittest.main()
