"""Top-k masking accepts idle DP ranks and reads live graph padding counts."""

import pytest
import torch

from sglang.kernels.ops.moe.dsv4 import mask_topk_ids, mask_topk_ids_to_int64
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b-kernel-unit", runner_config="1-gpu-large")
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA/PPU required"
)


@pytest.mark.parametrize("rows", [0, 1, 7])
@pytest.mark.parametrize("count_device", ["cpu", "cuda"])
@pytest.mark.parametrize("cast", [False, True])
def test_mask_topk_idle_and_nonempty(rows, count_device, cast):
    ids = torch.arange(rows * 16, dtype=torch.int32, device="cuda").reshape(rows, 16)
    original = ids.clone()
    valid = max(0, rows - 2)
    count = torch.tensor([valid], dtype=torch.int32, device=count_device)
    if cast:
        actual = mask_topk_ids_to_int64(ids, count)
        torch.testing.assert_close(ids, original)
    else:
        mask_topk_ids(ids, count)
        actual = ids
    expected = original.to(torch.int64 if cast else torch.int32)
    expected[valid:] = -1
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("cast", [False, True])
def test_mask_topk_graph_reads_padding_count(cast):
    original = torch.arange(7 * 16, dtype=torch.int32, device="cuda").reshape(7, 16)
    ids = original.clone()
    count = torch.tensor([7], dtype=torch.int32, device="cuda")
    if cast:
        mask_topk_ids_to_int64(ids, count)
    else:
        mask_topk_ids(ids, count)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        if cast:
            output = mask_topk_ids_to_int64(ids, count)
        else:
            mask_topk_ids(ids, count)
            output = ids
    for valid in (7, 0, 3):
        ids.copy_(original)
        count.fill_(valid)
        graph.replay()
        expected = original.to(torch.int64 if cast else torch.int32).clone()
        expected[valid:] = -1
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
