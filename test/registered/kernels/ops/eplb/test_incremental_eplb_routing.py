"""Split routing preserves occurrence order across kernel shapes and replay."""

import pytest
import torch

from sglang.srt.eplb.incremental.planner import load_runtime_extension
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=15, stage="base-b-kernel-unit", runner_config="1-gpu-large")
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA/PPU required"
)


def reference(ids, owners, valid, rank):
    result = torch.full_like(ids, -1, dtype=torch.int64)
    slots = [[slot for slot in row if slot >= 0] for row in owners.tolist()]
    seen = [0] * len(slots)
    for i, expert in enumerate(ids.flatten().tolist()):
        if i // ids.shape[1] >= valid or not 0 <= expert < len(slots):
            continue
        targets = slots[expert]
        index = (seen[expert] + rank + expert) % len(targets)
        result.view(-1)[i] = targets[index]
        seen[expert] += 1
    return result


def placement(width):
    owners = torch.full((17, width), -1, dtype=torch.int32)
    for expert in range(17):
        parts = 1 if expert % 3 else width
        owners[expert, :parts] = torch.arange(parts) * 17 + expert
    return owners


@pytest.mark.parametrize("rows", [0, 171, 342, 510, 683, 1024])
@pytest.mark.parametrize("width", [1, 32, 64])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("count_dtype", [torch.int32, torch.int64])
def test_split_routes_preserve_occurrence_order(rows, width, dtype, count_dtype):
    ids = torch.randint(-1, 18, (rows, 6), generator=torch.Generator().manual_seed(42))
    ids[: rows // 2, 0] = 0
    owners = placement(width)
    valid = max(0, rows - 7)
    expected = reference(ids, owners, valid, 31)
    actual = load_runtime_extension().route(
        ids.to(device="cuda", dtype=dtype),
        owners.cuda(),
        torch.tensor([valid], device="cuda", dtype=count_dtype),
        31,
    )
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)


def test_split_route_graph_reads_current_inputs():
    extension = load_runtime_extension()
    ids = torch.zeros((510, 6), device="cuda", dtype=torch.int64)
    owners = placement(32).cuda()
    valid = torch.tensor([510], device="cuda", dtype=torch.int32)
    extension.route(ids, owners, valid, 7)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = extension.route(ids, owners, valid, 7)
    generator = torch.Generator().manual_seed(17)
    for n in (0, 1, 503, 510):
        current_ids = torch.randint(-1, 18, (510, 6), generator=generator)
        current_owners = placement(32)
        if n % 2:
            current_owners[:, 1:] = -1
        ids.copy_(current_ids)
        owners.copy_(current_owners)
        valid.fill_(n)
        graph.replay()
        torch.testing.assert_close(
            output.cpu(), reference(current_ids, current_owners, n, 7), rtol=0, atol=0
        )
