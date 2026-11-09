"""Native CPU placement contracts, without acquiring an accelerator."""

import importlib.util
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=30, suite="base-a-test-cpu")


@pytest.fixture(scope="module")
def module():
    path = (
        Path(__file__).resolve().parents[4]
        / "python/sglang/srt/eplb/incremental/cpu_planner.py"
    )
    spec = importlib.util.spec_from_file_location("incremental_cpu_planner_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def native(module):
    return module.load_cpu_planner()


def layout(layers=2, ranks=32):
    experts, slots = 384, 384 // ranks + 2
    row = [
        r * (slots - 2) + s % (slots - 2) for r in range(ranks) for s in range(slots)
    ]
    resident = torch.tensor([row] * layers, dtype=torch.int32)
    owners = torch.full((layers, experts, ranks), -1, dtype=torch.int32)
    owners[:, :, 0] = torch.tensor(
        [e // (slots - 2) * slots + e % (slots - 2) for e in range(experts)]
    )
    tokens = torch.ones((layers, experts), dtype=torch.int64)
    return tokens, resident, owners


def plan(native, tokens, resident, owners, *, threads=4):
    output, routes, copies = (torch.empty_like(t) for t in (resident, owners, resident))
    ranks = owners.shape[-1]
    changed = native.plan_into(
        tokens,
        resident,
        owners,
        output,
        routes,
        copies,
        ranks,
        2,
        32,
        8,
        3,
        5,
        2,
        5,
        1000,
        threads,
    )
    assert isinstance(changed, bool)
    assert changed == (bool((copies >= 0).any()) or not torch.equal(owners, routes))
    return output, routes, copies


def check_placement(resident, owners, output, routes, copies):
    layers, experts, ranks = owners.shape
    slots = resident.shape[-1] // ranks
    for l in range(layers):
        protected = set(owners[l][owners[l] >= 0].tolist())
        assert int((copies[l] >= 0).sum()) <= 2
        for dst in (copies[l] >= 0).nonzero().flatten().tolist():
            src = int(copies[l, dst])
            expert = int(output[l, dst])
            assert dst not in protected
            assert src in owners[l, expert].tolist()
            assert int(resident[l, src]) == expert
        seen = set()
        for expert in range(experts):
            active = routes[l, expert][routes[l, expert] >= 0].tolist()
            assert active and len({s // slots for s in active}) == len(active)
            for slot in active:
                assert slot not in seen and int(output[l, slot]) == expert
                seen.add(slot)
        assert all(sum(s // slots == r for s in seen) < slots for r in range(ranks))


def test_split_merge_and_thread_count_are_deterministic(native):
    tokens, resident, owners = layout()
    maximum = 1
    for step in range(12):
        tokens.fill_(1)
        if step < 8:
            tokens[:, 176] = 3200
        one = plan(native, tokens, resident, owners, threads=1)
        four = plan(native, tokens, resident, owners, threads=4)
        for a, b in zip(one, four):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        check_placement(resident, owners, *four)
        resident, owners = four[:2]
        maximum = max(maximum, int((owners >= 0).sum(-1).max()))
    assert maximum > 2
    assert int((owners >= 0).sum(-1).max()) == 1


def test_zero_and_large_assignment_counts(native):
    tokens, resident, owners = layout(layers=1, ranks=64)
    tokens.zero_()
    output, routes, copies = plan(native, tokens, resident, owners)
    torch.testing.assert_close(output, resident)
    torch.testing.assert_close(routes, owners)
    assert (copies == -1).all()
    tokens[0, 0] = 1 << 24
    check_placement(resident, owners, *plan(native, tokens, resident, owners))


@pytest.mark.parametrize(
    "bad",
    [
        "negative",
        "overflow",
        "resident",
        "owner",
        "padding",
        "duplicate_rank",
        "dtype",
        "alias",
    ],
)
def test_invalid_snapshots_raise_without_crashing(native, bad):
    tokens, resident, owners = layout(layers=1)
    if bad == "negative":
        tokens[0, 0] = -1
    elif bad == "overflow":
        tokens[0, 0] = (1 << 24) + 1
    elif bad == "resident":
        resident[0, 0] = 384
    elif bad == "owner":
        owners[0, 0, 0] = resident.numel()
    elif bad == "padding":
        owners[0, 0, 2] = 12
    elif bad == "duplicate_rank":
        owners[0, 0, 1] = 12
    elif bad == "dtype":
        tokens = tokens.int()
    outputs = [torch.empty_like(t) for t in (resident, owners, resident)]
    if bad == "alias":
        outputs[0] = resident
    with pytest.raises(RuntimeError):
        native.plan_into(
            tokens, resident, owners, *outputs, 32, 2, 32, 8, 3, 5, 2, 5, 1000, 4
        )


def test_worker_guard_and_validation_precede_allocation(module, monkeypatch):
    monkeypatch.setattr(module, "load_cpu_planner", Mock())
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    planner = module.CpuPlanner(threads=4)
    args = (32, 2, 32, 8, 3, 5, 2, 5, 1000)
    tokens, resident, owners = layout(layers=1)
    with pytest.raises(RuntimeError, match="background worker"):
        planner.plan(tokens, resident, owners, *args)

    def tensor(shape, dtype):
        return SimpleNamespace(
            ndim=len(shape),
            shape=shape,
            dtype=dtype,
            is_cuda=True,
            device=torch.device("cuda", 0),
            is_contiguous=lambda: True,
        )

    inputs = [tensor(t.shape, t.dtype) for t in (tokens, resident, owners)]
    inputs[0].dtype = torch.float32
    with ThreadPoolExecutor(max_workers=1) as pool:
        with pytest.raises(ValueError, match="int64 counts"):
            pool.submit(planner.plan, *inputs, *args).result()
    assert planner.shape is None
    planner.library.plan_into.assert_not_called()


@pytest.mark.parametrize("ranks", [8, 16, 32])
def test_cpu_pull_records_match_planner_for_every_rank(native, ranks):
    tokens, resident, owners = layout(layers=2, ranks=ranks)
    slots = resident.shape[1] // ranks
    pulls = torch.empty((2, slots, 4), dtype=torch.int32)
    experts = torch.empty((2, slots), dtype=torch.int32)
    for step in range(12):
        tokens.fill_(1)
        tokens[:, step % tokens.shape[1]] = 10000
        output, routes, copies = plan(native, tokens, resident, owners)
        for rank in range(ranks):
            native.prepare_pulls(
                copies, owners, resident, output, pulls, experts, ranks, rank
            )
            for layer in range(2):
                for local in range(slots):
                    dst = rank * slots + local
                    src = int(copies[layer, dst])
                    expected = (
                        [-1, -1, local, layer]
                        if src < 0
                        else [src // slots, src % slots, local, layer]
                    )
                    assert pulls[layer, local].tolist() == expected
                    assert int(experts[layer, local]) == (
                        -1 if src < 0 else int(output[layer, dst])
                    )
        resident, owners = output, routes


@pytest.mark.parametrize("invalid", ["source", "expert", "active", "rank", "alias"])
def test_cpu_pull_validation_rejects_unsafe_records(native, invalid):
    _, resident, owners = layout(layers=1, ranks=4)
    slots = resident.shape[1] // 4
    output = resident.clone()
    sources = torch.full_like(resident, -1)
    pulls = torch.empty((1, slots, 4), dtype=torch.int32)
    experts = torch.empty((1, slots), dtype=torch.int32)
    dst = slots - 1
    sources[0, dst] = slots
    output[0, dst] = resident[0, slots]
    rank = 0
    if invalid == "source":
        sources[0, dst] = resident.numel()
    elif invalid == "expert":
        output[0, dst] = owners.shape[1]
    elif invalid == "active":
        sources[0, 0] = slots
        output[0, 0] = resident[0, slots]
    elif invalid == "rank":
        rank = 4
    else:
        experts = sources[:, :slots]
    with pytest.raises(RuntimeError):
        native.prepare_pulls(sources, owners, resident, output, pulls, experts, 4, rank)
