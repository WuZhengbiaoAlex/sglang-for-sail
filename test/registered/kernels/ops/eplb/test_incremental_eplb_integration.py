"""Decode-only routing and count collection."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from sglang.srt.eplb import expert_location_dispatch as dispatch
from sglang.srt.eplb.expert_location import ExpertLocationMetadata
from sglang.srt.eplb.incremental import decode_bindings as db
from sglang.srt.layers.moe.token_dispatcher.deepep import (
    _IncrementalEPLBDeepEPBufferProxy,
)
from sglang.srt.runtime_context import get_resources
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b-kernel-unit", runner_config="1-gpu-large")


def make_bindings(device="cpu"):
    # Two initial homes and two spare slots per EP rank.
    home = torch.tensor([[0, 1, 4, 5]], dtype=torch.int64, device=device)
    physical = torch.tensor([[0, 1, 0, 1, 2, 3, 2, 3]], device=device)
    all_physical = torch.full((1, 4, 8), -1, dtype=torch.int64, device=device)
    all_physical[:, :, 0] = home
    meta = ExpertLocationMetadata(
        physical_to_logical_map=physical,
        physical_to_logical_map_cpu=physical.cpu(),
        logical_to_all_physical_map=all_physical,
        logical_to_all_physical_map_cpu=all_physical.cpu(),
        logical_to_all_physical_map_num_valid=torch.ones_like(home),
        ep_size=2,
        logical_to_rank_dispatch_physical_map=home,
    )
    live = home.clone()
    live[0, 0] = 6
    planner = SimpleNamespace(
        ep_rank=0,
        _dispatch_out=live,
        owners=live.unsqueeze(-1),
        phys=torch.zeros((1, 4), dtype=torch.int32, device=device),
    )
    return meta, planner, db.IncrementalEPLBDecodeBindings.from_planner(meta, planner)


def test_multi_owner_routes_are_decode_only(monkeypatch):
    from dataclasses import replace

    from sglang.srt.eplb.incremental import planner as unified_planner

    meta, planner, bindings = make_bindings()
    bindings = replace(bindings, owners=torch.tensor([[[0, 2, 4, 6]]]))
    extension = Mock()
    loader = Mock(return_value=extension)
    monkeypatch.setattr(unified_planner, "load_runtime_extension", loader)
    ids = torch.tensor([[0]])
    valid = torch.tensor([1])
    assert db.route_decode_topk(ids, 0, valid) is None
    loader.assert_not_called()
    with bindings.activate():
        assert db.route_decode_topk(ids, 0, valid) is extension.route.return_value
        extension.route.assert_called_once()
    loader.reset_mock()
    loader.side_effect = AssertionError("Prefill must not load the split router")
    assert db.route_decode_topk(ids, 0, valid) is None
    assert db.dispatch_counts_buffer(0) is None
    loader.assert_not_called()


def install_home(monkeypatch, meta):
    monkeypatch.setattr(get_resources(), "expert_location_metadata", meta)
    monkeypatch.setattr(
        dispatch,
        "get_exec",
        lambda: SimpleNamespace(
            moe=SimpleNamespace(
                ep_dispatch_algorithm="static", moe_a2a_backend="deepep"
            )
        ),
    )


def route(ids):
    return dispatch.topk_ids_logical_to_physical(
        ids, dispatch.ExpertLocationDispatchInfo.init_new(0)
    )


class CounterBuffer:
    def low_latency_dispatch(self, increments, **kwargs):
        self.kwargs = kwargs
        counter = kwargs.get("cumulative_local_expert_recv_stats")
        if counter is not None:
            counter.add_(increments)


def make_proxy(buffer):
    return _IncrementalEPLBDeepEPBufferProxy(buffer, 0, db.dispatch_counts_buffer)


def test_decode_prefill_decode_counts_and_routing(monkeypatch):
    meta, planner, bindings = make_bindings()
    install_home(monkeypatch, meta)
    buffer = CounterBuffer()
    proxy = make_proxy(buffer)
    with bindings.activate():
        proxy.low_latency_dispatch(torch.tensor([3, 5, 0, 0], dtype=torch.int32))
    before_prefill = planner.phys.clone()
    proxy.low_latency_dispatch(torch.full((4,), 10000, dtype=torch.int32))
    assert buffer.kwargs["cumulative_local_expert_recv_stats"] is None
    assert "inactive_expert_slot" not in buffer.kwargs
    torch.testing.assert_close(planner.phys, before_prefill)
    torch.testing.assert_close(route(torch.arange(4)), torch.tensor([0, 1, 4, 5]))
    with bindings.activate():
        proxy.low_latency_dispatch(torch.tensor([7, 9, 0, 0], dtype=torch.int32))
    # Counts include both decode steps and exclude the intervening prefill.
    torch.testing.assert_close(
        planner.phys, torch.tensor([[10, 14, 0, 0]], dtype=torch.int32)
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA/PPU graph test")
def test_prefill_graph_and_decode_graph_keep_separate_bindings(monkeypatch):
    meta, planner, _ = make_bindings("cuda")
    bindings = db.IncrementalEPLBDecodeBindings.from_planner(meta, planner)
    install_home(monkeypatch, meta)
    ids = torch.arange(4, device="cuda")
    increments = torch.ones(4, device="cuda", dtype=torch.int32)
    proxy = make_proxy(CounterBuffer())
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        route(ids)
        with bindings.activate():
            route(ids)
            proxy.low_latency_dispatch(increments)
    torch.cuda.current_stream().wait_stream(stream)
    baseline = planner.phys.clone()
    decode_graph, prefill_graph = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
    with bindings.capture():
        with torch.cuda.graph(decode_graph, stream=stream):
            decode_ids = route(ids)
            proxy.low_latency_dispatch(increments)
    # Prefill captures home routes without counters or an inactive-slot input.
    with torch.cuda.graph(prefill_graph, stream=stream):
        prefill_ids = route(ids)
        proxy.low_latency_dispatch(increments)
    torch.cuda.synchronize()
    torch.testing.assert_close(planner.phys, baseline)
    for replica in (6, 7):
        planner._dispatch_out[0, 0] = replica
        before = planner.phys.clone()
        decode_graph.replay()
        prefill_graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(
            decode_ids,
            torch.tensor([replica, 1, 4, 5], device="cuda"),
        )
        torch.testing.assert_close(
            prefill_ids, torch.tensor([0, 1, 4, 5], device="cuda")
        )
        torch.testing.assert_close(planner.phys - before, increments.unsqueeze(0))
        assert get_resources().expert_location_metadata is meta


from sglang.srt.layers.moe.token_dispatcher.deepep import (
    DeepEPBuffer,
    DeepEPDispatchMode,
)

TARGET = (128, 5120, 384)
DRAFT = (128, 5120, 128)


@pytest.fixture
def state(monkeypatch):
    state = SimpleNamespace(
        buffer=Mock(low_latency_mode=True),
        dispatch_mode=DeepEPDispatchMode.LOW_LATENCY,
        low_latency_layout=TARGET,
    )
    monkeypatch.setattr(DeepEPBuffer, "_state", classmethod(lambda cls: state))
    return state


def test_target_draft_target_resets_incoming_control_words(state):
    DeepEPBuffer.set_dispatch_mode_as_low_latency(DRAFT)
    state.buffer.clean_low_latency_buffer.assert_called_once_with(*DRAFT)
    assert state.low_latency_layout == DRAFT
    state.buffer.reset_mock()
    DeepEPBuffer.set_dispatch_mode_as_low_latency(TARGET)
    state.buffer.clean_low_latency_buffer.assert_called_once_with(*TARGET)


def test_unchanged_layout_does_not_add_collective_clean(state):
    DeepEPBuffer.set_dispatch_mode_as_low_latency(TARGET)
    DeepEPBuffer.set_dispatch_mode_as_low_latency(TARGET)
    state.buffer.clean_low_latency_buffer.assert_not_called()
    DeepEPBuffer.set_dispatch_mode_as_normal()
    DeepEPBuffer.set_dispatch_mode_as_low_latency(TARGET)
    state.buffer.clean_low_latency_buffer.assert_called_once_with(*TARGET)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
