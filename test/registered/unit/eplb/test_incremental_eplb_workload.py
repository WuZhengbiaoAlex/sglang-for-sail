"""Kernel geometry belongs to the captured graph, not a mutable global setting."""

import pytest

from sglang.srt.eplb.incremental.workload import GraphWorkloads
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def test_profiles_follow_graph_keys_and_preserve_non_moe_layers():
    records = GraphWorkloads(3)
    with records.capture("batch32"):
        records.record(1, (64, 384, 1000))
        records.record(1, (64, 384, 1000))
    with records.capture("batch64"):
        records.record(1, (128, 500, 1000))
    assert records.for_graph("batch32").tolist() == [
        [1, 0, 1],
        [64, 384, 1000],
        [1, 0, 1],
    ]
    assert records.for_graph("batch64")[1].tolist() == [128, 500, 1000]


def test_unprofiled_backend_and_outside_capture_are_inert():
    records = GraphWorkloads(3)
    records.record(1, (64, 384, 1000))
    with records.capture("deepep"):
        pass
    assert records.for_graph("deepep") is None


def test_capture_failure_does_not_publish_partial_metadata():
    records = GraphWorkloads(3)
    with pytest.raises(RuntimeError, match="changed"):
        with records.capture("batch32"):
            records.record(1, (64, 384, 1000))
            records.record(1, (32, 384, 1000))
    assert not records.recording
    with pytest.raises(RuntimeError, match="Missing"):
        records.for_graph("batch32")


def test_mixed_or_idle_dp_shapes_use_common_token_objective():
    records = GraphWorkloads(2)
    with records.capture("batch32"):
        records.record(1, (64, 384, 1000))
    assert records.for_batch("batch32", [160] * 8) is records.for_graph("batch32")
    assert records.for_batch("batch32", [160, 80]) is None
    assert records.for_batch("batch16", [160, 80]) is None
    assert records.for_batch("idle", [160, 0]) is None
    assert records.for_batch("batch32", None) is None


def test_rank_agreement_checked_before_publishing_profile():
    def reject(key, values):
        assert key == "batch32"
        assert values == ((1, 0, 1), (64, 384, 1000))
        raise RuntimeError("different rank profile")

    records = GraphWorkloads(2, validator=reject)
    with pytest.raises(RuntimeError, match="different rank"):
        with records.capture("batch32"):
            records.record(1, (64, 384, 1000))
    with pytest.raises(RuntimeError, match="Missing"):
        records.for_graph("batch32")


@pytest.mark.parametrize("mismatch", [False, True])
def test_startup_agreement_rejects_different_rank_objectives(mismatch):
    from types import SimpleNamespace
    from unittest.mock import patch

    from sglang.srt.eplb.incremental.workload import validate_ep_profiles

    group = object()

    def gather(output, local, *, group):
        output[:] = [local, ("different shape", local[1]) if mismatch else local]

    with (
        patch(
            "sglang.srt.runtime_context.get_parallel",
            return_value=SimpleNamespace(moe_ep_group=SimpleNamespace(cpu_group=group)),
        ),
        patch("torch.distributed.get_world_size", return_value=2),
        patch("torch.distributed.all_gather_object", side_effect=gather) as collective,
    ):
        if mismatch:
            with pytest.raises(RuntimeError, match="differ across EP ranks"):
                validate_ep_profiles("batch32", ((64, 384, 1000),))
        else:
            validate_ep_profiles("batch32", ((64, 384, 1000),))
        assert collective.call_count == 1
