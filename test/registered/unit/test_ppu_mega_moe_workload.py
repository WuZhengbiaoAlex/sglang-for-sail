"""Logical-workload tuning must not change inactive paths or require a new wheel."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from sglang.srt.layers.moe.ppu_mega_moe import _get_mega_moe_kwargs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def tuned_kernel(*, num_logical_experts=None):
    pass


def legacy_kernel():
    pass


@pytest.mark.parametrize(
    "experts",
    [
        SimpleNamespace(layer_id=1),
        SimpleNamespace(layer_id=1, _incremental_eplb_counts=lambda layer: None),
    ],
)
def test_inactive_path_does_not_consult_eplb_or_kernel(experts):
    with (
        patch(
            "sglang.srt.eplb.expert_location.get_global_expert_location_metadata",
            side_effect=AssertionError("inactive path read EPLB metadata"),
        ),
        patch(
            "sglang.srt.layers.moe.ppu_mega_moe._supports_logical_expert_count",
            side_effect=AssertionError("inactive path inspected kernel"),
        ),
    ):
        assert _get_mega_moe_kwargs(experts, tuned_kernel) == {}


def test_decode_tunes_logical_count_and_preserves_physical_counter():
    counter = object()
    experts = SimpleNamespace(
        layer_id=1, num_experts=1024, _incremental_eplb_counts=lambda layer: counter
    )
    with patch(
        "sglang.srt.eplb.expert_location.get_global_expert_location_metadata",
        return_value=SimpleNamespace(num_logical_experts=896),
    ):
        assert _get_mega_moe_kwargs(experts, tuned_kernel) == {
            "cumulative_local_expert_recv_stats": counter,
            "num_logical_experts": 896,
        }


def test_old_kernel_retains_counters_without_unsupported_keyword():
    counter = object()
    experts = SimpleNamespace(
        layer_id=1, _incremental_eplb_counts=lambda layer: counter
    )
    with patch(
        "sglang.srt.eplb.expert_location.get_global_expert_location_metadata",
        side_effect=AssertionError("legacy kernel needs no metadata"),
    ):
        assert _get_mega_moe_kwargs(experts, legacy_kernel) == {
            "cumulative_local_expert_recv_stats": counter
        }


@pytest.mark.parametrize(
    "block_m, tokens, expected",
    [(64, 20, (64, 384, 1000)), (32, 20, None), (64, 40, None)],
)
def test_cost_profile_uses_actual_kernel_and_supported_geometry(
    block_m, tokens, expected
):
    from unittest.mock import Mock

    from sglang.srt.layers.moe.ppu_mega_moe import _record_kernel_workload

    query = Mock(return_value={"l1_block_m": block_m})
    backend = SimpleNamespace(get_mega_moe_kernel_config=query)
    buffer = SimpleNamespace(
        group=SimpleNamespace(size=lambda: 64),
        num_topk=16,
        hidden=3584,
        intermediate_hidden=3072,
    )
    with (
        patch(
            "sglang.srt.eplb.incremental.decode_bindings.recording_workload",
            return_value=True,
        ),
        patch("sglang.srt.eplb.incremental.decode_bindings.record_workload") as record,
    ):
        _record_kernel_workload(
            SimpleNamespace(layer_id=4),
            backend,
            buffer,
            tokens,
            {"num_logical_experts": 896},
        )
        query.assert_called_once_with(buffer, tokens, num_logical_experts=896)
        if expected is None:
            record.assert_not_called()
        else:
            record.assert_called_once_with(4, expected)


def test_runtime_forward_and_inactive_path_do_not_query_kernel_config():
    from unittest.mock import Mock

    from sglang.srt.layers.moe.ppu_mega_moe import _record_kernel_workload

    query = Mock(side_effect=AssertionError("runtime cost query"))
    backend = SimpleNamespace(get_mega_moe_kernel_config=query)
    with patch(
        "sglang.srt.eplb.incremental.decode_bindings.recording_workload",
        return_value=False,
    ):
        _record_kernel_workload(None, backend, None, 20, {"num_logical_experts": 896})
    with patch(
        "sglang.srt.eplb.incremental.decode_bindings.recording_workload",
        side_effect=AssertionError("inactive state lookup"),
    ):
        _record_kernel_workload(None, backend, None, 20, {})
    query.assert_not_called()
