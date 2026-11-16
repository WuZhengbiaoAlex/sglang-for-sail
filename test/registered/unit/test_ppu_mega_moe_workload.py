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
