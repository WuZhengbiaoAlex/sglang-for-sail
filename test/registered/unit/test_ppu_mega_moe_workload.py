"""Record the selected kernel cost without overriding backend workload estimates."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from sglang.srt.layers.moe.ppu_mega_moe import _record_kernel_workload
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


@pytest.mark.parametrize(
    "block_m, tokens, expected",
    [(64, 20, (64, 384, 1000)), (32, 20, None), (64, 40, None)],
)
def test_cost_profile_uses_actual_kernel_and_supported_geometry(
    block_m, tokens, expected
):
    def query_config(buffer, num_tokens):
        return {"l1_block_m": block_m}

    query = Mock(side_effect=query_config)
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
        patch(
            "sglang.srt.eplb.expert_location.get_global_expert_location_metadata",
            return_value=SimpleNamespace(num_logical_experts=896),
        ),
        patch("sglang.srt.eplb.incremental.decode_bindings.record_workload") as record,
    ):
        _record_kernel_workload(
            SimpleNamespace(layer_id=4),
            backend,
            buffer,
            tokens,
            {"cumulative_local_expert_recv_stats": object()},
        )
        query.assert_called_once_with(buffer, tokens)
        if expected is None:
            record.assert_not_called()
        else:
            record.assert_called_once_with(4, expected)


def test_runtime_forward_and_inactive_path_do_not_query_kernel_config():
    query = Mock(side_effect=AssertionError("runtime cost query"))
    backend = SimpleNamespace(get_mega_moe_kernel_config=query)
    with patch(
        "sglang.srt.eplb.expert_location.get_global_expert_location_metadata",
        side_effect=AssertionError("inactive path read EPLB metadata"),
    ):
        with patch(
            "sglang.srt.eplb.incremental.decode_bindings.recording_workload",
            return_value=False,
        ):
            _record_kernel_workload(
                None,
                backend,
                None,
                20,
                {"cumulative_local_expert_recv_stats": object()},
            )
        with patch(
            "sglang.srt.eplb.incremental.decode_bindings.recording_workload",
            side_effect=AssertionError("inactive state lookup"),
        ):
            _record_kernel_workload(None, backend, None, 20, {})
    query.assert_not_called()


def test_backend_without_config_query_keeps_token_planning():
    with (
        patch(
            "sglang.srt.eplb.incremental.decode_bindings.recording_workload",
            return_value=True,
        ),
        patch(
            "sglang.srt.eplb.expert_location.get_global_expert_location_metadata",
            side_effect=AssertionError("unsupported backend read EPLB metadata"),
        ),
        patch("sglang.srt.eplb.incremental.decode_bindings.record_workload") as record,
    ):
        _record_kernel_workload(
            None,
            SimpleNamespace(),
            None,
            20,
            {"cumulative_local_expert_recv_stats": object()},
        )
    record.assert_not_called()
