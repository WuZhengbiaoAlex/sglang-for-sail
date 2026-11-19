import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

import torch

from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.models import kimi_k3
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=4, suite="base-a-test-cpu")


class TestKimiMLAGateOverlap(CustomTestCase):
    def setUp(self):
        super().setUp()
        self.model = kimi_k3.KimiK3MLAAttention.__new__(kimi_k3.KimiK3MLAAttention)
        torch.nn.Module.__init__(self.model)
        self.streams = Mock()
        self.model.use_output_gate = True
        self.model._gate_alt_stream = self.streams.gate
        self.model._gate_bs_limit = 64
        self.model._gate_overlap_verify_160 = True
        self.model._gate_overlap_verify_160_late = False
        self.capture = self.enterContext(
            patch.object(kimi_k3, "get_is_capture_mode", return_value=True)
        )
        self.breakable = self.enterContext(
            patch.object(kimi_k3, "is_in_breakable_cuda_graph", return_value=False)
        )
        self.enterContext(
            patch.object(torch.cuda, "current_stream", return_value=self.streams.main)
        )

    def forks(self, rows, mode):
        self.model._fork_output_gate(
            torch.empty(rows, 1), SimpleNamespace(forward_mode=mode)
        )
        return self.model._gate_pending_stream is not None

    def test_exact_verify_shape(self):
        for rows in (0, 64, 65, 159, 160, 161):
            for mode in ForwardMode:
                with self.subTest(rows=rows, mode=mode):
                    expected = rows == 64 or (
                        rows == 160 and mode == ForwardMode.TARGET_VERIFY
                    )
                    self.assertEqual(self.forks(rows, mode), expected)

    def test_capture_guards_and_disabled_flag(self):
        mode = ForwardMode.TARGET_VERIFY
        self.model._gate_overlap_verify_160 = False
        self.assertFalse(self.forks(160, mode))
        self.assertTrue(self.forks(64, mode))
        self.model._gate_overlap_verify_160 = True
        self.capture.return_value = False
        self.assertFalse(self.forks(160, mode))
        self.capture.return_value = True
        self.breakable.return_value = True
        self.assertFalse(self.forks(160, mode))
        self.breakable.return_value = False
        self.model._gate_alt_stream = None
        self.assertFalse(self.forks(160, mode))

    def test_forward_forks_then_joins_projection(self):
        hidden = torch.ones(160, 1)
        projected = torch.zeros_like(hidden)
        batch = SimpleNamespace(forward_mode=ForwardMode.TARGET_VERIFY)
        self.model.g_proj = self.streams.project
        self.streams.project.return_value = (projected, None)
        self.streams.select.return_value = nullcontext()
        self.enterContext(patch.object(torch.cuda, "stream", self.streams.select))

        def attention(positions, x, forward_batch, zero_allocator):
            self.assertIs(self.model._gate_hidden_states, hidden)
            self.assertIs(self.model._gate_pending_stream, self.streams.gate)
            self.streams.attention()
            return self.model._compute_output_gate(x)

        with patch.object(
            kimi_k3.DeepseekV2AttentionMLA, "forward", side_effect=attention
        ) as attention_forward:
            self.assertIs(self.model(None, hidden, batch, None), projected)
        attention_forward.assert_called_once_with(None, hidden, batch, None)
        self.assertIsNone(self.model._gate_pending_stream)
        self.assertEqual(
            self.streams.mock_calls,
            [
                call.gate.wait_stream(self.streams.main),
                call.attention(),
                call.select(self.streams.gate),
                call.project(hidden),
                call.main.wait_stream(self.streams.gate),
            ],
        )

    def test_late_fork_between_prepare_and_core(self):
        self.model._gate_overlap_verify_160_late = True
        hidden = torch.ones(160, 1)
        batch = SimpleNamespace(forward_mode=ForwardMode.TARGET_VERIFY)
        state = object()
        self.streams.prepare.return_value = state
        self.streams.core.return_value = hidden
        self.model.forward_prepare = self.streams.prepare
        self.model.forward_core = self.streams.core
        result = self.model(
            None, hidden, batch, None, input_on_attention_tp_slices=True
        )
        self.assertIs(result, hidden)
        self.assertIs(self.model._gate_hidden_states, hidden)
        self.assertIs(self.model._gate_pending_stream, self.streams.gate)
        self.assertEqual(
            self.streams.mock_calls,
            [
                call.prepare(
                    None, hidden, batch, None, input_on_attention_tp_slices=True
                ),
                call.gate.wait_stream(self.streams.main),
                call.core(state),
            ],
        )

    def test_late_fork_preserves_other_forwards(self):
        self.model._gate_overlap_verify_160_late = True
        mode = ForwardMode.TARGET_VERIFY
        for rows, forward_mode, capture, breakable, has_stream in (
            (64, mode, True, False, True),
            (159, mode, True, False, True),
            (161, mode, True, False, True),
            (160, ForwardMode.DECODE, True, False, True),
            (160, mode, False, False, True),
            (160, mode, True, True, True),
            (160, mode, True, False, False),
        ):
            with self.subTest(
                rows=rows,
                mode=forward_mode,
                capture=capture,
                breakable=breakable,
                has_stream=has_stream,
            ):
                self.capture.return_value = capture
                self.breakable.return_value = breakable
                self.model._gate_alt_stream = self.streams.gate if has_stream else None
                hidden = torch.ones(rows, 1)
                batch = SimpleNamespace(forward_mode=forward_mode)
                with patch.object(kimi_k3.DeepseekV2AttentionMLA, "forward") as parent:
                    self.model(None, hidden, batch, None)
                parent.assert_called_once_with(None, hidden, batch, None)
                self.assertEqual(
                    self.model._gate_pending_stream is not None, rows == 64
                )


if __name__ == "__main__":
    unittest.main()
