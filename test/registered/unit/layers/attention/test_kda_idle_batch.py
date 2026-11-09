"""CPU regressions for logical-IDLE routing and zero-row KDA execution."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
    HybridLinearAttnBackend,
)
from sglang.srt.layers.attention.linear.kda_backend import KDAAttnBackend
from sglang.srt.layers.attention.mamba.mamba2_metadata import ForwardMetadata
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.models.kimi_k3 import KimiK3DecoderLayer
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestKDAIdleBatch(CustomTestCase):
    def make_hybrid(self, linear=None):
        backend = HybridLinearAttnBackend.__new__(HybridLinearAttnBackend)
        backend.full_attn_layers = [1]
        backend.full_attn_backend = Mock()
        backend.linear_attn_backend = linear if linear is not None else Mock()
        backend.attn_backend_list = [
            backend.full_attn_backend,
            backend.linear_attn_backend,
        ]
        return backend

    def test_kimi_preserves_empty_and_padded_logical_idle_without_attention(self):
        layer = SimpleNamespace(_run_self_attn_inner=Mock(), _trim_padded_attn=False)
        for rows in (0, 5, 160):
            for mode in (ForwardMode.IDLE, ForwardMode.TARGET_VERIFY):
                with self.subTest(rows=rows, mode=mode):
                    hidden = torch.arange(rows * 4, dtype=torch.float32).view(rows, 4)
                    batch = SimpleNamespace(
                        forward_mode=mode, _original_forward_mode=ForwardMode.IDLE
                    )
                    out = KimiK3DecoderLayer._run_self_attn(
                        layer, hidden, torch.arange(rows), batch, None
                    )
                    self.assertIs(out, hidden)
                    layer._run_self_attn_inner.assert_not_called()

        hidden = torch.ones(5, 4)
        layer._run_self_attn_inner.return_value = hidden + 7
        batch = SimpleNamespace(forward_mode=ForwardMode.TARGET_VERIFY)
        out = KimiK3DecoderLayer._run_self_attn(
            layer, hidden, torch.arange(5), batch, None
        )
        layer._run_self_attn_inner.assert_called_once()
        torch.testing.assert_close(out, hidden + 7)

    def test_hybrid_rewritten_idle_skips_full_metadata_and_returns_finite_zeros(self):
        backend = self.make_hybrid()
        idle = SimpleNamespace(
            forward_mode=ForwardMode.TARGET_VERIFY,
            _original_forward_mode=ForwardMode.IDLE,
        )
        backend.init_forward_metadata(idle)
        backend.full_attn_backend.init_forward_metadata.assert_not_called()
        backend.linear_attn_backend.init_forward_metadata.assert_called_once_with(idle)
        full_layer = SimpleNamespace(layer_id=1, tp_q_head_num=2, v_head_dim=4)
        for rows in (0, 8):
            q = torch.full((rows, 2, 4), float("nan"))
            out = backend.forward(q=q, layer=full_layer, forward_batch=idle)
            self.assertEqual(out.shape, (rows, 8))
            self.assertTrue(torch.all(out == 0))
        backend.full_attn_backend.forward_extend.assert_not_called()

        # Plain IDLE's padding is also initialized, for both attention families.
        idle.forward_mode = ForwardMode.IDLE
        linear_layer = SimpleNamespace(layer_id=0, num_v_heads=2, head_v_dim=4)
        out = backend.forward(
            mixed_qkv=torch.full((8, 24), float("nan")),
            layer=linear_layer,
            forward_batch=idle,
        )
        self.assertEqual(out.shape, (8, 2, 4))
        self.assertTrue(torch.all(out == 0))

        busy = SimpleNamespace(forward_mode=ForwardMode.TARGET_VERIFY)
        backend.full_attn_backend.reset_mock()
        backend.linear_attn_backend.reset_mock()
        backend.init_forward_metadata(busy)
        backend.full_attn_backend.init_forward_metadata.assert_called_once_with(busy)
        backend.linear_attn_backend.init_forward_metadata.assert_called_once_with(busy)
        sentinel = torch.ones(5, 8)
        backend.full_attn_backend.forward_extend.return_value = sentinel
        self.assertIs(
            backend.forward(
                q=torch.ones(5, 2, 4), layer=full_layer, forward_batch=busy
            ),
            sentinel,
        )

    def test_empty_kda_modes_reset_metadata_and_never_touch_state_or_kernels(self):
        backend = KDAAttnBackend.__new__(KDAAttnBackend)
        backend.req_to_token_pool = Mock()
        backend.kernel_dispatcher = Mock()
        hybrid = self.make_hybrid(backend)
        layer = SimpleNamespace(layer_id=0, num_v_heads=2, head_v_dim=4)
        for mode in (
            ForwardMode.IDLE,
            ForwardMode.EXTEND,
            ForwardMode.TARGET_VERIFY,
            ForwardMode.DRAFT_EXTEND_V2,
            ForwardMode.DECODE,
        ):
            with self.subTest(mode=mode):
                batch = SimpleNamespace(
                    batch_size=0,
                    forward_mode=mode,
                    req_pool_indices=torch.empty(0, dtype=torch.int64),
                )
                backend.forward_metadata = object()
                backend.init_forward_metadata(batch)
                self.assertEqual(backend.forward_metadata.query_start_loc.tolist(), [0])
                self.assertEqual(
                    backend.forward_metadata.mamba_cache_indices.numel(), 0
                )
                self.assertEqual(backend.forward_metadata.logical_num_tokens, 0)
                self.assertFalse(backend.forward_metadata.has_mamba_track_mask)
                out = hybrid.forward(
                    layer=layer,
                    forward_batch=batch,
                    mixed_qkv=torch.empty(0, 24, dtype=torch.bfloat16),
                    a=torch.empty(1, 0, 2, 4),
                    b=torch.empty(1, 0, 2),
                )
                expected_shape = (0, 2, 4) if mode.is_idle() else (1, 0, 2, 4)
                self.assertEqual(out.shape, expected_shape)
                self.assertEqual(out.dtype, torch.bfloat16)
        backend.req_to_token_pool.assert_not_called()
        self.assertEqual(backend.req_to_token_pool.mock_calls, [])
        self.assertEqual(backend.kernel_dispatcher.mock_calls, [])

    def test_nonempty_width_five_verify_keeps_conv_and_dispatch_contract(self):
        backend = KDAAttnBackend.__new__(KDAAttnBackend)
        layer = SimpleNamespace(
            layer_id=0,
            num_q_heads=2,
            num_v_heads=2,
            head_q_dim=4,
            head_k_dim=4,
            head_v_dim=4,
            q_dim=8,
            k_dim=8,
            v_dim=8,
            conv_weights=torch.ones(24, 3),
            bias=None,
            A_log=torch.zeros(2),
            dt_bias=torch.zeros(8),
            lower_bound=-5.0,
        )
        cache = SimpleNamespace(
            conv=[torch.zeros(3, 2, 24)],
            temporal=torch.zeros(3, 2, 4, 4),
            intermediate_ssm=torch.zeros(3, 5, 2, 4, 4),
            intermediate_conv_window=[torch.zeros(3, 5, 2, 24)],
            replayssm_rawv=None,
        )
        backend.req_to_token_pool = SimpleNamespace(mamba2_layer_cache=lambda _: cache)
        backend.forward_metadata = ForwardMetadata(
            query_start_loc=torch.tensor([0, 5, 10], dtype=torch.int32),
            mamba_cache_indices=torch.tensor([1, 2], dtype=torch.int32),
        )
        backend.verify_intermediate_state_indices = torch.arange(3, dtype=torch.int32)
        backend.accept_lens_pool = None
        backend._can_run_dspark_cutedsl_mtp = Mock(return_value=False)
        backend._can_run_fused_chain_verify = Mock(return_value=False)
        expected = torch.ones(1, 10, 2, 4)
        backend.kernel_dispatcher = SimpleNamespace(
            target_verify=Mock(return_value=expected)
        )
        batch = SimpleNamespace(
            forward_mode=ForwardMode.TARGET_VERIFY,
            spec_info=SimpleNamespace(draft_token_num=5, ragged_verify_layout=None),
        )
        mixed = torch.arange(240, dtype=torch.float32).view(10, 24)
        a, b = torch.ones(1, 10, 2, 4), torch.ones(1, 10, 2)
        with patch(
            "sglang.srt.layers.attention.linear.kda_backend.causal_conv1d_update",
            side_effect=lambda x, *args, **kwargs: x * 2,
        ) as conv:
            out = backend.forward_extend(layer, batch, mixed, a, b)
        self.assertIs(out, expected)
        self.assertEqual(conv.call_args.args[0].shape, (2, 24, 5))
        kwargs = backend.kernel_dispatcher.target_verify.call_args.kwargs
        torch.testing.assert_close(kwargs["q"], (mixed[:, :8] * 2).view(1, 10, 2, 4))
        torch.testing.assert_close(kwargs["k"], (mixed[:, 8:16] * 2).view(1, 10, 2, 4))
        torch.testing.assert_close(kwargs["v"], (mixed[:, 16:] * 2).view(1, 10, 2, 4))
        self.assertEqual(kwargs["cache_steps"], 5)
        self.assertIs(kwargs["intermediate_states_buffer"], cache.intermediate_ssm)
        self.assertIs(kwargs["a"], a)
        self.assertIs(kwargs["b"], b)


if __name__ == "__main__":
    unittest.main()
