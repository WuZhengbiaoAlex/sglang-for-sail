import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.models import kimi_k3
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=4, suite="base-a-test-cpu")


class TestKimiKDAGateNormFP8(CustomTestCase):
    def setUp(self):
        super().setUp()
        self.core = torch.ones((1, 5, 12, 128), dtype=torch.bfloat16)
        self.gate = torch.zeros((5, 4, 1536), dtype=torch.bfloat16)[:, -1]
        self.projection = Mock(return_value=(torch.ones(5, 4), None))
        self.projection._ppu_w8a8_fp8_enabled = True
        self.projection.quant_method.accepts_prequantized_input = True
        self.norm = Mock(return_value=self.core)
        self.norm.weight = torch.ones(128, dtype=torch.bfloat16)
        self.norm.eps = 1e-6
        self.model = SimpleNamespace(
            _use_kda_gate_norm_fp8=True,
            _kda_hip_fused_decode_ready=False,
            _kda_fused_decode_ready=False,
            _cp_tp_group=True,
            head_dim=128,
            all_reduce_fusion=False,
            do_fuse_qkvbfg=True,
            use_full_rank_gate=True,
            attn=Mock(return_value=self.core),
            forward_qkvbfg_fused=Mock(
                return_value=(None, torch.ones(5), torch.ones(5, 1536), self.gate)
            ),
            o_norm=self.norm,
            o_proj=self.projection,
        )
        self.hidden = torch.empty((5, 4), dtype=torch.bfloat16)
        self.qpair = (object(), object())
        self.fused = self.enterContext(
            patch(
                "sglang.kernels.ops.attention.fla.kda_gate_norm_fp8.kda_gate_norm_fp8",
                return_value=self.qpair,
            )
        )
        self.cp = self.enterContext(
            patch.object(kimi_k3, "is_cp_active", return_value=False)
        )
        self.ctx = SimpleNamespace(sp_active=False)
        self.enterContext(patch.object(kimi_k3, "get_forward", return_value=self.ctx))

    def forward(self, mode):
        return kimi_k3.KimiK3DeltaAttention.forward(
            self.model, self.hidden, None, SimpleNamespace(forward_mode=mode), None
        )

    def test_split_gate_goes_straight_to_projection(self):
        for mode in (ForwardMode.TARGET_VERIFY, ForwardMode.DECODE):
            with self.subTest(mode=mode):
                self.fused.reset_mock()
                output = self.forward(mode)
                self.fused.assert_called_once()
                x, gate, weight, eps = self.fused.call_args.args
                self.assertEqual(tuple(x.shape), (5, 12, 128))
                self.assertEqual(x.data_ptr(), self.core.data_ptr())
                self.assertEqual(gate.data_ptr(), self.gate.data_ptr())
                self.assertEqual(gate.stride(), (6144, 128, 1))
                self.assertIs(weight, self.norm.weight)
                self.assertEqual(eps, self.norm.eps)
                self.assertIs(self.projection.call_args.args[0], self.qpair)
                self.assertIs(output, self.projection.return_value[0])
                self.norm.assert_not_called()

    def test_unsupported_paths_keep_original_norm_and_projection(self):
        for case in (
            "flag_off",
            "non_fp8_weight",
            "no_tuple_support",
            "prefill",
            "sp",
            "not_bf16",
            "empty",
        ):
            with self.subTest(case=case):
                self.norm.reset_mock()
                self.model._use_kda_gate_norm_fp8 = case != "flag_off"
                self.projection._ppu_w8a8_fp8_enabled = case != "non_fp8_weight"
                self.projection.quant_method.accepts_prequantized_input = (
                    case != "no_tuple_support"
                )
                self.ctx.sp_active = case == "sp"
                self.norm.weight = self.norm.weight.to(
                    torch.float16 if case == "not_bf16" else torch.bfloat16
                )
                self.model.attn.return_value = (
                    self.core[:, :0] if case == "empty" else self.core
                )
                self.norm.return_value = self.model.attn.return_value
                if case == "empty":
                    self.model.forward_qkvbfg_fused.return_value = (
                        None,
                        torch.empty(0),
                        torch.empty(0, 1536),
                        self.gate[:0],
                    )
                self.forward(
                    ForwardMode.EXTEND
                    if case == "prefill"
                    else ForwardMode.TARGET_VERIFY
                )
                self.fused.assert_not_called()
                self.norm.assert_called_once()
                self.assertIsInstance(self.projection.call_args.args[0], torch.Tensor)

    def test_context_parallel_keeps_original_reduce_scatter(self):
        self.cp.return_value = True
        parallel = SimpleNamespace(attn_cp_size=1)
        with (
            patch.object(kimi_k3, "get_parallel", return_value=parallel),
            patch.object(
                kimi_k3, "cp_gather_full_sequence_states", return_value=self.hidden
            ),
            patch.object(
                kimi_k3, "cp_sequence_to_interleave_order", side_effect=lambda x, *_: x
            ),
            patch.object(kimi_k3, "attn_cp_reduce_scatter_tensor") as scatter,
        ):
            output = self.forward(ForwardMode.TARGET_VERIFY)
        self.assertEqual(tuple(output.shape), tuple(self.hidden.shape))
        self.assertTrue(self.projection.call_args.kwargs["skip_all_reduce"])
        self.norm.assert_called_once()
        scatter.assert_called_once()
        self.fused.assert_not_called()

    def test_already_consumed_norm_is_not_reapplied(self):
        self.model._kda_fused_decode_ready = True

        def attention(*args, **kwargs):
            self.model.attn._k3_onorm_consumed = True
            return self.core

        self.model.attn.side_effect = attention
        self.forward(ForwardMode.TARGET_VERIFY)
        self.norm.assert_not_called()
        self.fused.assert_not_called()
        self.assertIsInstance(self.projection.call_args.args[0], torch.Tensor)


if __name__ == "__main__":
    unittest.main()
