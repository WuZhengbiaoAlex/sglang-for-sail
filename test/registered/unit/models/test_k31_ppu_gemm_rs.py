"""Model-owned GEMM+RS scopes must preserve K3.1's row/residual contract."""

# ruff: noqa: E402 -- install CPU kernel stubs before importing model modules.

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.layers.communication import k31_ppu_gemm_rs as rs
from sglang.srt.layers.quantization.ppu_dense_w8a8_fp8 import (
    PPUDenseW8A8FP8LinearMethod,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.models.kimi_k3 import KimiK3DecoderLayer
from sglang.srt.runtime_context import get_forward

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestK31PPUGemmRS(CustomTestCase):
    def setUp(self):
        self.resource = SimpleNamespace(
            max_rows=320,
            group=SimpleNamespace(world_size=8, rank_in_group=3),
            apply=Mock(return_value=torch.empty(20, 7168, dtype=torch.bfloat16)),
        )
        self.handle = rs.FusedOProjReduceScatter(self.resource)
        self.batch = SimpleNamespace(forward_mode=ForwardMode.TARGET_VERIFY)

    def test_verify_scope_is_restored_on_exception(self):
        sentinel = object()
        with get_forward().scoped(k31_o_proj_rs_call=sentinel):
            with self.assertRaisesRegex(RuntimeError, "attention failed"):
                with self.handle.scope(self.batch, 160) as call:
                    self.assertIs(get_forward().k31_o_proj_rs_call, call)
                    raise RuntimeError("attention failed")
            self.assertIs(get_forward().k31_o_proj_rs_call, sentinel)

    def test_non_verify_idle_and_unsupported_shapes_fall_back(self):
        for mode in (
            ForwardMode.EXTEND,
            ForwardMode.DECODE,
            ForwardMode.DRAFT_EXTEND_V2,
            ForwardMode.IDLE,
        ):
            with self.subTest(mode=mode):
                batch = SimpleNamespace(forward_mode=mode)
                with self.handle.scope(batch, 160) as call:
                    self.assertIsNone(call)
        for rows in (0, 159, 328):
            with self.subTest(rows=rows), self.handle.scope(self.batch, rows) as call:
                self.assertIsNone(call)
        self.batch._original_forward_mode = ForwardMode.IDLE
        with self.handle.scope(self.batch, 160) as call:
            self.assertIsNone(call)
        del self.batch._original_forward_mode
        with get_forward().scoped(sp_active=True):
            with self.handle.scope(self.batch, 160) as call:
                self.assertIsNone(call)

    def test_quantized_projection_returns_only_explicit_scope_shard(self):
        # Meta tensors exercise dtype/layout guards without allocating weights.
        x = torch.empty(160, 1536, dtype=torch.float8_e4m3fn, device="meta")
        x_scale = torch.empty(160, 1, dtype=torch.float32, device="meta")
        layer = SimpleNamespace(
            weight=torch.empty(7168, 1536, dtype=torch.float8_e4m3fn, device="meta"),
            _ppu_w8a8_fp8_weight_scale=torch.empty(
                7168, 1, dtype=torch.float32, device="meta"
            ),
            _ppu_w8a8_fp8_enabled=True,
            _ppu_o_proj_rs=self.handle,
            reduce_results=False,
            use_decode_attn_tp=False,
        )
        quant = PPUDenseW8A8FP8LinearMethod("test.o_proj")
        with patch(
            "sglang.srt.layers.quantization.ppu_dense_w8a8_fp8.gemm_nt_f8f8bf16"
        ) as old_gemm:
            with self.handle.scope(self.batch, 160) as call:
                out = quant.apply(layer, (x, x_scale))
                self.assertEqual(tuple(out.shape), (20, 7168))
                self.assertEqual(call.shard_lo, 60)
                old_gemm.assert_not_called()
            # The same generic linear returns a full tensor outside the model
            # scope. It cannot accidentally carry sharded state into the draft.
            out = quant.apply(layer, (x, x_scale))
            self.assertEqual(tuple(out.shape), (160, 7168))
            old_gemm.assert_called_once()
        self.resource.apply.assert_called_once()

    def test_decoder_skips_second_rs_and_slices_full_prefix(self):
        self._check_decoder_prefix(already_sharded=False)

    def test_decoder_keeps_already_local_prefix(self):
        self._check_decoder_prefix(already_sharded=True)

    def _check_decoder_prefix(self, already_sharded):
        full_prefix = torch.arange(160 * 3).view(160, 3)
        prefix = full_prefix[60:80] if already_sharded else full_prefix
        reduced = torch.ones(20, 3)

        def attention(hidden, positions, batch, allocator):
            get_forward().k31_o_proj_rs_call.shard_lo = 60
            return reduced

        forward = Mock(side_effect=[(torch.zeros(160, 3), prefix), (reduced, prefix)])
        attn_res = SimpleNamespace(
            forward=forward,
            forward_sp_reduce_scatter=Mock(side_effect=AssertionError("duplicate RS")),
        )
        layer = SimpleNamespace(
            _ppu_o_proj_rs=self.handle,
            _sp_moe=True,
            all_reduce_fusion=False,
            is_block_write_layer=False,
            self_attention_res_proj=None,
            self_attention_res_norm=None,
            input_layernorm=None,
            mlp_res_proj=None,
            mlp_res_norm=None,
            post_attention_layernorm=None,
            _run_self_attn=attention,
            _finish_attn_reduce=Mock(side_effect=AssertionError("duplicate RS")),
            mlp=Mock(return_value=reduced),
        )
        out, residual, sharded = KimiK3DecoderLayer._forward_attn_residual(
            layer,
            None,
            torch.zeros(160, 3),
            prefix,
            attn_res,
            self.batch,
            None,
            input_sharded=False,
            keep_sharded=True,
        )
        self.assertIs(out, reduced)
        self.assertIsNone(residual)
        self.assertTrue(sharded)
        args, kwargs = forward.call_args_list[1]
        torch.testing.assert_close(args[1], full_prefix[60:80])
        self.assertEqual(kwargs["rows"], slice(60, 80))
        layer._finish_attn_reduce.assert_not_called()
        attn_res.forward_sp_reduce_scatter.assert_not_called()

    def test_first_dense_excluded_and_workspace_shared(self):
        def make_layer(sp_moe):
            proj = SimpleNamespace(
                quant_method=PPUDenseW8A8FP8LinearMethod("test.o_proj"),
                weight=torch.empty(7168, 1536, device="meta"),
                bias=None,
                reduce_results=False,
            )
            return SimpleNamespace(
                _sp_moe=sp_moe, self_attn=SimpleNamespace(o_proj=proj)
            )

        layers = [make_layer(False), make_layer(True), make_layer(True)]
        group = SimpleNamespace(
            world_size=8,
            rank_in_group=3,
            device_group=SimpleNamespace(group_name="attn_tp"),
        )
        model = SimpleNamespace(
            config=SimpleNamespace(
                dsa_config={}, attn_res_block_size=4, hidden_size=7168
            ),
            pp_group=SimpleNamespace(world_size=1),
            layers=layers,
        )
        with (
            patch.object(
                rs.envs.SGLANG_PPU_K31_FUSE_O_PROJ_RS, "get", return_value=True
            ),
            patch.object(rs, "is_ppu", return_value=True),
            patch.object(
                rs,
                "get_flags",
                return_value=SimpleNamespace(
                    moe=SimpleNamespace(in_speculative_scope=False),
                ),
            ),
            patch.object(rs.k3_sp_collective, "enabled", return_value=False),
            patch.object(
                rs,
                "get_parallel",
                return_value=SimpleNamespace(
                    attn_tp_group=group,
                    attn_cp_size=1,
                    attn_dcp_size=1,
                ),
            ),
            patch.object(
                rs,
                "get_exec",
                return_value=SimpleNamespace(
                    graph=SimpleNamespace(
                        cuda_graph_config=SimpleNamespace(
                            decode=SimpleNamespace(max_bs=64, bs=[8, 16, 32, 64]),
                        )
                    ),
                ),
            ),
            patch.object(
                rs,
                "get_spec",
                return_value=SimpleNamespace(
                    speculative_num_draft_tokens=5,
                ),
            ),
            patch.object(rs.torch.cuda, "current_device", return_value=0),
            patch.object(
                rs, "_SharedWorkspace", return_value=self.resource
            ) as allocate,
            patch.object(rs, "get_buffer", side_effect=lambda name, factory: factory()),
        ):
            rs.initialize_model(model)
        allocate.assert_called_once_with(group, 320, 7168)
        self.assertFalse(hasattr(layers[0], "_ppu_o_proj_rs"))
        for layer in layers[1:]:
            self.assertIs(layer._ppu_o_proj_rs.resource, self.resource)
            self.assertIs(layer.self_attn.o_proj._ppu_o_proj_rs, layer._ppu_o_proj_rs)

    def test_draft_construction_never_initializes_collectives(self):
        with (
            patch.object(
                rs.envs.SGLANG_PPU_K31_FUSE_O_PROJ_RS, "get", return_value=True
            ),
            patch.object(rs, "is_ppu", return_value=True),
            patch.object(
                rs,
                "get_flags",
                return_value=SimpleNamespace(
                    moe=SimpleNamespace(in_speculative_scope=True),
                ),
            ),
            patch.object(rs, "get_buffer") as allocate,
        ):
            rs.initialize_model(SimpleNamespace())
        allocate.assert_not_called()


if __name__ == "__main__":
    unittest.main()
