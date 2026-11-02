"""CPU contract checks for the PPU FlashKDA prefill and verify adapter."""

import types
import unittest
from unittest.mock import patch

import torch

from sglang.srt.layers.attention.linear.kernels import kda_flashkda
from sglang.srt.layers.attention.linear.kernels.kda_triton import TritonKDAKernel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestPPUFlashKDA(CustomTestCase):
    def test_decode_packs_qkv_and_forwards_raw_gate_and_safe_bound(self):
        kernel = kda_flashkda.FlashKDAKernel()
        q = torch.ones((1, 2, 6, 128), dtype=torch.bfloat16)
        k, v = q * 2, q * 3
        a = torch.full((1, 2, 768), 4, dtype=torch.bfloat16)
        b = torch.full((1, 2, 6), 5, dtype=torch.bfloat16)
        states = torch.zeros((3, 6, 128, 128))
        slots = torch.tensor([1, 0], dtype=torch.int32)
        calls = []

        def decode(*args, **kwargs):
            calls.append((args, kwargs))
            args[-1].fill_(7)

        module = types.SimpleNamespace(decode=decode, supports_decode_safe_gate=True)
        with patch.object(kda_flashkda, "is_ppu", return_value=True), patch.object(
            kda_flashkda, "_load_flash_kda", return_value=module
        ):
            out = kernel.decode(
                q, k, v, a, b,
                A_log=torch.zeros(6), dt_bias=torch.zeros(768),
                ssm_states=states, cache_indices=slots,
                query_start_loc=torch.arange(3), lower_bound=-5.0,
            )
        self.assertEqual(out.shape, q.shape)
        self.assertTrue(torch.all(out == 7))
        args, kwargs = calls[0]
        torch.testing.assert_close(args[0], torch.cat([x.reshape(2, 768) for x in (q, k, v)], -1))
        torch.testing.assert_close(args[1], a.reshape(2, 768))
        torch.testing.assert_close(args[2], b.reshape(2, 6))
        self.assertIs(args[6], states)
        torch.testing.assert_close(args[7], slots)
        self.assertEqual(kwargs["lower_bound"], -5.0)

    def test_decode_safe_gate_rejects_an_old_wheel(self):
        module = types.SimpleNamespace(decode=lambda *args: None)
        q = torch.zeros((1, 1, 6, 128), dtype=torch.bfloat16)
        with patch.object(kda_flashkda, "is_ppu", return_value=True), patch.object(
            kda_flashkda, "_load_flash_kda", return_value=module
        ), self.assertRaisesRegex(RuntimeError, "safe-gate"):
            kda_flashkda.FlashKDAKernel().decode(
                q, q, q, q.flatten(2), torch.zeros(1, 1, 6),
                A_log=torch.zeros(6), dt_bias=torch.zeros(768),
                ssm_states=torch.zeros(2, 6, 128, 128),
                cache_indices=torch.tensor([0]), query_start_loc=torch.arange(2),
                lower_bound=-5.0,
            )

    def test_unsupported_ppu_prefill_layout_uses_triton(self):
        kernel = kda_flashkda.FlashKDAKernel()
        q = torch.zeros((1, 64, 6, 64), dtype=torch.bfloat16)
        sentinel = object()
        with patch.object(kda_flashkda, "is_ppu", return_value=True), patch.object(
            kda_flashkda, "_triton_fallback", return_value=sentinel
        ) as fallback, patch.object(kernel, "_flashkda_extend") as native:
            result = kernel.extend(
                q,
                q,
                q,
                q,
                torch.zeros((1, 64, 6), dtype=torch.bfloat16),
                ssm_states=torch.zeros((2, 6, 64, 64)),
                cache_indices=torch.tensor([0]),
                query_start_loc=torch.tensor([0, 64]),
                A_log=torch.zeros(6),
                dt_bias=torch.zeros(6 * 64),
                lower_bound=-5.0,
            )
        self.assertIs(result, sentinel)
        fallback.assert_called_once()
        native.assert_not_called()

    def test_aligned_radix_track_uses_native_final_state(self):
        kernel = kda_flashkda.FlashKDAKernel()
        q = torch.zeros((1, 64, 6, 128), dtype=torch.bfloat16)
        out = torch.ones_like(q)
        with patch.object(
            kernel, "_should_fall_back", return_value=False
        ), patch.object(
            kernel, "_flashkda_extend", return_value=out
        ) as native, patch.object(
            kda_flashkda, "_triton_fallback"
        ) as fallback:
            result = kernel.extend(
                q,
                q,
                q,
                q,
                torch.zeros((1, 64, 6)),
                ssm_states=torch.zeros((2, 6, 128, 128)),
                cache_indices=torch.tensor([0]),
                query_start_loc=torch.tensor([0, 64]),
                A_log=torch.zeros(6),
                dt_bias=torch.zeros(6 * 128),
                lower_bound=-5.0,
                return_intermediate_states=True,
                track_ssm_h_src=torch.empty(0, dtype=torch.int32),
                track_state=None,
            )
        self.assertIs(result[0], out)
        self.assertIsNone(result[1])
        native.assert_called_once()
        fallback.assert_not_called()

    def test_unaligned_radix_track_keeps_snapshot_fallback(self):
        kernel = kda_flashkda.FlashKDAKernel()
        q = torch.zeros((1, 64, 6, 128), dtype=torch.bfloat16)
        sentinel = object()
        with patch.object(
            kernel, "_should_fall_back", return_value=False
        ), patch.object(kernel, "_flashkda_extend") as native, patch.object(
            kda_flashkda, "_triton_fallback", return_value=sentinel
        ) as fallback:
            result = kernel.extend(
                q,
                q,
                q,
                q,
                torch.zeros((1, 64, 6)),
                ssm_states=torch.zeros((2, 6, 128, 128)),
                cache_indices=torch.tensor([0]),
                query_start_loc=torch.tensor([0, 64]),
                lower_bound=-5.0,
                return_intermediate_states=True,
                track_ssm_h_src=torch.tensor([0], dtype=torch.int32),
                track_state=torch.empty((1, 6, 128, 128)),
            )
        self.assertIs(result, sentinel)
        fallback.assert_called_once()
        native.assert_not_called()

    def test_dense_verify_uses_wheel_and_keeps_committed_state(self):
        kernel = kda_flashkda.FlashKDAKernel()
        q = torch.zeros((1, 16, 6, 128), dtype=torch.bfloat16)
        state = torch.zeros((5, 6, 128, 128), dtype=torch.bfloat16)
        scratch = torch.zeros((5, 8, 6, 128, 128), dtype=torch.bfloat16)
        calls = []

        def verify(*args):
            calls.append(args)
            args[-1].fill_(2)

        module = types.SimpleNamespace(verify_strided=verify)
        with patch.object(
            kda_flashkda, "_load_flash_kda", return_value=module
        ), patch.object(kda_flashkda, "is_ppu", return_value=True):
            out = kernel.target_verify(
                A_log=torch.zeros(6),
                dt_bias=torch.zeros(6 * 128),
                q=q,
                k=q,
                v=q,
                a=q,
                b=torch.zeros((1, 16, 6), dtype=torch.bfloat16),
                ssm_states=state,
                cache_indices=torch.tensor([0, 1]),
                query_start_loc=torch.tensor([0, 8, 16]),
                intermediate_states_buffer=scratch,
                intermediate_state_indices=torch.tensor([0, 1, 2, 3, 4]),
                cache_steps=8,
                retrieve_parent_token=None,
                lower_bound=-4.0,
            )
        self.assertEqual(out.shape, q.shape)
        self.assertTrue(torch.all(out == 2))
        self.assertTrue(torch.all(state == 0))
        self.assertEqual(len(calls), 1)

    def test_tree_verify_uses_triton(self):
        kernel = kda_flashkda.FlashKDAKernel()
        q = torch.zeros((1, 16, 6, 128), dtype=torch.bfloat16)
        sentinel = object()
        with patch.object(
            kda_flashkda,
            "_load_flash_kda",
            return_value=types.SimpleNamespace(verify_strided=lambda *args: None),
        ), patch.object(kda_flashkda, "is_ppu", return_value=True), patch.object(
            TritonKDAKernel, "target_verify", return_value=sentinel
        ) as fallback:
            result = kernel.target_verify(
                A_log=torch.zeros(6),
                dt_bias=torch.zeros(768),
                q=q,
                k=q,
                v=q,
                a=q,
                b=torch.zeros((1, 16, 6), dtype=torch.bfloat16),
                ssm_states=torch.zeros((5, 6, 128, 128), dtype=torch.bfloat16),
                cache_indices=torch.tensor([0, 1]),
                query_start_loc=torch.tensor([0, 8, 16]),
                intermediate_states_buffer=torch.zeros((5, 8, 6, 128, 128)),
                intermediate_state_indices=torch.arange(5),
                cache_steps=8,
                retrieve_parent_token=torch.zeros(16, dtype=torch.int32),
                lower_bound=-4.0,
            )
        self.assertIs(result, sentinel)
        fallback.assert_called_once()


if __name__ == "__main__":
    unittest.main()
