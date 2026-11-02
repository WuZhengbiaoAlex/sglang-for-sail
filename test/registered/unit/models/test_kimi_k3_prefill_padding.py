"""Real attention rows and KV write locations must be trimmed together."""

import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.models.kimi_k3 import KimiK3DecoderLayer

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestKimiPrefillPadding(CustomTestCase):
    def test_attention_and_kv_locations_have_the_same_real_rows(self):
        for fail in (False, True):
            with self.subTest(fail=fail):
                loc = torch.arange(1184)
                batch = SimpleNamespace(
                    forward_mode=SimpleNamespace(
                        is_idle=lambda: False, is_extend=lambda: True
                    ),
                    extend_seq_lens_cpu=[1176],
                    out_cache_loc=loc,
                )

                def attention(hidden, positions, forward_batch, allocator):
                    self.assertEqual(hidden.shape[0], 1176)
                    self.assertEqual(positions.numel(), 1176)
                    self.assertEqual(forward_batch.out_cache_loc.numel(), 1176)
                    torch.testing.assert_close(
                        forward_batch.out_cache_loc, loc[:1176]
                    )
                    if fail:
                        raise ValueError("attention failed")
                    return hidden + 1

                layer = SimpleNamespace(
                    _trim_padded_attn=True, _run_self_attn_inner=attention
                )
                with (
                    patch(
                        "sglang.srt.models.kimi_k3.k3_sp_collective.o_proj_output_rows",
                        return_value=nullcontext(),
                    ),
                    patch(
                        "sglang.srt.models.kimi_k3.k3_sp_collective.finish_padded_o_proj_output",
                        return_value=None,
                    ),
                ):
                    if fail:
                        with self.assertRaisesRegex(ValueError, "attention failed"):
                            KimiK3DecoderLayer._run_self_attn(
                                layer, torch.zeros(1184, 4), loc, batch, None
                            )
                    else:
                        result = KimiK3DecoderLayer._run_self_attn(
                            layer, torch.zeros(1184, 4), loc, batch, None
                        )
                        torch.testing.assert_close(result[:1176], torch.ones(1176, 4))
                        torch.testing.assert_close(result[1176:], torch.zeros(8, 4))
                self.assertIs(batch.out_cache_loc, loc)


if __name__ == "__main__":
    unittest.main()
