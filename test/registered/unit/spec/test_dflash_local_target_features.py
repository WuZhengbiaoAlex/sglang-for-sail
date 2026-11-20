import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardMode
from sglang.srt.models import kimi_k3
from sglang.srt.models.dflash import DFlashDraftModel
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestDFlashLocalTargetFeatures(unittest.TestCase):
    def _run_kimi_capture(
        self, *, enabled=True, full_layer=None, physical=160, grammar_layout=None
    ):
        # Distinct token rows and layer columns reveal either ordering mistake.
        rank, width = 3, 8
        local_rows = physical // width
        interval = slice(rank * local_rows, (rank + 1) * local_rows)
        group = SimpleNamespace(world_size=width, rank_in_group=rank)
        pp = SimpleNamespace(is_first_rank=True, is_last_rank=True, world_size=1)
        parallel = SimpleNamespace(
            attn_tp_size=width, attn_tp_group=group, attn_cp_size=1, pp_group=pp
        )
        features = [
            torch.stack((torch.arange(physical), torch.full((physical,), i)), dim=1).to(
                torch.bfloat16
            )
            for i in range(8)
        ]
        # The local residual stream can borrow the same storage each layer.
        storage = torch.empty_like(features[0][interval])
        tap_layers = [0, 1, 3, 4, 6, 7]
        calls = []

        class FakeLayer:
            _sp_moe = True

            def __init__(self, i):
                self.i = i

            def __call__(self, **_):
                if self.i == full_layer:
                    return features[self.i], None, False
                return storage.copy_(features[self.i][interval]), None, True

        model = SimpleNamespace(
            start_layer=0,
            end_layer=len(features),
            config=SimpleNamespace(attn_res_block_size=8, moe_layer_freq=1),
            layers=[FakeLayer(i) for i in range(len(features))],
            pp_group=pp,
            dspark_layers_to_capture=tap_layers,
            _local_dflash_features_enabled=enabled,
            _trim_padded_attn=False,
            _dspark_capture_stream=lambda i, hidden, residual, bank, rows: hidden,
            output_attn_res_norm=None,
            output_attn_res_proj=None,
            norm=None,
        )
        batch = SimpleNamespace(
            forward_mode=ForwardMode.TARGET_VERIFY,
            spec_info=SimpleNamespace(
                defer_attn_tp_logits=True, ragged_verify_layout=grammar_layout
            ),
            spec_algorithm=SpeculativeAlgorithm.DFLASH,
            capture_hidden_mode=CaptureHiddenMode.FULL,
        )
        bank = Mock()
        bank.forward_sp_all_gather.return_value = (features[-1], None)

        def collective(local):
            layer = int(local[0, 1])
            torch.testing.assert_close(local, features[layer][interval])
            calls.append(layer)
            return features[layer].clone()

        with ExitStack() as stack:
            stack.enter_context(
                patch.object(kimi_k3, "get_parallel", return_value=parallel)
            )
            stack.enter_context(patch.object(kimi_k3, "is_ppu", return_value=True))
            stack.enter_context(
                patch.object(kimi_k3, "AttnResidual", return_value=bank)
            )
            stack.enter_context(
                patch.object(kimi_k3, "_sp_all_gather_rows", side_effect=collective)
            )
            stack.enter_context(
                patch.object(
                    kimi_k3,
                    "get_moe_a2a_backend",
                    return_value=SimpleNamespace(is_megamoe=lambda: True),
                )
            )
            stack.enter_context(kimi_k3.envs.SGLANG_PPU_K31_SP_ATTN_RES.override(True))
            _, taps = kimi_k3.KimiK3LinearModel.forward(
                model, None, None, batch, inputs_embeds=features[0]
            )
        expected = torch.cat([features[i] for i in tap_layers], dim=-1)
        if isinstance(taps, kimi_k3._LocalDFlashTargetFeatures):
            torch.testing.assert_close(taps.values, expected[interval])
        else:
            torch.testing.assert_close(torch.cat(taps, dim=-1), expected)
        return calls

    def test_six_layer_order_and_reused_storage(self):
        self.assertEqual(self._run_kimi_capture(), [])
        for kwargs in (
            dict(enabled=False),
            dict(physical=80),
            dict(grammar_layout=object()),
        ):
            with self.subTest(kwargs=kwargs):
                self.assertEqual(self._run_kimi_capture(**kwargs), [0, 1, 3, 4, 6, 7])

    def test_unsharded_tap_after_local_taps(self):
        self.assertEqual(self._run_kimi_capture(full_layer=3), [0, 1, 4, 6, 7])

    def test_projection_and_partial_graph_use_original_live_row_count(self):
        rows, hidden, tp, rank = 160, 4, 8, 3
        features = torch.randn(
            rows, 6 * hidden, generator=torch.Generator().manual_seed(63)
        )
        weight = torch.randn(
            hidden, 6 * hidden, generator=torch.Generator().manual_seed(79)
        )
        call_rows = []

        def projection(x):
            # Also encode FC's input M to detect an incorrectly padded partial replay.
            call_rows.append(len(x))
            return torch.nn.functional.rms_norm(x @ weight.T, (hidden,))

        worker = SimpleNamespace(project_target_hidden=projection)
        all_projected = projection(features)
        local = features.chunk(tp)[rank]
        group = Mock(world_size=tp)
        for live in (rows, 157, 75):
            with self.subTest(live=live):

                def gather(output, input_):
                    src = all_projected if live == rows else features
                    torch.testing.assert_close(input_, src.chunk(tp)[rank])
                    output.copy_(src)

                group.all_gather_into_tensor.side_effect = gather
                call_rows.clear()
                output = DFlashDraftModel.project_sharded_target_hidden(
                    worker, local, group=group, num_tokens=live
                )
                self.assertEqual(call_rows, [rows // tp if live == rows else live])
                torch.testing.assert_close(output, all_projected[:live])
        self.assertEqual(group.all_gather_into_tensor.call_count, 3)
        with self.assertRaises(ValueError):
            DFlashDraftModel.project_sharded_target_hidden(
                worker, local, group=group, num_tokens=rows + 1
            )


if __name__ == "__main__":
    unittest.main()
