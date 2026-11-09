"""CPU coverage for the BF16 DFlash window adapter; no native kernel execution."""

import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.layers.attention.dflash_flashmla_backend import (
    DFlashFlashMLABackend,
    build_dflash_causal_window_indices,
)
from sglang.srt.layers.radix_attention import AttentionType
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestDFlashFlashMLAIndices(CustomTestCase):
    @staticmethod
    def make_inputs(prefix_lengths, block_size=5):
        prefix = []
        blocks = []
        indptr = [0]
        # Different requests have disjoint, noncontiguous physical addresses.
        for request, length in enumerate(prefix_lengths):
            base = 1_000_000 * (request + 1)
            prefix.extend(base + 13 * i for i in range(length))
            blocks.extend(base + 300_000 + 17 * j for j in range(block_size))
            indptr.append(len(prefix))
        return (
            torch.tensor(indptr, dtype=torch.int32),
            torch.tensor(prefix, dtype=torch.int64),
            torch.tensor(blocks, dtype=torch.int64),
        )

    def check_reference(self, inputs, *, window_left=1023, block_size=5, rows=None):
        indptr, prefix, current = inputs
        bs = len(indptr) - 1
        if rows is None:
            rows = bs * block_size
        indices, counts = build_dflash_causal_window_indices(
            *inputs,
            block_size=block_size,
            window_left=window_left,
            num_query_rows=rows,
        )
        width = ((window_left + 1 + 127) // 128) * 128
        self.assertEqual(indices.shape, (rows, 1, width))
        self.assertEqual(indices.dtype, torch.int32)
        self.assertEqual(counts.dtype, torch.int32)
        # Independent reference: concatenate complete sequence slots, take the
        # causal prefix of that list, then keep its most recent W+1 elements.
        for row in range(rows):
            request, position = divmod(row, block_size)
            if request >= bs or row >= current.numel():
                visible = []
            else:
                committed = prefix[indptr[request] : indptr[request + 1]].tolist()
                proposal = current[
                    request * block_size : request * block_size + position + 1
                ].tolist()
                visible = (committed + proposal)[-(window_left + 1) :]
            self.assertEqual(counts[row].item(), len(visible), f"row {row}")
            self.assertEqual(indices[row, 0, : len(visible)].tolist(), visible)
            self.assertTrue(torch.all(indices[row, 0, len(visible) :] == -1))
        return indices, counts

    def test_empty_batch_and_zero_query_rows(self):
        self.check_reference(self.make_inputs([]), rows=0)
        self.check_reference(self.make_inputs([]), rows=7)
        self.check_reference(self.make_inputs([3]), rows=0)

    def test_zero_prefix_reads_own_causal_block(self):
        indices, counts = self.check_reference(self.make_inputs([0]))
        self.assertEqual(counts.tolist(), [1, 2, 3, 4, 5])
        self.assertNotEqual(indices[4, 0, 4].item(), -1)

    def test_short_prefix_and_sliding_window_boundaries(self):
        self.check_reference(
            self.make_inputs([0, 1, 7, 31, 1019, 1020, 1021, 1022, 1023, 1024, 1025])
        )

    def test_full_32_request_block_and_long_noncontiguous_prefixes(self):
        indices, counts = self.check_reference(
            self.make_inputs([4096 + request * 3 for request in range(32)])
        )
        self.assertEqual(indices.shape, (160, 1, 1024))
        self.assertTrue(torch.all(counts == 1024))

    def test_topk_alignment_and_self_only_window(self):
        self.check_reference(self.make_inputs([0, 31]), window_left=6)
        self.check_reference(self.make_inputs([0, 31]), window_left=0)

    def test_trailing_query_padding_never_reads_next_request(self):
        self.check_reference(self.make_inputs([7, 2048]), rows=13)
        indptr, prefix, current = self.make_inputs([7, 2048])
        self.check_reference((indptr, prefix, current[:8]), rows=10)

    def test_empty_current_slots_produce_only_padding(self):
        indptr, prefix, _ = self.make_inputs([7])
        self.check_reference(
            (indptr, prefix, torch.empty(0, dtype=torch.int64)), rows=5
        )

    def test_same_shape_inputs_are_read_again_after_inplace_update(self):
        # CPU analogue of graph-replay input mutation: the helper must derive
        # indices from tensor values, never memoize them by shape/data pointer.
        inputs = self.make_inputs([7, 31])
        before, _ = self.check_reference(inputs)
        ptrs = [tensor.data_ptr() for tensor in inputs]
        inputs[0][1] = 9
        inputs[1].add_(21)
        inputs[2].add_(39)
        after, _ = self.check_reference(inputs)
        self.assertEqual(ptrs, [tensor.data_ptr() for tensor in inputs])
        self.assertFalse(torch.equal(before, after))

    def test_invalid_static_dimensions_fail(self):
        inputs = self.make_inputs([1])
        for block_size, window_left, rows in ((0, 1023, 5), (5, -1, 5), (5, 1, -1)):
            with self.subTest(
                block_size=block_size, window_left=window_left, rows=rows
            ):
                with self.assertRaises(ValueError):
                    build_dflash_causal_window_indices(
                        *inputs,
                        block_size=block_size,
                        window_left=window_left,
                        num_query_rows=rows,
                    )


class TestDFlashFlashMLAForward(CustomTestCase):
    def test_empty_forward_does_not_require_metadata_or_touch_native(self):
        backend = DFlashFlashMLABackend.__new__(DFlashFlashMLABackend)
        layer = SimpleNamespace(tp_q_head_num=12, qk_head_dim=576, v_head_dim=512)
        batch = SimpleNamespace(
            batch_size=0,
            forward_mode=SimpleNamespace(is_target_verify=lambda: True),
            spec_algorithm=SimpleNamespace(is_dflash=lambda: True),
        )
        for rows in (0, 5):
            q = torch.full((rows, 12, 576), float("nan"), dtype=torch.bfloat16)
            output = backend.forward_extend(q, None, None, layer, batch)
            self.assertEqual(output.shape, (rows, 12 * 512))
            self.assertTrue(torch.all(output == 0))

    def test_split_rope_kv_write_head_padding_and_allmasked_output(self):
        backend = DFlashFlashMLABackend.__new__(DFlashFlashMLABackend)
        backend.target_verify_num_tokens_per_req = 5
        backend.forward_metadata = SimpleNamespace(
            kv_indptr=torch.tensor([0, 0], dtype=torch.int32),
            kv_indices=torch.empty(0, dtype=torch.int64),
            custom_mask=None,
            swa_out_cache_loc=None,
            out_cache_loc_full_physical=None,
        )
        pool = torch.zeros((20, 1, 576), dtype=torch.bfloat16)
        backend.token_to_kv_pool = SimpleNamespace(get_key_buffer=lambda _: pool)
        writes = []

        def write_cache(batch, layer, loc_info, k, v):
            writes.append(loc_info)
            pool[loc_info.loc] = k

        backend._set_kv_buffer = write_cache
        layer = SimpleNamespace(
            layer_id=0,
            tp_q_head_num=12,
            qk_head_dim=576,
            v_head_dim=512,
            attn_type=AttentionType.DECODER,
            is_cross_attention=False,
            sliding_window_size=1023,
            logit_cap=0,
            xai_temperature_len=None,
            k_scale=None,
            v_scale=None,
            scaling=0.125,
        )
        batch = SimpleNamespace(
            batch_size=1,
            forward_mode=SimpleNamespace(is_target_verify=lambda: True),
            spec_algorithm=SimpleNamespace(is_dflash=lambda: True),
            spec_info=SimpleNamespace(num_tokens_per_req=5),
            out_cache_loc=torch.tensor([3, 6, 9, 12, 15]),
            _attn_output=torch.empty((7, 12 * 512), dtype=torch.bfloat16),
        )
        q = torch.ones((7, 12, 512), dtype=torch.bfloat16)
        q_rope = torch.full((7, 12, 64), 2.0, dtype=torch.bfloat16)
        k = torch.full((5, 1, 512), 3.0, dtype=torch.bfloat16)
        k_rope = torch.full((5, 1, 64), 4.0, dtype=torch.bfloat16)
        captured = {}

        def native(**kwargs):
            captured.update(kwargs)
            # Deliberately emulate an undefined all-masked softmax for padding.
            out = torch.full((7, 64, 512), float("nan"), dtype=torch.bfloat16)
            out[:5] = 6
            return out, None, None

        module_path = "sglang.srt.hardware_backend.ppu.attention.flash_mla"
        module = ModuleType(module_path)
        module.flash_mla_sparse_fwd = Mock(side_effect=native)
        with patch.dict("sys.modules", {module_path: module}):
            output = backend.forward_extend(
                q, k, k, layer, batch, q_rope=q_rope, k_rope=k_rope
            )

        self.assertIs(output, batch._attn_output)
        self.assertEqual(len(writes), 1)
        self.assertTrue(torch.all(pool[batch.out_cache_loc, :, :512] == 3))
        self.assertTrue(torch.all(pool[batch.out_cache_loc, :, 512:] == 4))
        self.assertTrue(torch.all(pool[0] == 0))
        self.assertEqual(captured["q"].shape, (7, 64, 576))
        self.assertTrue(torch.all(captured["q"][:, :12, :512] == 1))
        self.assertTrue(torch.all(captured["q"][:, :12, 512:] == 2))
        self.assertTrue(torch.all(captured["q"][:, 12:] == 0))
        self.assertEqual(captured["topk_length"].tolist(), [1, 2, 3, 4, 5, 0, 0])
        self.assertEqual(captured["indices"][4, 0, :5].tolist(), [3, 6, 9, 12, 15])
        self.assertTrue(torch.all(output[:5] == 6))
        self.assertTrue(torch.all(output[5:] == 0))
        module.flash_mla_sparse_fwd.assert_called_once()


if __name__ == "__main__":
    unittest.main()
