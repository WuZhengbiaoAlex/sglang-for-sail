import unittest
from unittest.mock import Mock

import torch

from sglang.srt.layers.vocab_parallel_top1 import vocab_parallel_top1
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestVocabParallelTop1(CustomTestCase):
    def test_rank_order_ties_padding_and_trailing_empty_shard(self):
        for dtype in (torch.bfloat16, torch.float32):
            with self.subTest(dtype=dtype):
                # 16 physical columns over four ranks. Physical columns 10..15
                # are padding; rank 2 is partial and rank 3 is entirely empty.
                logits = torch.randn(9, 16, generator=torch.Generator().manual_seed(41))
                logits[0, :10] = float("-inf")
                logits[1, :10] = -0.0
                logits[1, 5] = 0.0
                logits[2, [2, 6, 9]] = float("inf")
                logits[3, [3, 7, 9]] = float("nan")
                logits[4, [8, 9]] = 30.0
                logits[5, 9] = 32.0
                logits[:, 10:] = float("nan")
                logits = logits.to(dtype)
                reference = logits[:, :10].float().argmax(-1)
                starts = (0, 4, 8, 10)
                widths = (4, 4, 2, 0)
                values = []
                ids = []
                for rank, (start, valid) in enumerate(zip(starts, widths)):
                    if valid:
                        shard = logits[:, rank * 4 : rank * 4 + valid]
                        index = shard.float().argmax(-1)
                        values.append(shard.gather(1, index[:, None]).flatten())
                        ids.append(index + start)
                    else:
                        values.append(logits.new_full((len(logits),), float("-inf")))
                        ids.append(torch.full((len(logits),), start, dtype=torch.long))

                for rank, (start, valid) in enumerate(zip(starts, widths)):
                    def gather(output, local):
                        # Simulate the other shards using the independent argmax
                        # reference and verify this rank's actual contribution.
                        parts = values if local.is_floating_point() else ids
                        torch.testing.assert_close(local, parts[rank], equal_nan=True)
                        output.copy_(torch.cat(parts))

                    group = Mock(world_size=4)
                    group.all_gather_into_tensor.side_effect = gather
                    got = vocab_parallel_top1(
                        logits[:, rank * 4 : rank * 4 + 4],
                        vocab_start=start,
                        valid_columns=valid,
                        group=group,
                    )
                    torch.testing.assert_close(got, reference)
                    self.assertEqual(group.all_gather_into_tensor.call_count, 2)

    def test_one_rank_and_empty_batch(self):
        group = Mock(world_size=1)
        logits = torch.tensor([[0.0, 1.0, 1.0, float("nan")]])
        got = vocab_parallel_top1(logits, vocab_start=70000, valid_columns=3, group=group)
        self.assertEqual(got.tolist(), [70001])
        empty = vocab_parallel_top1(logits[:0], vocab_start=0, valid_columns=3, group=group)
        self.assertEqual(empty.shape, (0,))
        self.assertEqual(empty.dtype, torch.long)
        group.all_gather_into_tensor.assert_not_called()


if __name__ == "__main__":
    unittest.main()
