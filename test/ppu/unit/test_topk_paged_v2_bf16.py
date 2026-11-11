import pytest
import torch

from sglang.kernels.ops.attention.dsv4.topk import topk_transform_paged_v2_bf16
from sglang.srt.utils import is_ppu


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not is_ppu(), reason="requires a PPU"
)
PAGE_SIZE = 64


def check_result(scores, lengths, slots, page_table):
    valid = slots >= 0
    pages = slots.clamp_min(0).long() // PAGE_SIZE
    inverse = page_table.argsort(dim=1)
    raw = inverse.gather(1, pages) * PAGE_SIZE + slots % PAGE_SIZE
    raw = torch.where(valid, raw, slots)
    for row, length in enumerate(lengths):
        count = min(length, raw.shape[1])
        indices = raw[row, :count].long()
        assert (raw[row, count:] == -1).all()
        assert ((0 <= indices) & (indices < length)).all()
        assert indices.unique().numel() == count
        actual = scores[row, indices].float().sort(descending=True).values
        expected = scores[row, :length].float().topk(count).values
        assert ((actual == expected) | (actual.isnan() & expected.isnan())).all()


def make_scores(width, lengths):
    rng = torch.Generator().manual_seed(17)
    scores = torch.randn(len(lengths), width, generator=rng).bfloat16()
    # Put a larger value at the end of the threshold bucket. Its BF16 key shares
    # a 13-bit prefix with 1.0. Test both exactly-full and overflowing caches.
    for row, count in ((0, 8192), (1, 8193)):
        length = lengths[row]
        scores[row].zero_()
        scores[row, : count - 100] = 1
        scores[row, length - 100 : length] = 1.0546875
    # Raw bits exercise negative scores, both zeros, subnormals, infinities and NaNs.
    bits = torch.arange(65536).to(torch.int16).view(torch.bfloat16)
    scores[2, : bits.numel()] = bits
    return scores


@pytest.mark.parametrize("k", [1, 512, 1024, 2048])
def test_exact_bf16_paged_v2(k):
    width = 131072
    lengths = [35003, 131071, 65536, 16385, 16384, 4097, 17, 0]
    scores = make_scores(width, lengths)
    device_scores = scores.cuda()
    lens = torch.tensor(lengths, dtype=torch.int32, device="cuda")
    rng = torch.Generator().manual_seed(23)
    table = torch.stack(
        [
            torch.randperm(width // PAGE_SIZE, dtype=torch.int32, generator=rng)
            for _ in lengths
        ]
    )
    mapped = torch.empty((len(lengths), k), dtype=torch.int32, device="cuda")
    topk_transform_paged_v2_bf16(
        device_scores, lens, table.cuda(), mapped, PAGE_SIZE
    )
    check_result(scores, lengths, mapped.cpu(), table)


@pytest.mark.parametrize("width", [8192, 16384])
def test_bf16_paged_v2_register_only(width):
    k = 2048
    lengths = [width - 1]
    rng = torch.Generator().manual_seed(29)
    scores = torch.randn(1, width, generator=rng).bfloat16()
    device_scores = scores.cuda().as_strided((1, width), (1, 1))
    table = torch.arange(width // PAGE_SIZE, dtype=torch.int32).unsqueeze(0)
    lens = torch.tensor(lengths, dtype=torch.int32, device="cuda")
    mapped = torch.empty((1, k), dtype=torch.int32, device="cuda")
    topk_transform_paged_v2_bf16(device_scores, lens, table.cuda(), mapped, PAGE_SIZE)
    check_result(scores, lengths, mapped.cpu(), table)


def test_bf16_paged_v2_graph_switches_row_path():
    k, width = 2048, 1 << 20
    before = [4097, 35003, 0]
    after = [35003, 16383, 2047]
    rng = torch.Generator().manual_seed(19)
    scores = torch.randn(3, 35008, generator=rng).bfloat16()
    storage = torch.full((3, width), torch.inf, dtype=torch.bfloat16, device="cuda")
    storage[:, : scores.shape[1]].copy_(scores)
    lens = torch.tensor(before, dtype=torch.int32, device="cuda")
    table = torch.arange(width // PAGE_SIZE, dtype=torch.int32).repeat(3, 1)
    device_table = table.cuda()
    mapped = torch.empty((3, k), dtype=torch.int32, device="cuda")

    def run():
        topk_transform_paged_v2_bf16(storage, lens, device_table, mapped, PAGE_SIZE)

    run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for lengths in (before, after):
        lens.copy_(torch.tensor(lengths, dtype=torch.int32))
        graph.replay()
        check_result(scores, lengths, mapped.cpu(), table)


def test_bf16_paged_v2_requires_aligned_score_base():
    scores = torch.empty((2, 8192), dtype=torch.bfloat16, device="cuda")[:, 1:]
    lens = torch.tensor([4096, 4096], dtype=torch.int32, device="cuda")
    table = torch.arange(8192 // PAGE_SIZE, dtype=torch.int32, device="cuda").repeat(2, 1)
    out = torch.empty((2, 2048), dtype=torch.int32, device="cuda")
    with pytest.raises(RuntimeError, match="BF16 scores must be 16-byte aligned"):
        topk_transform_paged_v2_bf16(scores, lens, table, out, PAGE_SIZE)
