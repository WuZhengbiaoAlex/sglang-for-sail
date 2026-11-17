"""Single-PPU production-dispatch checks for FP32/BF16 TopK, with optional timing."""

import argparse
import gc
import json
import os
import statistics
from pathlib import Path
from unittest.mock import patch

import torch

from sglang.kernels.ops.attention.dsv4 import topk as tk

TOPK = 2048


def ordered_keys(scores):
    raw = scores.view(torch.int16).int() & 65535
    keys = raw ^ torch.where((raw & 32768) != 0, 65535, 32768)
    return torch.where((raw & 32767) > 32640, 65535, keys)


def check_selection(scores, lengths, output):
    """Check exact unordered TopK, including the native NaN/zero key policy."""
    indices = output.long()
    valid = indices >= 0
    assert bool(
        ((indices == -1) | (valid & (indices < lengths[:, None]))).all()
    ), "bounds"
    assert torch.equal(valid.sum(1), lengths.clamp(max=output.shape[1]).long()), "count"
    sorted_indices = indices.sort(1).values
    assert bool(
        (
            (sorted_indices[:, 1:] != sorted_indices[:, :-1])
            | (sorted_indices[:, 1:] == -1)
        ).all()
    ), "duplicate"
    keys = ordered_keys(scores).float() if scores.dtype == torch.bfloat16 else scores
    padding = (
        torch.arange(scores.shape[1], device=scores.device)[None, :]
        >= lengths[:, None]
    )
    expected = keys.masked_fill(padding, -torch.inf).topk(output.shape[1], dim=1).values
    expected = expected.sort(1).values
    actual = keys.gather(1, indices.clamp_min(0)).masked_fill(~valid, -torch.inf)
    assert torch.equal(actual.sort(1).values, expected), "ordered-key multiset"


def make_op(scores, lengths, split, pages=None, page_size=64, topk=TOPK):
    """Exercise the production BF16 dispatcher, including scratch allocation."""
    output = torch.empty(
        (scores.shape[0], topk), device=scores.device, dtype=torch.int32
    )

    def run():
        flags = {"SGLANG_PPU_DSA_SPLIT_TOPK_BF16": str(int(split))}
        with patch.dict(os.environ, flags):
            tk.topk_transform_paged_v2_bf16(scores, lengths, pages, output, page_size)
        return output

    return run


def check_edges(capacity=262144):
    scores = torch.randn((20, capacity), device="cuda", dtype=torch.bfloat16)
    initial = [
        0, 1, 2047, 2048, 2049, 4095, 4096, 4097, 8191, 8192,
        8193, 16384, 16385, 49152, 49153, 147455, 147456, 147457,
        262143, 262144,
    ]
    initial = [min(x, capacity) for x in initial]
    lengths = torch.tensor(initial, device="cuda", dtype=torch.int32)
    original = lengths.clone()
    ops = {name: make_op(scores, lengths, split) for name, split in (
        ("native", False), ("split", True)
    )}
    graphs = {}
    for name, op in ops.items():
        check_selection(scores, lengths, op())
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = op()
        graphs[name] = graph, output

    states = [
        "flip", "empty", "ones", "near", "positive_negative_inf",
        "signed_zero", "nan_mixed", "nan_all",
    ]
    for state in states:
        lengths.copy_(original.flip(0) if state == "flip" else original)
        if state == "flip":
            scores.mul_(-0.875)
        elif state == "empty":
            lengths.zero_()
            scores.normal_()
        elif state == "ones":
            scores.fill_(1)
        elif state == "near":
            scores.fill_(1)
            scores[:, ::32] = 1.0078125
        elif state == "positive_negative_inf":
            scores.fill_(-torch.inf)
            scores[:, ::128] = torch.inf
        elif state == "signed_zero":
            scores.zero_()
            scores.view(torch.int16)[:, 1::2] = -32768
        elif state == "nan_mixed":
            scores.normal_()
            scores.view(torch.int16)[:, :5000] = 32705
            scores.view(torch.int16)[:, 5000:10000] = -63
        else:
            scores.view(torch.int16).fill_(32705)
        for graph, output in graphs.values():
            graph.replay()
            torch.cuda.synchronize()
            check_selection(scores, lengths, output)
        print("PASS", state, flush=True)
    return {
        "implementations": list(ops),
        "states": ["initial", *states],
        "initial_lengths": initial,
        "check_count": len(ops) * (1 + len(states)),
    }


def check_dispatch():
    # Guard boundaries, generalized row counts, smaller K and output fallbacks.
    cases = [
        (1, 16384, 2048, False, 64),
        (1, 16392, 2048, False, 64),
        (20, 24576, 2048, False, 64),
        (1, 30712, 2048, False, 64),
        (1, 30720, 2048, False, 64),
        (256, 30720, 2048, False, 64),
        (256, 32768, 2048, False, 64),
        (257, 32768, 2048, False, 64),
        (5, 98304, 2048, False, 64),
        (21, 98304, 2048, False, 64),
        (20, 32760, 2048, False, 64),
        (20, 32768, 2048, False, 64),
        (20, 32776, 2048, False, 64),
        (20, 147456, 2048, False, 64),
        (20, 147464, 2048, False, 64),
        (20, 262152, 2048, False, 64),
        (1, 131928, 2048, False, 64),
        (5, 131928, 2048, False, 64),
        (19, 131928, 2048, False, 64),
        (21, 131928, 2048, False, 64),
        (160, 32768, 2048, False, 64),
        (20, 65536, 1024, False, 64),
        (20, 65536, 2048, True, 64),
        (20, 65536, 2048, False, 32),
    ]
    for rows, width, k, physical, page_size in cases:
        scores = torch.randn((rows, width), device="cuda", dtype=torch.bfloat16)
        lengths = torch.arange(rows, device="cuda", dtype=torch.int32)
        lengths = (width - lengths * 7919).clamp_min_(0)
        lengths[-1] = 0
        pages = None
        if physical:
            pages = torch.arange(
                (width + page_size - 1) // page_size, device="cuda", dtype=torch.int32
            ).repeat(rows, 1) + 7
        for split in (False, True):
            output = make_op(scores, lengths, split, pages, page_size, k)()
            if physical:
                output = torch.where(output >= 0, output - 7 * page_size, -1)
            check_selection(scores, lengths, output)
        print("PASS dispatch", rows, width, k, physical, page_size, flush=True)
    return len(cases) * 2


def check_fp32():
    dtype = torch.float32
    cases = [
        (8192, 20), (32768, 20), (131928, 20), (32768, 160),
        (65536, 1), (65536, 5), (65536, 24), (65536, 25), (65536, 26),
        (81920, 28), (81920, 31), (81920, 32), (98304, 26), (98304, 31),
        (131928, 32), (131928, 33),
        (32768, 5), (65528, 5), (131976, 5),
        (131968, 20), (133120, 20), (133120, 5), (133120, 32),
        (147456, 20), (147460, 20), (262144, 20),
    ]
    for width, rows in cases:
        # Include a padded score stride, causal boundaries, empty/short rows,
        # all ties and infinities (init/local-token masks).
        stride = (width + 7) // 8 * 8
        scores = torch.randn((rows, stride), device="cuda", dtype=dtype)[:, :width]
        lengths = torch.tensor(
            ([0, 1, 2047, 2048, 2049, 8192, 8193, 16384, 16385, width] * rows)[:rows],
            dtype=torch.int32,
            device="cuda",
        ).clamp_max_(width)
        lengths[0] = width
        if rows > 1:
            lengths[-1] = 0
        if rows > 8:
            scores[8].zero_()
        if rows > 9:
            scores[9].fill_(-torch.inf)
            scores[9, :4096] = torch.inf
        output = torch.empty((rows, 2048), dtype=torch.int32, device="cuda")
        plan = tk.plan_topk_v2(lengths)

        def run():
            tk.topk_transform_paged_v2(scores, lengths, None, output, 64, plan)

        run()
        check_selection(scores, lengths, output)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()
        scores.mul_(0.5)
        lengths.div_(2, rounding_mode="floor")
        plan.copy_(tk.plan_topk_v2(lengths))
        graph.replay()
        check_selection(scores, lengths, output)
        pages = (
            torch.arange((stride + 63) // 64, dtype=torch.int32, device="cuda").repeat(
                rows, 1
            )
            + 7
        )
        tk.topk_transform_paged_v2(scores, lengths, pages, output, 64, plan)
        check_selection(scores, lengths, torch.where(output >= 0, output - 7 * 64, -1))
        print("PASS topk", rows, width, flush=True)
    return len(cases)


def time_ops(ops, samples, rounds):
    """Graph timing: 8 invocations/sample, rotated order in each round."""
    records = []
    names = list(ops)
    for repeat in range(rounds):
        graphs = {}
        for name, op in ops.items():
            for _ in range(3):
                op()
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for _ in range(8):
                    op()
            graph.replay()
            torch.cuda.synchronize()
            graphs[name] = graph
        timings = {name: [] for name in names}
        rotated = names[repeat % len(names) :] + names[: repeat % len(names)]
        for sample in range(samples):
            order = rotated if sample % 2 == 0 else list(reversed(rotated))
            for name in order:
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                graphs[name].replay()
                end.record()
                end.synchronize()
                timings[name].append(start.elapsed_time(end) * 1000 / 8)
        records.append(timings)
    return {
        "medians_us": {
            name: statistics.median(statistics.median(r[name]) for r in records)
            for name in names
        },
        "samples_us": records,
    }


def benchmark(samples, rounds):
    results = []
    for capacity, pattern in [
        (32768, "normal"), (32768, "ones"), (32768, "ordered"),
        (32768, "short24576"), (32768, "ordered_short24576"),
        (65536, "normal"), (131928, "normal"), (262144, "normal"),
        (262144, "ones"), (262144, "near"), (262144, "ordered"),
    ]:
        stride = (capacity + 511) // 512 * 512
        scores = torch.empty((20, stride), device="cuda", dtype=torch.bfloat16)
        scores = scores[:, :capacity]
        if pattern in ("normal", "short24576"):
            scores.normal_()
        elif pattern in ("ones", "near"):
            scores.fill_(1)
            if pattern == "near":
                scores[:, ::32] = 1.0078125
        else:
            scores.copy_(torch.linspace(-4, 4, capacity, device="cuda")[None, :])
        lengths = torch.full((20,), capacity, device="cuda", dtype=torch.int32)
        if pattern.endswith("short24576"):
            lengths.fill_(24576)
        ops = {"native": make_op(scores, lengths, False),
               "split": make_op(scores, lengths, True)}
        for op in ops.values():
            check_selection(scores, lengths, op())
        result = {"capacity": capacity, "pattern": pattern}
        result.update(time_ops(ops, samples, rounds))
        print("RESULT", json.dumps(result["medians_us"]), capacity, pattern, flush=True)
        results.append(result)
        del scores, lengths, ops
        gc.collect()
        torch.cuda.empty_cache()
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--samples", type=int, default=21)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    assert args.samples > 0 and args.rounds > 0
    torch.set_num_threads(2)
    torch.cuda.set_device(0)
    torch.manual_seed(20261112)
    fp32_cases = 0
    for split in ("0", "1"):
        with patch.dict(os.environ, {"SGLANG_PPU_DSA_SPLIT_TOPK": split}):
            fp32_cases += check_fp32()
    data = {"fp32_cases": fp32_cases, "edges": check_edges(),
            "direct_edges": check_edges(32768), "dispatch_cases": check_dispatch()}
    if args.benchmark:
        data["benchmark"] = benchmark(args.samples, args.rounds)
    data["complete"] = True
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(data, indent=2) + "\n")
    summary = {k: v for k, v in data.items() if k != "benchmark"}
    print("COMPLETE", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
