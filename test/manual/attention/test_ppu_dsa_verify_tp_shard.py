#!/usr/bin/env python3
"""Eight-PPU differential test of the production serial VERIFY shard method.

No model, server or performance claims. Run only on an authorized idle node:
  torchrun --standalone --nproc-per-node=8 -- test/manual/attention/test_ppu_dsa_verify_tp_shard.py \
      --run --output-dir /new/evidence/directory --include-128k

Small cases cover B16/32/64, per-rank physical page permutations, empty/padded
shards, and repeated graph replay. The optional final case uses B32/next5/128K.
Every rank retains the same logical K data under a different physical layout.
Only model construction and persistent-cache ownership are replaced by fixtures;
the production Indexer method, DeepGEMM, TopK and attention-TP collective run.
Logits and selected multisets are exact. Native TopK does not promise column
order (atomic slot allocation); order differences remain diagnostic evidence.
"""

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import socket
import time
import traceback

NEXT_N, HEADS, DIM, PAGE, TOPK, TP = 5, 32, 128, 64, 2048, 8
MEMORY_LIMIT = 1024**3


def write_new(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def execute(args, result):
    import torch
    import torch.distributed as dist
    import deep_gemm as dg

    from sglang.kernels.ops.attention.dsa.paged_mqa_logits import (
        deepgemm_paged_mqa_logits_native,
    )
    from sglang.kernels.ops.attention.dsv4.topk import (
        plan_topk_v2,
        topk_transform_paged_v2,
    )
    from sglang.srt.distributed import parallel_state as ps
    from sglang.srt.layers.attention.dsa.dsa_indexer import Indexer
    from sglang.srt.layers.attention.dsa.indexer_request_sharding import (
        IndexerRequestShardMetadata,
    )
    from sglang.srt.runtime_context import get_parallel, reset_context
    from sglang.srt.utils import is_ppu
    from sglang.test.kernels.paged_mqa import generate_paged_mqa_test_data
    from sglang.test.test_utils import publish_build_topology

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    assert int(os.environ["WORLD_SIZE"]) == TP, "Exactly eight PPU processes required"
    assert is_ppu() and torch.cuda.is_available(), "PPU runtime required"
    torch.cuda.set_device(local_rank)
    free, total = torch.cuda.mem_get_info()
    assert free >= MEMORY_LIMIT, "Need at least 1 GiB free per PPU"
    result.update(
        device=torch.cuda.get_device_name(),
        deep_gemm_version=importlib.metadata.version("deep-gemm"),
        free_bytes_before=free,
        total_device_bytes=total,
        phases=[],
    )
    assert result["deep_gemm_version"] == "1.1.0.dev21+v1.0.0.ppu2.2.1.gff182e2", (
        "This focused test contract freezes DeepGEMM dev21"
    )

    ps.set_custom_all_reduce(False)
    try:
        ps.init_distributed_environment(
            world_size=TP,
            rank=rank,
            local_rank=local_rank,
            distributed_init_method="env://",
            timeout=180,
        )
        publish_build_topology(tp_size=TP, ep_size=TP, world_rank=rank)
        ps.initialize_model_parallel()
        group = get_parallel().attn_tp_group
        assert group.world_size == TP and group.rank_in_group == rank
        result["group"] = {
            "size": group.world_size,
            "rank": group.rank_in_group,
            "name": group.unique_name,
        }
        memory_baseline = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()

        def same_across_ranks(value):
            values = [None] * TP
            dist.all_gather_object(values, value, group=group.cpu_group)
            assert all(v == values[0] for v in values), (
                "Logical fixture differs across ranks"
            )

        def tensor_sha(tensor):
            return hashlib.sha256(
                tensor.contiguous().view(torch.uint8).cpu().numpy().tobytes()
            ).hexdigest()

        def run_case(batch, max_len, graph_actuals):
            # One shared logical prefix keeps the 128K fixture small. Physical
            # permutations differ by rank, but each table maps to identical K.
            fixture = generate_paged_mqa_test_data(
                1, NEXT_N, HEADS, 2 * max_len, max_len
            )
            assert fixture["context_lens"].item() == max_len
            logical_kv = fixture["kv_fused"].contiguous()
            base_table = fixture["block_table"].expand(batch, -1).contiguous()
            num_pages = logical_kv.shape[0]
            physical_kv = torch.empty_like(logical_kv)
            table = torch.empty_like(base_table)
            expanded_table = torch.empty(
                (batch * NEXT_N, base_table.shape[1]), dtype=torch.int32, device="cuda"
            )
            ctx = torch.zeros((batch, NEXT_N), dtype=torch.int32, device="cuda")
            local_batch = batch // TP
            local_ctx = torch.zeros(
                (local_batch, NEXT_N), dtype=torch.int32, device="cuda"
            )
            torch.manual_seed(70421 + batch + max_len)
            q = torch.randn((batch * NEXT_N, HEADS, DIM), device="cuda").to(
                torch.float8_e4m3fn
            )
            weights = torch.rand((batch * NEXT_N, HEADS), device="cuda").add_(0.125)
            q_other = (-q.float()).to(q.dtype)
            weights_other = weights.flip(1).contiguous()
            same_across_ranks(
                {
                    "q": tensor_sha(q),
                    "weights": tensor_sha(weights),
                    "logical_kv": tensor_sha(logical_kv),
                }
            )
            del fixture

            def schedule(values):
                return dg.get_paged_mqa_logits_metadata(
                    values,
                    PAGE,
                    dg.get_num_sms(),
                    metadata_extra=(NEXT_N, HEADS, DIM, 1),
                )

            full_schedule = schedule(torch.full_like(ctx, max_len))
            local_schedule = schedule(torch.full_like(local_ctx, max_len))
            full_plan = plan_topk_v2(ctx.flatten())
            local_plan = plan_topk_v2(local_ctx.flatten())
            shard = IndexerRequestShardMetadata(
                batch, NEXT_N, rank, TP, local_ctx, local_schedule, local_plan
            )
            indexer = object.__new__(Indexer)
            indexer.index_topk = TOPK
            indexer.num_init_tokens = indexer.num_local_tokens = 0
            indexer.use_fp4 = indexer.bf16_indexer = indexer.use_int8 = False
            indexer.use_fp8 = True
            original_dispatch = indexer._get_paged_mqa_logits_dispatch
            observed_logits = []

            def observed_dispatch(*values, **kwargs):
                assert tuple(values[0].shape) == (local_batch, NEXT_N, HEADS, DIM)
                assert tuple(values[3].shape) == (local_batch, NEXT_N)
                out = original_dispatch(*values, **kwargs)
                observed_logits.append(out)
                return out

            indexer._get_paged_mqa_logits_dispatch = observed_dispatch

            def refresh(actual, revision):
                # A deterministic bijection maps canonical logical pages to
                # rank-local physical storage; permuting the table alone would
                # be an invalid fixture because it changes the logical data.
                gen = torch.Generator(device="cpu").manual_seed(
                    1907 + rank + 31 * revision
                )
                perm = torch.randperm(num_pages, generator=gen).to("cuda")
                physical_kv.view(torch.uint8).index_copy_(
                    0, perm, logical_kv.view(torch.uint8)
                )
                table.copy_(perm[base_table.long()].to(torch.int32))
                expanded_table.copy_(table.repeat_interleave(NEXT_N, dim=0))
                ctx.zero_()
                # Adjacent query lengths remain distinct. Several short rows
                # deliberately cross page boundaries or have fewer than K keys.
                starts = (
                    max_len
                    - 5
                    - (torch.arange(actual, device="cuda") % 8) * 65
                    - revision % 17
                )
                if max_len <= 4096 and actual >= 3:
                    starts[:3] = torch.tensor([59, 60, 61], device="cuda")
                if actual:
                    ctx[:actual].copy_(
                        (starts[:, None] + torch.arange(1, 6, device="cuda")).int()
                    )
                local_ctx.copy_(ctx[rank * local_batch : (rank + 1) * local_batch])
                for destination, values in (
                    (full_schedule, ctx),
                    (local_schedule, local_ctx),
                ):
                    fresh = schedule(values)
                    assert (
                        destination.shape[0] >= fresh.shape[0]
                        and destination.shape[1:] == fresh.shape[1:]
                    )
                    destination.zero_()
                    destination[: fresh.shape[0]].copy_(fresh)
                full_plan.copy_(plan_topk_v2(ctx.flatten()))
                local_plan.copy_(plan_topk_v2(local_ctx.flatten()))

            def production(q_input=q, weight_input=weights):
                return indexer._get_topk_paged_tp_sharded(
                    q_input,
                    physical_kv,
                    weight_input,
                    expanded_table,
                    shard,
                    max_seq_len=max_len,
                    page_size=PAGE,
                )

            def reference(q_input=q, weight_input=weights):
                logits = deepgemm_paged_mqa_logits_native(
                    dg.fp8_paged_mqa_logits,
                    q_input,
                    physical_kv,
                    weight_input,
                    ctx,
                    expanded_table,
                    full_schedule,
                    max_len,
                    q_offset=batch * NEXT_N,
                    B=batch,
                    next_n=NEXT_N,
                )
                physical = torch.empty(
                    (batch * NEXT_N, TOPK), dtype=torch.int32, device="cuda"
                )
                logical = torch.empty_like(physical)
                topk_transform_paged_v2(
                    logits,
                    ctx.flatten(),
                    expanded_table,
                    physical,
                    PAGE,
                    full_plan,
                    out_raw_indices=logical,
                )
                return logits, physical, logical

            def compare(label, actual, outputs, locals_, input_pairs):
                torch.cuda.synchronize()
                records = []
                for output, local_logits, pair in zip(outputs, locals_, input_pairs):
                    baseline_logits, baseline, logical = reference(*pair)
                    # Repeat the unchanged native dual-output path on the same
                    # logits. topk_impl.cuh allocates output slots with atomicAdd;
                    # raw-index and dual-output specializations need not order
                    # the same selected positions identically. Record this
                    # control without relaxing exact selection membership.
                    repeated = torch.empty_like(baseline)
                    repeated_logical = torch.empty_like(logical)
                    topk_transform_paged_v2(
                        baseline_logits,
                        ctx.flatten(),
                        expanded_table,
                        repeated,
                        PAGE,
                        full_plan,
                        out_raw_indices=repeated_logical,
                    )
                    torch.cuda.synchronize()
                    start = shard.query_start
                    mismatch = []
                    lengths = local_ctx.flatten().cpu().tolist()
                    for row, length in enumerate(lengths):
                        if length and not torch.equal(
                            local_logits[row, :length].view(torch.int32),
                            baseline_logits[start + row, :length].view(torch.int32),
                        ):
                            mismatch.append(row)
                    order_equal = torch.equal(output, baseline)
                    output_sorted = output.sort(dim=1)
                    baseline_sorted = baseline.sort(dim=1)
                    sets_equal = torch.equal(
                        output_sorted.values, baseline_sorted.values
                    )
                    baseline_repeat_order_equal = torch.equal(baseline, repeated)
                    baseline_repeat_sets_equal = torch.equal(
                        baseline_sorted.values, repeated.sort(dim=1).values
                    )
                    padded_safe = bool((output[actual * NEXT_N :] == -1).all())
                    # Align both selections by physical ID, preserving exact
                    # multiplicities and -1 padding. The baseline logical ID
                    # follows its own sorting permutation; sorting logical IDs
                    # independently would break this physical/logical pairing.
                    aligned_logical = logical.gather(1, baseline_sorted.indices)
                    valid = aligned_logical >= 0
                    p = aligned_logical.clamp_min(0)
                    canonical_page = base_table.repeat_interleave(NEXT_N, dim=0).gather(
                        1, (p // PAGE).long()
                    )
                    canonical_slot = canonical_page * PAGE + p % PAGE

                    # Fused storage is page-major: all 64*128 key bytes,
                    # followed by 64*4 scale bytes. It is not 132 bytes/token
                    # interleaved. Index narrow key/scale views before joining
                    # them, so this check never gathers whole 8448-byte pages.
                    def selected_bytes(pool, slots):
                        raw = pool.view(torch.uint8).reshape(
                            num_pages, PAGE * (DIM + 4)
                        )
                        pages, offsets = slots // PAGE, slots % PAGE
                        key = raw[:, : PAGE * DIM].reshape(num_pages, PAGE, DIM)[
                            pages, offsets
                        ]
                        scale = raw[:, PAGE * DIM :].reshape(num_pages, PAGE, 4)[
                            pages, offsets
                        ]
                        return torch.cat((key, scale), dim=-1)

                    physical_bounds = bool(
                        ((output >= -1) & (output < num_pages * PAGE)).all()
                    )
                    # Independently prove the gathered logical selection means
                    # the same ordered K/scale bytes on every receiver, despite
                    # different physical page permutations. Mask invalid bytes
                    # before hashing, rather than hashing arbitrary page zero.
                    ordered_selected = selected_bytes(
                        physical_kv, output.clamp(0, num_pages * PAGE - 1).long()
                    )
                    ordered_selected.masked_fill_((output < 0).unsqueeze(-1), 0)
                    ordered_sha = tensor_sha(ordered_selected)
                    del ordered_selected
                    all_rank_hashes = [None] * TP
                    dist.all_gather_object(
                        all_rank_hashes, ordered_sha, group=group.cpu_group
                    )
                    ordered_k_same_across_ranks = all(
                        digest == all_rank_hashes[0] for digest in all_rank_hashes
                    )
                    got = selected_bytes(
                        physical_kv,
                        output_sorted.values.clamp(0, num_pages * PAGE - 1).long(),
                    )
                    want = selected_bytes(logical_kv, canonical_slot.long())
                    selected_k_equal = physical_bounds and torch.equal(
                        got[valid], want[valid]
                    )
                    records.append(
                        {
                            "local_logits_bitwise_equal": not mismatch,
                            "logit_mismatch_local_rows": mismatch,
                            "physical_topk_order_equal": order_equal,
                            "physical_topk_sets_equal": sets_equal,
                            "baseline_repeat_order_equal": baseline_repeat_order_equal,
                            "baseline_repeat_sets_equal": baseline_repeat_sets_equal,
                            "physical_topk_within_pool": physical_bounds,
                            "selected_k_bytes_equal": selected_k_equal,
                            "selected_k_alignment": "physical_id_sorted_with_paired_baseline_logical_id",
                            "ordered_k_bytes_same_across_ranks": ordered_k_same_across_ranks,
                            "ordered_k_bytes_sha256": ordered_sha,
                            "all_rank_ordered_k_bytes_sha256": all_rank_hashes,
                            "padding_all_minus_one": padded_safe,
                        }
                    )
                    del (
                        baseline_logits,
                        baseline,
                        logical,
                        repeated,
                        repeated_logical,
                        got,
                        want,
                    )
                peak = torch.cuda.max_memory_allocated() - memory_baseline
                record = {
                    "phase": label,
                    "bucket": batch,
                    "actual_batch": actual,
                    "context_capacity": max_len,
                    "local_query_rows": local_batch * NEXT_N,
                    "local_active_requests": max(
                        0, min(local_batch, actual - rank * local_batch)
                    ),
                    "checks": records,
                    "peak_tensor_bytes_above_group_init": peak,
                }
                passed = (
                    all(
                        all(
                            r[k]
                            for k in (
                                "local_logits_bitwise_equal",
                                "physical_topk_sets_equal",
                                "baseline_repeat_sets_equal",
                                "physical_topk_within_pool",
                                "selected_k_bytes_equal",
                                "ordered_k_bytes_same_across_ranks",
                                "padding_all_minus_one",
                            )
                        )
                        for r in records
                    )
                    and peak < MEMORY_LIMIT
                )
                record["passed"] = passed
                result["phases"].append(record)
                write_new(
                    args.output_dir
                    / f"rank{rank:02d}_phase{len(result['phases']):02d}.json",
                    record,
                )
                # All ranks finish this phase and agree on failure before any
                # process enters the next collective (no rank-local early exit).
                statuses = [None] * TP
                dist.all_gather_object(statuses, passed, group=group.cpu_group)
                assert all(statuses), (
                    f"Differential failure in {label}: ranks {statuses}"
                )

            refresh(batch, 0)
            eager = production()
            eager_logits = observed_logits[-1]
            compare("eager", batch, [eager], [eager_logits], [(q, weights)])
            del eager, eager_logits
            observed_logits.clear()

            if graph_actuals:
                tracked = {
                    "q": q,
                    "weights": weights,
                    "q_other": q_other,
                    "weights_other": weights_other,
                    "kv": physical_kv,
                    "ctx": ctx,
                    "local_ctx": local_ctx,
                    "pages": expanded_table,
                    "full_schedule": full_schedule,
                    "local_schedule": local_schedule,
                    "full_plan": full_plan,
                    "local_plan": local_plan,
                }
                pointers = {name: tensor.data_ptr() for name, tensor in tracked.items()}
                # Exercise two distinct layer outputs in the small graph. A
                # shared mutable output scratch would clobber the first one.
                pairs = (
                    [(q, weights), (q_other, weights_other)]
                    if max_len <= 4096
                    else [(q, weights)]
                )
                with group.graph_capture() as capture:
                    for _ in range(3):
                        for pair in pairs:
                            production(*pair)
                    observed_logits.clear()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=capture.stream):
                        outputs = [production(*pair) for pair in pairs]
                captured_logits = list(observed_logits)
                output_pointers = [out.data_ptr() for out in outputs]
                assert len(set(output_pointers)) == len(output_pointers)
                for revision, actual in enumerate(graph_actuals, 1):
                    refresh(actual, revision)
                    # Same-shaped input updates must affect replay without
                    # replacing any captured pointer.
                    weights.mul_(0.99)
                    weights_other.mul_(0.98)
                    q.copy_((-q.float() * 0.75).to(q.dtype))
                    q_other.copy_((-q.float()).to(q.dtype))
                    graph.replay()
                    compare(
                        f"graph_replay_{revision}",
                        actual,
                        outputs,
                        captured_logits,
                        pairs,
                    )
                    assert pointers == {
                        name: tensor.data_ptr() for name, tensor in tracked.items()
                    }
                    assert output_pointers == [out.data_ptr() for out in outputs]
                del graph, outputs, captured_logits, tracked
            observed_logits.clear()
            gc.collect()

        run_case(16, 4096, [1, 0, 16])
        run_case(32, 4096, [32, 31, 29, 25, 9, 8, 7, 1, 0, 32])
        run_case(64, 4096, [])
        if args.include_128k:
            run_case(32, 131072, [29, 32])
        result["passed"] = all(p["passed"] for p in result["phases"])
        result["peak_tensor_bytes_above_group_init"] = (
            torch.cuda.max_memory_allocated() - memory_baseline
        )
        statuses = [None] * TP
        dist.all_gather_object(statuses, result["passed"], group=group.cpu_group)
        assert all(statuses)
    finally:
        # Only groups owned by this torchrun worker; no process killing,
        # container/service operations, driver reset or cache deletion.
        ps.destroy_model_parallel()
        ps.destroy_distributed_environment()
        reset_context()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--include-128k", action="store_true")
    args = parser.parse_args()
    if not args.run:
        parser.print_help()
        return 0
    if args.output_dir is None:
        parser.error("--run requires a fresh --output-dir")
    rank = int(os.environ["RANK"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / f"rank{rank:02d}_result.json"
    intent = args.output_dir / f"rank{rank:02d}_intent.json"
    result = {
        "schema": "ppu_dsa_verify_tp_shard_differential_v2",
        "passed": False,
        "rank": rank,
        "pid": os.getpid(),
        "hostname": socket.gethostname(),
        "started_wall": time.time(),
        "include_128k": args.include_128k,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "limits": [
            "Single attentionTP8 group; not a cross-DP model or accuracy test.",
            "Exact old/native vs sharded differential, not an absolute FP64 oracle.",
            "Native TopK column order is diagnostic; exact multisets and aligned K/scale bytes are required.",
            "No throughput or latency claim; no server or requests.",
        ],
    }
    write_new(intent, result)
    try:
        execute(args, result)
    except BaseException:
        result["error"] = traceback.format_exc()
        raise
    finally:
        result["completed_wall"] = time.time()
        write_new(output, result)


if __name__ == "__main__":
    main()
