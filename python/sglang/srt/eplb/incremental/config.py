"""Selection and startup constraints for incremental decode EPLB."""

# Planner policy. Edit these constants together on every EP rank; no CLI/env flags.
# Ratios are relative to the mean token load per EP rank, not per expert.
CPU_PLANNER_THREADS = 4
SPLIT_TARGET = 0.6
MERGE_TARGET = 0.4
# Per layer, per target decode step, across the entire EP group. Reusing a
# resident copy costs no transfer budget. This is not a per-rank limit.
MAX_COPIES_PER_LAYER_PER_STEP = 2
# Repair-phase owner relocations; a two-way exchange costs two. Split/merge
# membership changes are separate, but share the same copy budget above.
MAX_OWNER_MOVES_PER_LAYER_PER_STEP = 32
# Ignore repair gains smaller than mean_rank_load**2 / this divisor.
# Larger values admit smaller improvements (and potentially more churn).
EXCHANGE_GAIN_DIVISOR = 1000


def is_enabled(config=None) -> bool:
    if config is None:
        from sglang.srt.runtime_context import get_exec

        config = get_exec().moe
    return config.enable_eplb and config.eplb_algorithm == "incremental"


def validate_options(config) -> None:
    if not config.enable_eplb:
        raise ValueError("--eplb-algorithm incremental requires --enable-eplb")
    if getattr(config, "enable_eplb_async", False):
        raise ValueError("incremental EPLB owns its asynchronous placement lifecycle")
    if config.ep_dispatch_algorithm not in (None, "static"):
        raise ValueError("incremental EPLB requires --ep-dispatch-algorithm static")
    if config.expert_distribution_recorder_mode is not None:
        raise ValueError(
            "incremental EPLB uses decode-only receive counts; disable the distribution recorder"
        )
    if getattr(config, "init_expert_location", "trivial") != "trivial":
        raise ValueError("incremental EPLB initializes its own safe slot placement")
    if config.ep_size <= 1:
        raise ValueError("incremental EPLB requires EP size greater than one")
    if config.ep_num_redundant_experts != 2 * config.ep_size:
        raise ValueError(
            "incremental EPLB requires exactly two spare slots per EP rank"
        )
    if config.moe_a2a_backend not in ("deepep", "megamoe"):
        raise ValueError(
            "incremental EPLB requires --moe-a2a-backend deepep or megamoe"
        )
    if config.moe_a2a_backend == "deepep" and config.deepep_mode not in (
        "auto",
        "low_latency",
    ):
        raise ValueError("incremental EPLB requires --deepep-mode auto or low_latency")
    if not config.disable_shared_experts_fusion:
        raise ValueError("incremental EPLB requires --disable-shared-experts-fusion")
    if (
        config.disable_cuda_graph
        or config.cuda_graph_config.decode.backend == "disabled"
    ):
        raise ValueError("incremental EPLB requires decode CUDA graphs")
    if config.num_continuous_decode_steps != 1:
        raise ValueError("incremental EPLB requires --num-continuous-decode-steps 1")
    if config.disaggregation_mode == "prefill":
        raise ValueError("incremental EPLB cannot run on a prefill-only server")
    if config.enable_lora:
        raise ValueError("incremental EPLB does not support LoRA expert replicas")
    if config.speculative_algorithm not in (None, "DSPARK", "DFLASH"):
        raise ValueError(
            "incremental EPLB supports DSPARK, DFLASH, or non-speculative decode"
        )
    if getattr(config, "enable_two_batch_overlap", False) or getattr(
        config, "enable_single_batch_overlap", False
    ):
        raise ValueError(
            "incremental EPLB requires one target graph per placement boundary; "
            "two/single-batch overlap is not supported"
        )
    if config.enable_pdmux:
        raise ValueError(
            "incremental EPLB requires a common prefill/decode placement boundary"
        )
    if config.elastic_ep_backend is not None:
        raise ValueError("incremental EPLB does not support elastic EP")


def is_routed_expert_module(module) -> bool:
    """Shared storage must be excluded from allocation and weight discovery."""
    return (
        not bool(getattr(module, "is_shared_fused_moe", False))
        and bool(getattr(module, "uses_expert_location", True))
        and not bool(getattr(module, "is_draft", False))
    )
