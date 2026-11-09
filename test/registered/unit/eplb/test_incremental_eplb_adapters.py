"""Incremental selection must preserve the existing windowed/async EPLB paths."""

import ast
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Optional, Union
from unittest.mock import Mock

import pytest
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

ROOT = Path(__file__).resolve().parents[4] / "python/sglang/srt"


def load_function(path, name, namespace, cls=None):
    tree = ast.parse((ROOT / path).read_text())
    nodes = tree.body
    if cls:
        nodes = next(n for n in nodes if getattr(n, "name", None) == cls).body
    node = next(n for n in nodes if getattr(n, "name", None) == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), path, "exec"), namespace)
    return namespace[name]


def options(**overrides):
    cfg = dict(
        enable_eplb=True,
        eplb_algorithm="incremental",
        enable_eplb_async=False,
        ep_dispatch_algorithm=None,
        expert_distribution_recorder_mode=None,
        init_expert_location="trivial",
        ep_size=32,
        ep_num_redundant_experts=64,
        moe_a2a_backend="deepep",
        deepep_mode="auto",
        disable_shared_experts_fusion=True,
        disable_cuda_graph=False,
        cuda_graph_config=SimpleNamespace(decode=SimpleNamespace(backend="full")),
        num_continuous_decode_steps=1,
        disaggregation_mode="null",
        enable_lora=False,
        speculative_algorithm="DSPARK",
        enable_two_batch_overlap=False,
        enable_single_batch_overlap=False,
        enable_pdmux=False,
        elastic_ep_backend=None,
        ep_join_mode=None,
    )
    cfg.update(overrides)
    return SimpleNamespace(**cfg)


@pytest.mark.parametrize(
    "enabled,algorithm", [(False, "auto"), (True, "auto"), (True, "incremental")]
)
def test_algorithm_selection_preserves_recorder_policy(enabled, algorithm):
    cfg = options(enable_eplb=enabled, eplb_algorithm=algorithm)
    resolutions = []

    def resolve(server_args, source, **fields):
        resolutions.append(fields)
        for name, value in fields.items():
            setattr(server_args, name, value)

    fn = load_function(
        "arg_groups/parallel_hook.py",
        "handle_eplb_and_dispatch",
        dict(
            Any=object,
            resolving_view=lambda x: x,
            resolved_view=lambda x: x,
            declare_resolution=resolve,
            logger=Mock(),
        ),
    )
    fn(cfg)
    if not enabled:
        assert resolutions == []
    else:
        assert cfg.ep_dispatch_algorithm == "static"
        assert cfg.expert_distribution_recorder_mode == (
            None if algorithm == "incremental" else "stat"
        )


def test_incremental_rejects_competing_async_runtime_before_resolution():
    resolve = Mock()
    fn = load_function(
        "arg_groups/parallel_hook.py",
        "handle_eplb_and_dispatch",
        dict(Any=object, resolving_view=lambda x: x, declare_resolution=resolve),
    )
    with pytest.raises(ValueError, match="asynchronous placement lifecycle"):
        fn(options(enable_eplb_async=True))
    resolve.assert_not_called()


@pytest.mark.parametrize(
    "incremental,draft,expected",
    [
        (True, False, False),
        (False, False, True),
        (True, True, False),
        (False, True, False),
    ],
)
def test_only_windowed_algorithm_constructs_existing_manager(
    incremental, draft, expected
):
    manager = Mock()
    fn = load_function(
        "model_executor/model_runner.py",
        "maybe_init_eplb_manager",
        dict(
            EPLBManager=manager,
            get_exec=lambda: SimpleNamespace(moe=SimpleNamespace(enable_eplb=True)),
            is_incremental_eplb_enabled=lambda: incremental,
        ),
        "ModelRunner",
    )
    runner = SimpleNamespace(is_draft_worker=draft, model_config=object())
    fn(runner)
    assert manager.called is expected
    assert runner.eplb_manager is (manager.return_value if expected else None)


def test_incremental_does_not_warm_up_windowed_cpp_runtime():
    warmup = Mock(side_effect=AssertionError("windowed runtime must remain inactive"))
    fn = load_function(
        "model_executor/model_runner.py",
        "_warmup_eplb_cpp_extensions",
        dict(
            is_incremental_eplb_enabled=lambda: True,
            warmup_eplb_expert_location_cpp=warmup,
            warmup_eplb_deepseek_cpp=warmup,
            warmup_eplb_async_runtime_cpp=warmup,
        ),
        "ModelRunner",
    )
    fn(SimpleNamespace(server_args=options(), is_draft_worker=False))
    warmup.assert_not_called()


@pytest.mark.parametrize("algorithm", [None, "DSPARK", "DFLASH"])
def test_fixed_width_speculative_options(algorithm):
    validate = load_function("eplb/incremental/config.py", "validate_options", {})
    validate(options(speculative_algorithm=algorithm))


@pytest.mark.parametrize("algorithm", ["EAGLE", "EAGLE3", "NEXTN", "DFLASH_SELECTOR"])
def test_unvalidated_speculative_options_rejected(algorithm):
    validate = load_function("eplb/incremental/config.py", "validate_options", {})
    with pytest.raises(ValueError, match="speculative decode"):
        validate(options(speculative_algorithm=algorithm))


@pytest.mark.parametrize("mixed", [False, True])
def test_dflash_mixed_prefill_verify_stays_eager(mixed):
    guard = load_function(
        "speculative/dflash_worker_v2.py",
        "_apply_target_verify_graph_guards",
        {
            "ForwardBatch": object,
            "_is_npu": False,
            "get_parallel": lambda: SimpleNamespace(enable_dp_attention=True),
        },
        "DFlashWorkerV2",
    )
    batch = SimpleNamespace(
        can_run_decode_cuda_graph=True,
        original_global_num_tokens_cpu=[160, 160, 0, 160],
    )
    guard(batch, is_extend_in_batch=mixed)
    assert batch.can_run_decode_cuda_graph is (not mixed)


@pytest.mark.parametrize("target,raises", [(True, False), (True, True), (False, False)])
def test_target_graph_scope_does_not_leak_to_draft(target, raises):
    active = []

    @contextmanager
    def activate():
        active.append(True)
        try:
            yield
        finally:
            active.pop()

    def replay(*args):
        assert bool(active) is target
        if raises:
            raise RuntimeError("graph failure")
        return "result"

    execute = load_function(
        "model_executor/runner/decode_cuda_graph_runner.py",
        "execute",
        dict(
            ForwardBatch=object,
            Optional=Optional,
            Union=Union,
            PPProxyTensors=object,
            LogitsProcessorOutput=object,
        ),
        "DecodeCudaGraphRunner",
    )
    runner = SimpleNamespace(
        _incremental_eplb_decode_bindings=(
            SimpleNamespace(activate=activate) if target else None
        ),
        _execute=replay,
    )
    if raises:
        with pytest.raises(RuntimeError, match="graph failure"):
            execute(runner, object())
    else:
        assert execute(runner, object()) == "result"
    assert active == []
