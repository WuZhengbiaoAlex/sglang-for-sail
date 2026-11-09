"""Kimi EPLB routing and MegaMoE storage/counting contracts, without a server."""

import ast
from pathlib import Path
from types import SimpleNamespace
from typing import Optional
from unittest.mock import Mock

import pytest
import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")
ROOT = Path(__file__).resolve().parents[4] / "python/sglang/srt"


def definition(path, name, namespace, cls=None):
    nodes = ast.parse((ROOT / path).read_text()).body
    if cls:
        nodes = next(n for n in nodes if getattr(n, "name", None) == cls).body
    node = next(n for n in nodes if getattr(n, "name", None) == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), path, "exec"), namespace)
    return namespace[name]


@pytest.mark.parametrize("enabled", [False, True])
def test_kimi_routes_with_current_metadata_only_when_enabled(enabled):
    metadata = object()
    init = (
        Mock(return_value=metadata)
        if enabled
        else Mock(side_effect=AssertionError("OFF/draft read metadata"))
    )
    fn = definition(
        "models/kimi_k3.py",
        "_select_topk",
        dict(
            ExpertLocationDispatchInfo=SimpleNamespace(init_new=init),
        ),
        "KimiK3MoE",
    )
    module = SimpleNamespace(_uses_eplb=enabled, layer_idx=7, topk=Mock())
    hidden, logits, count = object(), object(), torch.tensor([3], dtype=torch.int32)
    fn(module, hidden, logits, count)
    if enabled:
        init.assert_called_once_with(layer_id=7)
        module.topk.assert_called_once_with(
            hidden,
            logits,
            num_token_non_padded=count,
            expert_location_dispatch_info=metadata,
        )
    else:
        init.assert_not_called()
        module.topk.assert_called_once_with(hidden, logits)


@pytest.mark.parametrize("enabled", [False, True])
def test_kimi_forward_passes_local_padding_bound_only_for_eplb(enabled):
    fn = definition(
        "models/kimi_k3.py",
        "forward",
        dict(
            torch=torch,
            Optional=Optional,
            ForwardBatch=object,
        ),
        "KimiK3MoE",
    )
    hidden = torch.zeros(8, 16)
    count = torch.tensor([5], dtype=torch.int32)
    getter = (
        Mock(return_value=count)
        if enabled
        else Mock(side_effect=AssertionError("OFF computed padding"))
    )
    module = SimpleNamespace(
        _dp_attention=False,
        _ep_a2a=True,
        _uses_eplb=enabled,
        _eligible_for_fused_front=False,
        _forward_unfused=Mock(return_value=hidden),
    )
    fn(module, hidden, forward_batch=SimpleNamespace(moe_num_token_non_padded=getter))
    assert module._forward_unfused.call_args.kwargs["num_token_non_padded"] is (
        count if enabled else None
    )


@pytest.mark.parametrize("wrapper", [False, True])
@pytest.mark.parametrize("k31", [False, True])
def test_kimi_metadata_uses_canonical_logical_count(wrapper, k31):
    ns = dict(ModelConfigForExpertLocation=SimpleNamespace)
    method = definition(
        "models/kimi_k3.py",
        "get_model_config_for_expert_location",
        ns,
        "KimiK3LinearForCausalLM",
    )
    text_cls = type(
        "KimiK3LinearForCausalLM", (), {"get_model_config_for_expert_location": method}
    )
    if k31:
        tree = ast.parse((ROOT / "models/kimi_k3.py").read_text())
        actual = next(
            n
            for n in tree.body
            if getattr(n, "name", None) == "KimiK31LinearForCausalLM"
        )
        assert [ast.unparse(base) for base in actual.bases] == [
            "KimiK3LinearForCausalLM"
        ]
        text_cls = type("KimiK31LinearForCausalLM", (text_cls,), {})
    config = SimpleNamespace(
        num_hidden_layers=48, num_experts=384, n_routed_experts=448, num_expert_group=1
    )
    cls = text_cls
    if wrapper:
        method = definition(
            "models/kimi_k3.py",
            "get_model_config_for_expert_location",
            {},
            "KimiK3ForConditionalGeneration",
        )
        cls = type(
            "KimiK3ForConditionalGeneration",
            (),
            {
                "get_model_config_for_expert_location": method,
                "language_model_class": text_cls,
            },
        )
        config = SimpleNamespace(text_config=config)
    metadata = cls.get_model_config_for_expert_location(config)
    assert (metadata.num_layers, metadata.num_logical_experts, metadata.num_groups) == (
        48,
        384,
        1,
    )


def test_topk_declares_real_layer_and_draft_role():
    # The runtime router indexes placement by TopK.layer_id. Missing it can
    # silently count every layer as zero before routing becomes split-aware.
    tree = ast.parse((ROOT / "models/kimi_k3.py").read_text())
    cls = next(n for n in tree.body if getattr(n, "name", None) == "KimiK3MoE")
    call = next(
        n
        for n in ast.walk(cls)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id == "TopK"
    )
    kwargs = {k.arg: ast.unparse(k.value) for k in call.keywords}
    assert kwargs["layer_id"] == "self.layer_idx"
    assert kwargs["is_nextn"] == "self.experts.is_draft"


@pytest.mark.parametrize("active", [False, True])
def test_megamoe_count_adapter_uses_decode_binding(active):
    fn = definition("layers/moe/utils.py", "get_expert_recv_stats_kwargs", {})
    counter = torch.zeros(14, dtype=torch.int32) if active else None
    callback = Mock(return_value=counter)
    experts = SimpleNamespace(layer_id=6, _incremental_eplb_counts=callback)
    result = fn(experts)
    if active:
        assert result["cumulative_local_expert_recv_stats"] is counter
    else:
        assert result == {}
    callback.assert_called_once_with(6)
    assert fn(SimpleNamespace()) == {}


def test_megamoe_aliases_rehome_execution_views_without_model_hook():
    owner = torch.nn.Parameter(torch.arange(24.0).reshape(4, 2, 3), requires_grad=False)
    view = owner.data
    module = SimpleNamespace(mega_l1_weights=(view,), mega_l2_weights=())
    fn = definition(
        "eplb/incremental/mover.py",
        "collect_weight_aliases",
        dict(
            _iter_moe_modules=lambda model: [(0, module)],
        ),
    )
    aliases = fn(SimpleNamespace(), [{"w13_weight": owner}])
    assert aliases[id(owner)][0] is view
    module.mega_l1_weights = (owner.clone(),)
    with pytest.raises(ValueError, match="cached expert weights"):
        fn(SimpleNamespace(), [{"w13_weight": owner}])


def test_megamoe_is_supported_independently_of_deepep_mode():
    from test_incremental_eplb_adapters import options

    fn = definition("eplb/incremental/config.py", "validate_options", {})
    fn(options(moe_a2a_backend="megamoe", deepep_mode="normal"))


@pytest.mark.parametrize("active", [False, True])
def test_ppu_megamoe_forwards_optional_physical_receive_counts(monkeypatch, active):
    import sys

    stats = torch.zeros(14, dtype=torch.int32) if active else None
    compute, dispatch = Mock(), Mock()
    monkeypatch.setitem(
        sys.modules,
        "deep_gemm",
        SimpleNamespace(
            get_symm_buffer_for_mega_moe=Mock(return_value=object()),
            mega_moe_pre_dispatch=dispatch,
            fp4_fp4_mega_moe=compute,
        ),
    )
    state = {}
    fn = definition(
        "layers/moe/ppu_mega_moe.py",
        "forward",
        dict(
            torch=torch,
            get_expert_recv_stats_kwargs=lambda experts: (
                {"cumulative_local_expert_recv_stats": stats}
                if stats is not None
                else {}
            ),
            envs=SimpleNamespace(
                SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK=SimpleNamespace(
                    get=lambda: 16
                )
            ),
            get_buffer=lambda *args: state,
            get_parallel=lambda: SimpleNamespace(
                moe_ep_group=SimpleNamespace(device_group=object())
            ),
        ),
    )
    experts = SimpleNamespace(
        _mega_moe_weights_built=True,
        num_experts=448,
        mega_l1_weights=object(),
        mega_l2_weights=object(),
    )
    for rows in (3, 0):
        output = fn(
            experts,
            torch.zeros(rows, 8),
            torch.zeros(rows, 2, dtype=torch.int32),
            torch.ones(rows, 2),
            hidden_size=8,
            intermediate_size=16,
            top_k=2,
        )
        assert output.shape == (rows, 8)
        assert (
            compute.call_args.kwargs.get("cumulative_local_expert_recv_stats") is stats
        )
        if not active:
            assert "cumulative_local_expert_recv_stats" not in compute.call_args.kwargs
        assert compute.call_args.args[1] is experts.mega_l1_weights
        assert dispatch.call_args.args[1].dtype == torch.int64


def test_megamoe_diagnostics_reports_delta_and_skips_prefill():
    from functools import wraps

    fn = definition(
        "eplb/incremental/balance_debug.py",
        "_wrap_mega",
        dict(wraps=wraps),
        "BalanceDebug",
    )
    stats = torch.tensor([20, 30], dtype=torch.int32)

    def kernel(*, cumulative_local_expert_recv_stats):
        if cumulative_local_expert_recv_stats is not None:
            cumulative_local_expert_recv_stats.add_(torch.tensor([3, 7]))

    debug = SimpleNamespace(
        capture_key="decode",
        counts={},
        _mega_count_layers={stats.data_ptr(): 5},
        measure=Mock(
            side_effect=lambda layer, name, fn, *args, **kwargs: fn(*args, **kwargs)
        ),
    )
    wrapped = fn(debug, kernel)
    wrapped(cumulative_local_expert_recv_stats=stats)
    assert debug.counts["decode"][5].tolist() == [3, 7]
    assert debug.measure.call_args.args[:2] == (5, "mega_moe_ms")
    wrapped(cumulative_local_expert_recv_stats=None)
    assert debug.measure.call_count == 1


@pytest.mark.parametrize("supported", [False, True])
def test_megamoe_startup_requires_receive_count_api(monkeypatch, supported):
    import sys

    def with_stats(*, cumulative_local_expert_recv_stats=None):
        pass

    def without_stats():
        pass

    monkeypatch.setitem(
        sys.modules,
        "deep_gemm",
        SimpleNamespace(
            fp4_fp4_mega_moe=with_stats if supported else without_stats,
        ),
    )
    monkeypatch.setitem(
        sys.modules, "sglang.srt.utils", SimpleNamespace(is_ppu=lambda: True)
    )
    module = SimpleNamespace(
        quant_method=type("Mxfp4MoEMethod", (), {})(),
        _mega_moe_weights_built=True,
        mega_l1_weights=(),
        mega_l2_weights=(),
    )
    fn = definition(
        "eplb/incremental/mover.py",
        "validate_mega_moe_experts",
        dict(
            _iter_moe_modules=lambda model: [(0, module)],
        ),
    )
    if supported:
        fn(object())
        module._mega_moe_weights_built = False
        with pytest.raises(ValueError, match="prepared MXFP4"):
            fn(object())
    else:
        with pytest.raises(ValueError, match="cumulative_local_expert_recv_stats"):
            fn(object())


def test_fused_front_cannot_bypass_eplb_remapping():
    descriptor = definition(
        "models/kimi_k3.py",
        "_routing_contract_ok",
        dict(
            cached_property=__import__("functools").cached_property,
        ),
        "KimiK3MoE",
    )
    module = SimpleNamespace(_eligible_for_fused_front=False, _uses_eplb=True)
    assert not descriptor.func(module)


def test_ppu_megamoe_cached_views_do_not_retain_transform_bases(monkeypatch):
    import sys
    import weakref

    bases = []

    def transform(l1, l2, *, activation):
        assert activation == "swiglu"

        def pack(tensor):
            base = tensor.clone()
            bases.append(weakref.ref(base))
            return base.reshape(tensor.shape)

        return tuple(map(pack, l1)), tuple(map(pack, l2))

    monkeypatch.setitem(
        sys.modules,
        "deep_gemm",
        SimpleNamespace(transform_weights_for_mega_moe=transform),
    )
    fn = definition("layers/moe/ppu_mega_moe.py", "prepare_weights", {})
    experts = torch.nn.Module()
    names = ("w13_weight", "w13_weight_scale", "w2_weight", "w2_weight_scale")
    for value, name in enumerate(names):
        experts.register_parameter(
            name,
            torch.nn.Parameter(
                torch.full((2, 4, 8), value, dtype=torch.uint8), requires_grad=False
            ),
        )
    fn(experts, activation="silu")
    for name, view in zip(names, experts.mega_l1_weights + experts.mega_l2_weights):
        owner = getattr(experts, name)
        assert view._base is None
        assert view.data_ptr() == owner.data_ptr()
        assert view.stride() == owner.stride()
        torch.testing.assert_close(view, owner)
    assert all(base() is None for base in bases)
