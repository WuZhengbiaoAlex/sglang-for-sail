"""Storage isolation, publication safety and diagnostic contracts."""

import importlib
import json
import sys
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


@pytest.fixture
def modules(monkeypatch):
    root = Path(__file__).resolve().parents[4] / "python/sglang"
    # Exercise the real leaf modules without loading optional serving backends.
    for name, path in (
        ("sglang", root),
        ("sglang.srt", root / "srt"),
        ("sglang.srt.eplb", root / "srt/eplb"),
        ("sglang.srt.eplb.incremental", root / "srt/eplb/incremental"),
    ):
        mod = ModuleType(name)
        mod.__path__ = [str(path)]
        monkeypatch.setitem(sys.modules, name, mod)
    prefix = "sglang.srt.eplb.incremental."
    for name in list(sys.modules):
        if name.startswith(prefix):
            monkeypatch.delitem(sys.modules, name)
    resources = SimpleNamespace(expert_location_metadata=object())
    runtime = ModuleType("sglang.srt.runtime_context")
    runtime.get_resources = lambda: resources
    runtime.get_exec = Mock(side_effect=AssertionError("unexpected runtime lookup"))
    monkeypatch.setitem(sys.modules, runtime.__name__, runtime)
    environ = ModuleType("sglang.srt.environ")
    environ.envs = SimpleNamespace(
        SGLANG_INCREMENTAL_EPLB_DEBUG_DIR=SimpleNamespace(get=lambda: ""),
    )
    monkeypatch.setitem(sys.modules, environ.__name__, environ)
    location = ModuleType("sglang.srt.eplb.expert_location")

    @dataclass
    class Metadata:
        logical_to_rank_dispatch_physical_map: torch.Tensor

    location.ExpertLocationMetadata = Metadata
    monkeypatch.setitem(sys.modules, location.__name__, location)

    fused = ModuleType("sglang.srt.layers.moe.fused_moe_triton.layer")
    fused.FusedMoE = type("FusedMoE", (), {})
    monkeypatch.setitem(sys.modules, fused.__name__, fused)

    def load(name):
        return importlib.import_module(prefix + name)

    yield SimpleNamespace(
        load=load, resources=resources, runtime=runtime, Metadata=Metadata
    )
    for name in list(sys.modules):
        if name.startswith(prefix):
            sys.modules.pop(name, None)


def load_definitions(relative_path, names, namespace):
    """Execute adapter definitions without importing optional accelerator libraries."""
    import ast

    root = Path(__file__).resolve().parents[4]
    path = root / relative_path
    tree = ast.parse(path.read_text(), filename=str(path))
    definitions = [node for node in tree.body if getattr(node, "name", None) in names]
    assert {node.name for node in definitions} == set(names)
    exec(
        compile(ast.Module(body=definitions, type_ignores=[]), str(path), "exec"),
        namespace,
    )
    return SimpleNamespace(**namespace)


@pytest.mark.parametrize(
    "enabled,target,routed",
    [(False, True, True), (True, False, True), (True, True, False)],
)
def test_storage_isolation_never_reads_target_metadata(
    modules, enabled, target, routed
):
    cfg = modules.load("config")
    context = modules.load("model_context")
    getter = Mock(
        side_effect=AssertionError("metadata consulted outside target routed storage")
    )
    defs = load_definitions(
        "python/sglang/srt/layers/moe/fused_moe_triton/layer.py",
        {
            "_normalize_incremental_eplb_num_experts",
            "_maybe_normalize_incremental_eplb_num_experts",
        },
        dict(
            is_incremental_eplb_enabled=lambda: enabled,
            is_routed_expert_module=cfg.is_routed_expert_module,
            get_global_expert_location_metadata=getter,
        ),
    )
    with context.incremental_eplb_target_model_context() if target else nullcontext():
        assert (
            defs._maybe_normalize_incremental_eplb_num_experts(
                SimpleNamespace(uses_expert_location=routed), 128, 0
            )
            == 128
        )
    getter.assert_not_called()


@pytest.mark.parametrize("supplied", [384, 448])
def test_storage_accepts_logical_or_physical_count(modules, supplied):
    cfg = modules.load("config")
    context = modules.load("model_context")
    defs = load_definitions(
        "python/sglang/srt/layers/moe/fused_moe_triton/layer.py",
        {
            "_normalize_incremental_eplb_num_experts",
            "_maybe_normalize_incremental_eplb_num_experts",
        },
        dict(
            is_incremental_eplb_enabled=lambda: True,
            is_routed_expert_module=cfg.is_routed_expert_module,
            get_global_expert_location_metadata=lambda: SimpleNamespace(
                num_logical_experts=384, num_physical_experts=448
            ),
        ),
    )
    with context.incremental_eplb_target_model_context():
        assert (
            defs._maybe_normalize_incremental_eplb_num_experts(
                SimpleNamespace(), supplied, 0
            )
            == 448
        )
        with pytest.raises(ValueError, match="disable-shared-experts-fusion"):
            defs._maybe_normalize_incremental_eplb_num_experts(
                SimpleNamespace(), supplied, 1
            )
        with pytest.raises(ValueError, match="mismatch"):
            defs._maybe_normalize_incremental_eplb_num_experts(
                SimpleNamespace(), 385, 0
            )


@pytest.mark.parametrize("fails", [False, True])
def test_nested_draft_cannot_inherit_target_construction(modules, monkeypatch, fails):
    context = modules.load("model_context")
    original = modules.resources.expert_location_metadata
    with context.incremental_eplb_target_model_context():
        try:
            with context.incremental_eplb_target_model_context(is_target=False):
                assert not context.is_building_incremental_eplb_target()
                assert modules.resources.expert_location_metadata is original
                if fails:
                    raise ValueError("draft load failed")
        except ValueError:
            assert fails
        assert context.is_building_incremental_eplb_target()
        assert modules.resources.expert_location_metadata is original
    assert not context.is_building_incremental_eplb_target()


@pytest.mark.parametrize("failure", [None, "fence", "transfer"])
def test_publication_requires_safe_completed_transfer(modules, failure):
    module = modules.load("puller")
    puller = object.__new__(module.ExpertPuller)
    order = []

    def operation(name):
        def run(*args, **kwargs):
            if failure == name:
                raise RuntimeError(name)
            order.append(name)

        return run

    puller._debug = puller._debug_round = None
    puller._consume_publication = lambda: (7, True)
    puller._expand_batch = lambda: module.CopyBatch([100], [200], [64], [0], 1, 64)
    puller._stream = object()
    puller.planner = SimpleNamespace(
        assert_copy_safe=operation("fence"),
        commit=operation("commit"),
        stage_adoption=operation("adopt"),
    )
    puller.transport = SimpleNamespace(submit=operation("transfer"))
    puller._event = SimpleNamespace(record=operation("event"))
    puller.stats = dict(batches=0, experts=0, copies=0, bytes=0)
    if failure:
        with pytest.raises(RuntimeError, match=failure):
            puller._submit_publication()
        assert "adopt" not in order and "event" not in order
        if failure == "fence":
            assert not order
    else:
        puller._submit_publication()
        assert order == ["fence", "commit", "transfer", "event", "adopt"]
        assert puller._inflight_generation == 7


@pytest.mark.parametrize("transpose", [False, True])
def test_weight_aliases_preserve_values_and_storage(modules, transpose):
    module = modules.load("mover")
    owner = torch.nn.Parameter(torch.arange(24.0).reshape(4, 2, 3), requires_grad=False)
    if transpose:
        owner.data = owner.data.transpose(1, 2)
    view, expected, strides = owner.data, owner.clone(), owner.stride()
    model = SimpleNamespace(
        routed_experts_weights_of_layer={0: [view]}, named_modules=lambda: []
    )
    mover = object.__new__(module.ExpertWeightMover)
    mover.weight_aliases = module.collect_weight_aliases(model, [{"weight": owner}])
    mover._initialize_spares = lambda *args: None
    size = owner.numel() * owner.element_size()
    arena = torch.empty(size, dtype=torch.uint8)
    mover._rehome_regions(
        arena, [SimpleNamespace(tensor=owner, total_bytes=size, offset=0)]
    )
    assert owner.data_ptr() == view.data_ptr() == arena.data_ptr()
    assert owner.stride() == view.stride() == strides
    torch.testing.assert_close(owner, expected)
    torch.testing.assert_close(view, expected)
    with pytest.raises(ValueError, match="weight contract"):
        module.collect_weight_aliases(model, [{"weight": torch.zeros_like(owner)}])


@pytest.mark.parametrize(
    "attribute", ["is_shared_fused_moe", "uses_expert_location", "is_draft"]
)
def test_shared_and_draft_weights_are_not_collected(modules, attribute):
    module = modules.load("mover")
    expert = module.FusedMoE()
    setattr(expert, attribute, attribute != "uses_expert_location")
    assert (
        module._iter_moe_modules(
            SimpleNamespace(named_modules=lambda: [("expert", expert)])
        )
        == []
    )


class Event:
    def __init__(self, *, enable_timing=True):
        assert enable_timing
        self.ready = False
        self.stamp = 0

    def record(self, stream):
        self.stamp = stream

    def query(self):
        return self.ready

    def elapsed_time(self, other):
        assert self.ready and other.ready
        return other.stamp - self.stamp

    def synchronize(self):
        raise AssertionError("diagnostics must not synchronize")


def test_transfer_log_is_complete_without_blocking(tmp_path, modules):
    transfer_debug = modules.load("transfer_debug")
    debug = transfer_debug.TransferDebug(
        tmp_path, rank=1, ranks_per_node=1, event_factory=Event
    )
    trace = transfer_debug.RoundTrace(7, Event)
    trace.data.update(
        generation=7,
        global_work=True,
        local_experts=1,
        local_region_copies=1,
        local_bytes=4096,
    )
    move = debug.move(0, 0, 1, 0, 4096, expert=176)
    trace.data["moves"] = [move]
    for name, stamp in [
        ("boundary_start", 0),
        ("target_start", 1),
        ("target_end", 10),
        ("copy_start", 4),
        ("copy_end", 7),
    ]:
        trace.mark(name, stamp)
    try:
        debug.submitted(trace)
        debug.poll()
        assert len(debug.pending) == 1
        for event in trace.events.values():
            event.ready = True
        debug.poll()
        assert not debug.pending
    finally:
        debug.close()
    rows = [json.loads(line) for line in debug.path.read_text().splitlines()]
    record = next(row for row in rows if row["type"] == "round")
    assert record["moves"] == [
        dict(
            layer=0,
            expert=176,
            src_rank=0,
            src_slot=1,
            dst_rank=1,
            dst_slot=0,
            src_node=0,
            dst_node=1,
            bytes=4096,
        )
    ]
    timing = next(row for row in rows if row["type"] == "timing")
    assert timing["copy_stream_ms"] == 3
    assert timing["target_replay_region_ms"] == 9
    assert timing["copy_done_minus_target_end_ms"] == -3
    assert rows[-1]["dropped_records"] == rows[-1]["incomplete_timings"] == 0


@pytest.mark.parametrize(
    "failure", ["worker", "copy", "plan", "peers", "transport", "debug", None]
)
def test_shutdown_attempts_all_cleanup_after_failure(modules, monkeypatch, failure):
    module = modules.load("puller")
    order = []

    def operation(name):
        def run(*args, **kwargs):
            order.append(name)
            if name == failure:
                raise RuntimeError(name)

        return run

    monkeypatch.setattr(module.torch.cuda, "device", lambda _: nullcontext())
    puller = object.__new__(module.ExpertPuller)
    puller._closed = False
    puller._finish_pending = operation("worker")
    puller._executor = SimpleNamespace(shutdown=operation("executor"))
    puller._stream = SimpleNamespace(device=0, synchronize=operation("copy"))
    puller.planner = SimpleNamespace(
        plan_stream=SimpleNamespace(synchronize=operation("plan"))
    )
    puller.transport = SimpleNamespace(
        requires_peer_memory=False,
        synchronize_shutdown=operation("peers"),
        close=operation("transport"),
    )
    puller._debug = SimpleNamespace(close=operation("debug"))
    if failure:
        with pytest.raises(RuntimeError, match=failure):
            puller.close()
    else:
        puller.close()
    assert order == [
        "worker",
        "executor",
        "copy",
        "plan",
        "peers",
        "transport",
        "debug",
    ]
    puller.close()
    assert len(order) == 7


def test_mooncake_deregisters_other_arenas_after_failure(modules, monkeypatch):
    module = modules.load("transport")
    import torch.distributed as dist

    destroy = Mock()
    monkeypatch.setattr(dist, "destroy_process_group", destroy)
    unregister = Mock(side_effect=[0, -1, 0])
    transport = module.MooncakeTransport(
        SimpleNamespace(engine=SimpleNamespace(unregister_memory=unregister)), object()
    )
    arenas = [object(), object(), object()]
    transport._registered = [100, 200, 300]
    transport._arenas = arenas.copy()
    with pytest.raises(RuntimeError, match="deregistration"):
        transport.close()
    assert [call.args[0] for call in unregister.call_args_list] == [100, 200, 300]
    assert transport._registered == [200]
    assert transport._arenas == [arenas[1]]
    destroy.assert_called_once()
    unregister.side_effect = None
    unregister.return_value = 0
    transport.close()
    assert transport._registered == [] and transport._arenas == []
    destroy.assert_called_once()


def test_mooncake_shutdown_has_bounded_peer_wait(modules, monkeypatch):
    module = modules.load("transport")
    import torch.distributed as dist

    barrier = Mock(side_effect=RuntimeError("missing peer"))
    monkeypatch.setattr(dist, "monitored_barrier", barrier)
    transport = module.MooncakeTransport(None, object())
    with pytest.raises(RuntimeError, match="missing peer"):
        transport.synchronize_shutdown()
    assert barrier.call_args.kwargs["timeout"].total_seconds() == 10
    assert barrier.call_args.kwargs["wait_all_ranks"] is True


@pytest.mark.parametrize("protocol", ["rdma", "nvlink"])
def test_mooncake_pool_selection(modules, monkeypatch, protocol):
    module = modules.load("transport")
    envs = sys.modules["sglang.srt.environ"].envs
    envs.MOONCAKE_PROTOCOL = SimpleNamespace(get=lambda: protocol)
    for name in (
        "MC_FORCE_HCA",
        "MC_FORCE_TCP",
        "MC_INTRANODE_NVLINK",
        "MC_INTRA_NVLINK",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("MC_FORCE_MNNVL", "1")
    utils = ModuleType("sglang.srt.disaggregation.mooncake.utils")
    pool = object()
    utils.init_mooncake_custom_mem_pool = Mock(return_value=(True, pool, "NVLINK"))
    monkeypatch.setitem(sys.modules, utils.__name__, utils)
    selected = module.MooncakeTransport._create_memory_pool(3)
    if protocol == "nvlink":
        assert selected is pool
        utils.init_mooncake_custom_mem_pool.assert_called_once_with(
            "cuda:3", custom_mem_pool_type="NVLINK"
        )
        utils.init_mooncake_custom_mem_pool.return_value = (False, None, None)
        with pytest.raises(RuntimeError, match="memory pool is unavailable"):
            module.MooncakeTransport._create_memory_pool(3)
    else:
        assert selected is None
        utils.init_mooncake_custom_mem_pool.assert_not_called()


@pytest.mark.parametrize(
    "conflict",
    [None, "MC_FORCE_HCA", "MC_FORCE_TCP", "MC_INTRANODE_NVLINK", "MC_INTRA_NVLINK"],
)
def test_nvlink_rejects_ambiguous_transport_before_allocation(
    modules, monkeypatch, conflict
):
    module = modules.load("transport")
    sys.modules["sglang.srt.environ"].envs.MOONCAKE_PROTOCOL = SimpleNamespace(
        get=lambda: "nvlink"
    )
    monkeypatch.delenv("MC_FORCE_MNNVL", raising=False)
    if conflict is not None:
        monkeypatch.setenv("MC_FORCE_MNNVL", "1")
        # Mooncake checks presence, so even "0" is a conflicting override.
        monkeypatch.setenv(conflict, "0")
    with pytest.raises(ValueError, match="requires MC_FORCE_MNNVL"):
        module.MooncakeTransport._create_memory_pool(0)


def test_mover_allocates_mooncake_arena_inside_transport_pool(modules, monkeypatch):
    module = modules.load("transport")
    mover = modules.load("mover")
    from contextlib import contextmanager

    pool, arena, inside = object(), object(), []

    @contextmanager
    def use_pool(selected, *, device):
        assert selected is pool and device == "cuda:3"
        inside.append(True)
        try:
            yield
        finally:
            inside.pop()

    def allocate(size, *, dtype, device):
        assert inside and size == 4096 and dtype is torch.uint8 and device == "cuda:3"
        return arena

    monkeypatch.setattr(torch.cuda, "use_mem_pool", use_pool)
    monkeypatch.setattr(torch, "empty", allocate)
    state = SimpleNamespace(
        transport=module.MooncakeTransport(None, None, memory_pool=pool),
        _handles={},
        ep_size=4,
    )
    assert (
        mover.ExpertWeightMover._allocate_arena(state, 7, 4096, "cuda:3", None) is arena
    )
    assert state._handles[7] is None and not inside


def test_nvlink_registers_shared_backing_allocation_once(modules):
    transport = modules.load("transport").MooncakeTransport

    def tensor(ptr, size):
        return SimpleNamespace(
            data_ptr=lambda: ptr, numel=lambda: size, element_size=lambda: 1
        )

    first, second, third = tensor(100, 20), tensor(120, 20), tensor(500, 40)
    pool = SimpleNamespace(
        snapshot=lambda: [
            dict(address=100, total_size=100),
            dict(address=500, total_size=80),
        ]
    )
    instance = transport(None, None, memory_pool=pool)
    assert instance._registration_regions([first, second, third]) == [
        (100, 100, [first, second]),
        (500, 80, [third]),
    ]
    # A pointer crossing the allocation boundary must not be partially exposed.
    with pytest.raises(RuntimeError, match="one NVLink pool allocation"):
        instance._registration_regions([tensor(180, 30)])
    assert transport(None, None)._registration_regions([second]) == [(120, 20, second)]


def test_missing_extension_does_not_create_transport(modules, monkeypatch):
    planner = ModuleType("sglang.srt.eplb.incremental.planner")
    planner.is_available = Mock(return_value=False)
    planner.extension_error = Mock(return_value="missing compiler")
    monkeypatch.setitem(sys.modules, planner.__name__, planner)
    factory = Mock()
    defs = load_definitions(
        "python/sglang/srt/eplb/incremental/bootstrap.py",
        {"init_incremental_eplb"},
        dict(
            IncrementalEPLBComponents=object,
            get_global_expert_location_metadata=lambda: object(),
        ),
    )
    with pytest.raises(RuntimeError, match="missing compiler"):
        defs.init_incremental_eplb(
            tp_size=32,
            moe_ep_rank=0,
            device="cuda",
            gpu_id=0,
            model=None,
            transport_factory=factory,
        )
    factory.assert_not_called()
    planner.is_available.assert_called_once_with(peer_memory=False)


def test_scheduler_releases_host_resources_after_puller_failure():
    import ast
    from unittest.mock import Mock

    root = Path(__file__).resolve().parents[4]
    source = root / "python/sglang/srt/managers/scheduler.py"
    tree = ast.parse(source.read_text())
    run = next(
        n for n in tree.body if getattr(n, "name", None) == "run_scheduler_process"
    )
    block = next(
        n
        for n in ast.walk(run)
        if isinstance(n, ast.If)
        and isinstance(n.test, ast.Attribute)
        and n.test.attr == "gracefully_exit"
    )
    puller = SimpleNamespace(close=Mock(side_effect=RuntimeError("worker failed")))
    release, abort, logger = Mock(), Mock(), Mock()
    scheduler = SimpleNamespace(
        gracefully_exit=True,
        release_host_resources=release,
        tp_worker=SimpleNamespace(
            model_runner=SimpleNamespace(incremental_eplb_puller=puller)
        ),
    )
    exec(
        compile(ast.Module(body=[block], type_ignores=[]), str(source), "exec"),
        dict(scheduler=scheduler, logger=logger, abort_distributed_environment=abort),
    )
    release.assert_called_once()
    logger.exception.assert_called_once()


@pytest.mark.parametrize(
    "metadata_count,num_experts,num_local_experts,expected",
    [
        (None, 128, 4, 128),
        (None, 384, 12, 384),
        (384, 256, 12, 384),
        (448, 448, 14, 448),
    ],
)
def test_deepep_geometry_uses_model_placement_not_recorder(
    metadata_count, num_experts, num_local_experts, expected
):
    import ast

    source = Path(__file__).resolve().parents[4] / (
        "python/sglang/srt/layers/moe/token_dispatcher/deepep.py"
    )
    tree = ast.parse(source.read_text())
    cls = next(
        n for n in tree.body if getattr(n, "name", None) == "_DeepEPDispatcherImplBase"
    )
    init = next(n for n in cls.body if getattr(n, "name", None) == "__init__")
    getter = Mock(
        return_value=(
            SimpleNamespace(num_physical_experts=metadata_count)
            if metadata_count is not None
            else None
        )
    )
    recorder = Mock(side_effect=AssertionError("recorder does not define geometry"))
    layouts = set()
    namespace = dict(
        torch=torch,
        DeepEPMode=object,
        DeepEPBuffer=SimpleNamespace(
            _state=lambda: SimpleNamespace(registered_layouts=layouts)
        ),
        use_deepep=True,
        get_global_expert_location_metadata=getter,
        get_global_expert_distribution_recorder=recorder,
        envs=SimpleNamespace(
            SGLANG_OPT_ULTRA_EP_PPU=SimpleNamespace(get=lambda: False),
            SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=SimpleNamespace(
                get=lambda: 128
            ),
        ),
    )
    exec(
        compile(ast.Module(body=[init], type_ignores=[]), str(source), "exec"),
        namespace,
    )
    instance = SimpleNamespace(set_deepep_dispatcher_dtype=Mock())
    namespace["__init__"](
        instance,
        group=SimpleNamespace(size=lambda: 32),
        router_topk=8,
        permute_fusion=False,
        num_experts=num_experts,
        num_local_experts=num_local_experts,
        hidden_size=5120,
        params_dtype=torch.bfloat16,
        deepep_mode=SimpleNamespace(enable_low_latency=lambda: True),
    )
    assert instance.num_experts == expected
    assert layouts == {(128, 5120, expected)}
    recorder.assert_not_called()


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("draft", [False, True])
@pytest.mark.parametrize("fails", [False, True])
def test_model_load_placement_isolation_independent_of_eplb(
    modules, monkeypatch, enabled, draft, fails
):
    import ast
    import contextlib

    class Bag(SimpleNamespace):
        @contextlib.contextmanager
        def override(self, **updates):
            old = {key: getattr(self, key) for key in updates}
            self.__dict__.update(updates)
            try:
                yield
            finally:
                self.__dict__.update(old)

    target_metadata = object()
    resources = Bag(expert_location_metadata=target_metadata)
    moe = Bag(ep_num_redundant_experts=64)
    context = modules.load("model_context")
    definitions = load_definitions(
        "python/sglang/srt/layers/moe/utils.py",
        {"draft_expert_placement_context"},
        dict(
            contextmanager=contextlib.contextmanager,
            get_resources=lambda: resources,
            get_exec=lambda: SimpleNamespace(moe=moe),
        ),
    )
    utils = ModuleType("sglang.srt.layers.moe.utils")
    utils.draft_expert_placement_context = definitions.draft_expert_placement_context
    monkeypatch.setitem(sys.modules, utils.__name__, utils)
    source = (
        Path(__file__).resolve().parents[4]
        / "python/sglang/srt/model_executor/model_runner.py"
    )
    tree = ast.parse(source.read_text())
    cls = next(
        node for node in tree.body if getattr(node, "name", None) == "ModelRunner"
    )
    method = next(
        node
        for node in cls.body
        if getattr(node, "name", None) == "load_model_with_expert_placement_context"
    )
    namespace = dict(contextlib=contextlib, is_incremental_eplb_enabled=lambda: enabled)
    exec(
        compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"),
        namespace,
    )

    def load():
        assert resources.expert_location_metadata is (
            None if draft else target_metadata
        )
        assert moe.ep_num_redundant_experts == (0 if draft else 64)
        assert context.is_building_incremental_eplb_target() == (enabled and not draft)
        if fails:
            raise ValueError("model load failed")

    runner = SimpleNamespace(is_draft_worker=draft, load_model=Mock(side_effect=load))
    with (
        pytest.raises(ValueError, match="model load failed") if fails else nullcontext()
    ):
        namespace[method.name](runner)
    runner.load_model.assert_called_once_with()
    assert resources.expert_location_metadata is target_metadata
    assert moe.ep_num_redundant_experts == 64
    assert not context.is_building_incremental_eplb_target()


@pytest.fixture
def deepep_buffer_defs():
    from enum import Enum, IntEnum, auto
    from typing import NamedTuple

    class Mode(Enum):
        NORMAL = auto()
        LOW_LATENCY = auto()
        AUTO = auto()

        def enable_low_latency(self):
            return self in (Mode.LOW_LATENCY, Mode.AUTO)

        def is_low_latency(self):
            return self is Mode.LOW_LATENCY

        def is_normal(self):
            return self is Mode.NORMAL

    return load_definitions(
        "python/sglang/srt/layers/moe/token_dispatcher/deepep.py",
        {"DeepEPBuffer", "DeepEPDispatchMode"},
        dict(
            NamedTuple=NamedTuple,
            dist=torch.distributed,
            IntEnum=IntEnum,
            auto=auto,
            DeepEPMode=Mode,
        ),
    ), Mode


def test_deepep_shared_capacity_accepts_target_draft_layouts(
    deepep_buffer_defs, monkeypatch
):
    defs, mode = deepep_buffer_defs
    group = SimpleNamespace(size=lambda: 32)
    state = SimpleNamespace(
        buffer=Mock(low_latency_mode=True),
        num_rdma_bytes=512 * 5120 * 384,
        num_qps_per_rank=12,
    )
    buffer = defs.DeepEPBuffer
    monkeypatch.setattr(buffer, "_state", classmethod(lambda cls: state))
    buffer.get_deepep_buffer.__func__.__globals__["Buffer"] = SimpleNamespace(
        get_low_latency_rdma_size_hint=lambda tokens, width, ranks, experts: (
            tokens * width * experts
        )
    )
    # K3.1 draft can differ in both width and expert count. Capacity, rather
    # than equality with the target's geometry, determines compatibility.
    for tokens, width, experts in [(512, 5120, 384), (256, 4096, 128)]:
        assert (
            buffer.get_deepep_buffer(group, width, 2, mode.LOW_LATENCY, tokens, experts)
            is state.buffer
        )
    for tokens, width, experts in [(512, 8192, 384), (128, 1024, 416)]:
        with pytest.raises(RuntimeError, match="cannot accommodate"):
            buffer.get_deepep_buffer(group, width, 2, mode.LOW_LATENCY, tokens, experts)


def test_deepep_layout_transition_does_not_depend_on_balancer(
    deepep_buffer_defs, monkeypatch
):
    defs, mode = deepep_buffer_defs
    target, draft = (512, 5120, 384), (512, 5120, 128)
    state = SimpleNamespace(
        buffer=Mock(low_latency_mode=True),
        dispatch_mode=defs.DeepEPDispatchMode.LOW_LATENCY,
        low_latency_layout=target,
    )
    buffer = defs.DeepEPBuffer
    monkeypatch.setattr(buffer, "_state", classmethod(lambda cls: state))
    buffer.set_dispatch_mode_as_low_latency(target)
    state.buffer.clean_low_latency_buffer.assert_not_called()
    for layout in (draft, target):
        state.buffer.reset_mock()
        buffer.set_dispatch_mode_as_low_latency(layout)
        state.buffer.clean_low_latency_buffer.assert_called_once_with(*layout)
        assert state.low_latency_layout == layout
    state.buffer.reset_mock()
    buffer.set_dispatch_mode_as_normal()
    buffer.set_dispatch_mode_as_low_latency(target)
    state.buffer.clean_low_latency_buffer.assert_called_once_with(*target)

    adapter_defs = load_definitions(
        "python/sglang/srt/model_executor/runner_utils/deepep_adapter.py",
        {"DeepEPCudaGraphRunnerAdapter"},
        dict(
            DeepEPBuffer=buffer,
            get_moe_a2a_backend=lambda: SimpleNamespace(is_deepep=lambda: True),
            get_deepep_mode=lambda: SimpleNamespace(
                resolve=lambda **kwargs: mode.LOW_LATENCY
            ),
        ),
    )
    adapters = []
    for layout in (target, draft):
        adapter = adapter_defs.DeepEPCudaGraphRunnerAdapter()
        adapter.capture(is_extend_in_batch=False)
        buffer.set_dispatch_mode_as_low_latency(layout)
        adapter.capture_complete()
        adapters.append(adapter)
    for adapter, layout in zip(adapters * 2, (target, draft) * 2):
        state.buffer.reset_mock()
        adapter.replay()
        assert state.low_latency_layout == layout
        state.buffer.clean_low_latency_buffer.assert_called_once_with(*layout)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))


def test_cpu_publication_generations_and_empty_plans(modules):
    planner_module = modules.load("planner")
    puller_module = modules.load("puller")
    planner = object.__new__(planner_module.IncrementalPlanner)
    planner.publication = (0, False)
    planner._cpu_planner = SimpleNamespace(stage=Mock())
    planner.next_p2l = planner.next_owners = object()
    planner.pull_list = planner.pull_experts = object()
    planner.ep_rank = 0
    puller = object.__new__(puller_module.ExpertPuller)
    puller.planner, puller._last_seq = planner, 0
    for generation, work in enumerate([True, False, False, True], 1):
        planner.plan = lambda: work
        planner._plan_and_publish()
        assert puller._consume_publication() == (generation, work)
        with pytest.raises(RuntimeError, match="Invalid IncrementalEPLB"):
            puller._consume_publication()
    assert planner._cpu_planner.stage.call_count == 2
    planner._cpu_planner.stage.side_effect = RuntimeError("invalid copy")
    with pytest.raises(RuntimeError, match="invalid copy"):
        planner._plan_and_publish()
    assert planner.publication == (4, True)
