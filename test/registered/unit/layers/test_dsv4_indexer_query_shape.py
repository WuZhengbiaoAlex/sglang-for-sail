"""Indexer query dimensions follow the selected compression ratio."""

import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


@pytest.mark.parametrize("ratio", [1, 2, 4])
@pytest.mark.parametrize("rows", [1, 510])
def test_query_shape_without_other_compression_metadata(ratio, rows):
    path = (
        Path(__file__).resolve().parents[4]
        / "python/sglang/srt/layers/attention/deepseek_v4_backend.py"
    )
    tree = ast.parse(path.read_text())
    backend = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "DeepseekV4AttnBackend"
    )
    method = next(
        n
        for n in backend.body
        if isinstance(n, ast.FunctionDef) and n.name == "init_forward_metadata_indexer"
    )
    expand = Mock(return_value=object())
    namespace = dict(
        DSV4AttnMetadata=object,
        torch=SimpleNamespace(Size=tuple),
        PagedIndexerMetadata=SimpleNamespace,
        expand_index_page_table=expand,
        _is_xpu=False,
    )
    exec(
        compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"),
        namespace,
    )
    lengths = SimpleNamespace(shape=(rows,))
    core = SimpleNamespace(
        page_table=object(),
        c1_topk_lengths_clamp1=lengths if ratio == 1 else None,
        c2_topk_lengths_clamp1=lengths if ratio == 2 else None,
        c4_topk_lengths_raw=lengths if ratio == 4 else None,
    )
    runner = SimpleNamespace(
        page_size=256,
        index_n_heads=64,
        index_head_dim=128,
        enable_deepseek_v4_fp4_indexer=False,
        dsa_topk_backend=SimpleNamespace(should_use_topk_v2=lambda: True),
        token_to_kv_pool=SimpleNamespace(get_index_k_page_size=lambda _: 64),
    )
    result = namespace["init_forward_metadata_indexer"](
        runner, core, compress_ratio=ratio
    )
    assert result.q_fp8_shape == (rows, 1, 64, 128)
    assert result.compressed_seq_lens is lengths
    assert result.compressed_page_size == 64
    assert expand.called == (ratio != 4)
