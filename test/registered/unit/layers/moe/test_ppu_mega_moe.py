"""CPU contract tests for the PPU MegaMoE adapter (no model/server)."""

import sys
import types
import unittest
from unittest.mock import patch

import torch
from sglang.srt.layers.moe import ppu_mega_moe as adapter
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestPPUMegaMoE(CustomTestCase):
    def test_platform_gate_keeps_unsupported_devices_and_overlap_rejected(self):
        from sglang.srt.arg_groups import mega_moe_hook as hook

        cfg = types.SimpleNamespace(
            moe_a2a_backend="megamoe",
            enable_two_batch_overlap=False,
            enable_single_batch_overlap=False,
        )
        platform = types.SimpleNamespace(is_cuda=False, is_hip=False)
        with patch.object(hook, "resolving_view", return_value=cfg), patch.object(
            hook, "get_platform", return_value=platform
        ), patch.object(hook, "is_ppu", return_value=True), patch(
            "sglang.srt.utils.get_device_sm", return_value=89
        ):
            self.assertIsNone(hook.check_mega_moe_compat(object()))
            cfg.enable_two_batch_overlap = True
            with self.assertRaises(ValueError):
                hook.check_mega_moe_compat(object())
            cfg.enable_two_batch_overlap = False
        with patch.object(hook, "resolving_view", return_value=cfg), patch.object(
            hook, "get_platform", return_value=platform
        ), patch.object(hook, "is_ppu", return_value=False):
            with self.assertRaises(ValueError):
                hook.check_mega_moe_compat(object())

    def test_weight_transform_is_once_and_releases_old_layout(self):
        e = torch.nn.Module()
        for name in ["w13_weight", "w13_weight_scale", "w2_weight", "w2_weight_scale"]:
            e.register_parameter(
                name,
                torch.nn.Parameter(
                    torch.zeros((1, 2, 2), dtype=torch.uint8), requires_grad=False
                ),
            )
        calls = []

        def transform(a, b, *, activation):
            calls.append(activation)
            return tuple(x + 1 for x in a), tuple(x + 2 for x in b)

        with patch.dict(
            sys.modules,
            {
                "deep_gemm": types.SimpleNamespace(
                    transform_weights_for_mega_moe=transform
                )
            },
        ):
            adapter.prepare_weights(e, activation="situ")
            adapter.prepare_weights(e, activation="situ")
        self.assertEqual(calls, ["situ"])
        self.assertEqual(e.w13_weight.data_ptr(), e.mega_l1_weights[0].data_ptr())
        self.assertEqual(e.w2_weight_scale.data_ptr(), e.mega_l2_weights[1].data_ptr())
        self.assertTrue(torch.all(e.w13_weight == 1))
        self.assertTrue(torch.all(e.w2_weight_scale == 2))

    def test_forward_routes_capacity_empty_rank_and_reuse(self):
        buffers, allocations, dispatches, computes = {}, [], [], []
        group = object()
        experts = types.SimpleNamespace(
            _mega_moe_weights_built=True,
            num_experts=16,
            mega_l1_weights=object(),
            mega_l2_weights=object(),
        )

        def allocate(g, **kw):
            allocations.append((g, kw))
            return object()

        def dispatch(x, ids, weights, buf):
            self.assertEqual(ids.dtype, torch.int64)
            self.assertEqual(weights.dtype, torch.float32)
            dispatches.append((x.clone(), weights.clone()))

        def compute(y, l1, l2, buf, **kw):
            self.assertIs(l1, experts.mega_l1_weights)
            self.assertEqual(kw["activation"], "situ")
            self.assertEqual(kw["situ_beta"], 4.0)
            self.assertEqual(kw["recipe"], (1, 1, 32))
            x, w = dispatches[-1]
            y.copy_(x * w.sum(-1, keepdim=True))
            computes.append(y.shape)

        dg = types.SimpleNamespace(
            get_symm_buffer_for_mega_moe=allocate,
            mega_moe_pre_dispatch=dispatch,
            fp4_fp4_mega_moe=compute,
        )
        capacity = types.SimpleNamespace(get=lambda: 4)
        env = types.SimpleNamespace(
            SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK=capacity
        )
        with patch.dict(sys.modules, {"deep_gemm": dg}), patch.object(
            adapter, "envs", env
        ), patch.object(
            adapter, "get_buffer", lambda name, factory: buffers
        ), patch.object(
            adapter,
            "get_parallel",
            lambda: types.SimpleNamespace(
                moe_ep_group=types.SimpleNamespace(device_group=group)
            ),
        ):

            def run(n):
                return adapter.forward(
                    experts,
                    torch.ones((n, 4), dtype=torch.bfloat16),
                    torch.zeros((n, 2), dtype=torch.int32),
                    torch.ones((n, 2)),
                    hidden_size=4,
                    intermediate_size=8,
                    top_k=2,
                    activation="situ",
                )

            self.assertTrue(torch.all(run(2) == 2))
            self.assertTrue(torch.all(run(1) == 2))
            self.assertEqual(run(0).shape, (0, 4))
            with self.assertRaises(ValueError):
                run(5)
        self.assertEqual(len(allocations), 1)
        self.assertEqual(allocations[0][1]["mma_type"], "fp4xfp4")
        self.assertEqual(allocations[0][1]["activation"], "situ")
        self.assertEqual(len(computes), 3)


if __name__ == "__main__":
    unittest.main()
