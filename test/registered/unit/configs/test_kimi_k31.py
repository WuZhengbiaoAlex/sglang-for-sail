"""K3.1 nested DSA config must reach model, cache and backend consumers."""

import unittest
from types import SimpleNamespace

from sglang.srt.configs.kimi_k3 import (
    KimiK3Config,
    KimiK31Config,
    map_kimi_k31_mtp_weight_name,
)
from sglang.srt.configs.model_config import (
    ModelConfig,
    get_dsa_index_head_dim,
    get_dsa_index_n_heads,
    get_dsa_index_topk,
    is_deepseek_dsa,
    is_kimi_k3,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestKimiK31Config(CustomTestCase):
    def test_nested_dsa_roundtrip(self):
        config = KimiK31Config(
            text_config={
                "dsa_config": {
                    "dsa_layers": [4, 8],
                    "index_topk": 2048,
                    "index_head_dim": 128,
                    "index_num_heads": 32,
                    "index_rope_dim": 0,
                },
            }
        )
        restored = KimiK31Config.from_dict(config.to_dict())
        for value in (config, restored):
            for view in (value, value.text_config):
                self.assertTrue(is_deepseek_dsa(view))
                self.assertTrue(is_kimi_k3(view))
                self.assertEqual(get_dsa_index_topk(view), 2048)
                self.assertEqual(get_dsa_index_head_dim(view), 128)
                self.assertEqual(get_dsa_index_n_heads(view), 32)
                self.assertEqual(view.index_rope_dim, 0)
            # Transformers logs a diff during from_pretrained: no default
            # construction may discard or require a second DSA config.
            self.assertIn('"kimi_k31"', value.to_json_string())

    def test_k3_remains_dense_mla(self):
        config = KimiK3Config(architectures=["KimiK3ForConditionalGeneration"])
        self.assertFalse(is_deepseek_dsa(config))
        self.assertFalse(is_deepseek_dsa(config.text_config))

    def test_missing_dsa_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "dsa_config"):
            KimiK31Config()

    def test_draft_constructs_only_one_dense_layer(self):
        outer = KimiK3Config(
            architectures=["KimiK3ForConditionalGeneration"],
            text_config={"num_hidden_layers": 93, "num_nextn_predict_layers": 1},
        )
        # A shortened target may share json_model_override_args with its
        # real draft; the export still stores the trained block at layer 93.
        outer.text_config.num_hidden_layers = 4
        holder = SimpleNamespace(
            is_draft_model=True, hf_config=outer, hf_text_config=outer.text_config
        )
        ModelConfig._config_draft_model(holder)
        self.assertEqual(holder.hf_config.architectures, ["KimiK31ForCausalLMNextN"])
        self.assertEqual(holder.hf_config.num_hidden_layers, 1)
        self.assertEqual(holder.hf_config.mtp_checkpoint_layer, 93)
        self.assertEqual(holder.hf_config.full_attention_layer_ids, [0])
        self.assertEqual(holder.hf_config.linear_layer_ids, [])
        self.assertIsNone(holder.hf_config.attn_res_block_size)

    def test_mtp_weight_mapping(self):
        prefix = "language_model.model.layers.93."
        pairs = {
            "embed_tokens.weight": "model.embed_tokens.weight",
            "eh_proj.weight": "model.eh_proj.weight",
            "shared_head.head.weight": "lm_head.weight",
            "shared_head.norm.weight": "model.shared_head.norm.weight",
            "self_attn.q_a_proj.weight": "model.layers.0.self_attn.q_a_proj.weight",
            "block_sparse_moe.experts.895.w1.weight_packed": "model.layers.0.mlp.experts.895.w1.weight_packed",
            "mlp_res_proj.weight": None,
        }
        for source, expected in pairs.items():
            with self.subTest(source=source):
                self.assertEqual(
                    map_kimi_k31_mtp_weight_name(prefix + source, 93), expected
                )
        with self.assertRaisesRegex(ValueError, "draft-only"):
            map_kimi_k31_mtp_weight_name(
                "language_model.model.layers.92.self_attn.q_a_proj.weight", 93
            )


if __name__ == "__main__":
    unittest.main()
