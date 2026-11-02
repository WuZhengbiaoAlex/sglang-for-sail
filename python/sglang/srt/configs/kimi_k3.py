from transformers.configuration_utils import PretrainedConfig

from sglang.srt.configs.kimi_linear import KimiLinearConfig


class KimiK3VisionConfig(PretrainedConfig):
    model_type = "kimi_k3_vision"

    def __init__(
        self,
        patch_size: int = 14,
        init_pos_emb_height: int = 64,
        init_pos_emb_width: int = 64,
        init_pos_emb_time: int = 4,
        pos_emb_type: str = "divided_fixed",
        vt_num_attention_heads: int = 12,
        vt_num_hidden_layers: int = 27,
        vt_hidden_size: int = 1024,
        vt_intermediate_size: int = 4096,
        merge_kernel_size: tuple[int, int] = (2, 2),
        video_attn_type: str = "spatial_temporal",
        merge_type: str = "sd2_tpool",
        _attn_implementation: str = "flash_attention_2",
        mm_projector_type: str = "patchmergerv2",
        mm_hidden_size: int | None = None,
        projector_hidden_act: str = "gelu",
        projector_ln_eps: float = 1e-5,
        qkv_hidden_size: int = 1536,
        norm_type: str = "rmsnorm",
        attn_bias: bool = False,
        patch_embed_proj_bias: bool = False,
        mlp_type: str = "mlp2",
        linear_bias: bool = False,
        activation_func: str = "gelu_pytorch_tanh",
        pos_emb_interpolation_mode: str = "bilinear",
        text_hidden_size: int = 2304,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.patch_size = patch_size
        self.init_pos_emb_height = init_pos_emb_height
        self.init_pos_emb_width = init_pos_emb_width
        self.init_pos_emb_time = init_pos_emb_time
        self.pos_emb_type = pos_emb_type
        self.vt_num_attention_heads = vt_num_attention_heads
        self.vt_num_hidden_layers = vt_num_hidden_layers
        self.vt_hidden_size = vt_hidden_size
        self.vt_intermediate_size = vt_intermediate_size
        self.merge_kernel_size = tuple(merge_kernel_size)
        self.video_attn_type = video_attn_type
        self.merge_type = merge_type
        self._attn_implementation = _attn_implementation

        self.mm_projector_type = mm_projector_type
        self.mm_hidden_size = (
            mm_hidden_size if mm_hidden_size is not None else vt_hidden_size
        )
        self.projector_hidden_act = projector_hidden_act
        self.projector_ln_eps = projector_ln_eps
        self.text_hidden_size = text_hidden_size

        self.qkv_hidden_size = qkv_hidden_size
        self.norm_type = norm_type
        self.attn_bias = attn_bias
        self.patch_embed_proj_bias = patch_embed_proj_bias
        self.mlp_type = mlp_type
        self.linear_bias = linear_bias
        self.activation_func = activation_func
        self.pos_emb_interpolation_mode = pos_emb_interpolation_mode

        # Aliases consumed by the K2.5 vision implementation.
        self.num_attention_heads = vt_num_attention_heads
        self.num_hidden_layers = vt_num_hidden_layers
        self.hidden_size = vt_hidden_size
        self.intermediate_size = vt_intermediate_size


class KimiK3Config(PretrainedConfig):
    model_type = "kimi_k3"

    def __init__(
        self,
        text_config: dict | KimiLinearConfig | None = None,
        vision_config: dict | KimiK3VisionConfig | None = None,
        ignore_index: int = -100,
        media_placeholder_token_id: int = 163605,
        pad_token_id: int = 0,
        image_placeholder: str = "<|kimi_image_placeholder|>",
        **kwargs,
    ):
        if text_config is None:
            self.text_config = KimiLinearConfig()
        elif isinstance(text_config, dict):
            self.text_config = KimiLinearConfig(**text_config)
        else:
            self.text_config = text_config

        if self.text_config.num_nextn_predict_layers:
            # Preserve checkpoint numbering before debug layer-count overrides
            # are applied by get_config(). The external MTP export uses layer 93.
            if not hasattr(self.text_config, "mtp_checkpoint_layer"):
                self.text_config.mtp_checkpoint_layer = (
                    self.text_config.num_hidden_layers
                )

        if vision_config is None:
            self.vision_config = KimiK3VisionConfig()
        elif isinstance(vision_config, dict):
            self.vision_config = KimiK3VisionConfig(**vision_config)
        else:
            self.vision_config = vision_config

        if self.vision_config.text_hidden_size != self.text_config.hidden_size:
            self.vision_config.text_hidden_size = self.text_config.hidden_size

        self.ignore_index = ignore_index
        self.media_placeholder_token_id = media_placeholder_token_id
        self.image_placeholder = image_placeholder

        if getattr(self.text_config, "quantization_config", None) is not None:
            self.quantization_config = self.text_config.quantization_config

        super().__init__(pad_token_id=pad_token_id, **kwargs)

    @property
    def hidden_size(self) -> int:
        return self.text_config.hidden_size

    @property
    def vocab_size(self) -> int:
        return self.text_config.vocab_size


class KimiK31Config(KimiK3Config):
    model_type = "kimi_k31"
    has_no_defaults_at_init = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        text = self.text_config
        dsa = getattr(text, "dsa_config", None)
        if not isinstance(dsa, dict):
            raise ValueError("Kimi-K3.1 requires text_config.dsa_config")
        # The checkpoint nests DSA under text_config; the attention backend,
        # cache builder and PD metadata also consult the outer HF config.
        normalized = {
            "index_topk": dsa["index_topk"],
            "index_head_dim": dsa["index_head_dim"],
            "index_n_heads": dsa["index_num_heads"],
            "index_rope_dim": dsa.get("index_rope_dim", 0),
            "index_k_norm_type": dsa.get("index_k_norm_type", "layernorm"),
            "index_topk_freq": dsa.get("index_topk_freq", 1),
            "index_skip_topk_offset": dsa.get("index_skip_topk_offset", 1),
        }
        for name, value in normalized.items():
            setattr(self, name, value)
            setattr(text, name, value)
        text.architectures = ["KimiK31LinearForCausalLM"]
        self.architectures = ["KimiK31ForConditionalGeneration"]


def map_kimi_k31_mtp_weight_name(name: str, checkpoint_layer: int) -> str | None:
    prefix = f"language_model.model.layers.{checkpoint_layer}."
    if not name.startswith(prefix):
        raise ValueError(f"Unexpected tensor in Kimi MTP draft-only export: {name}")
    name = name[len(prefix) :]
    # The export's nonexpert-provenance.json documents these fallback keys
    # as ignored by the reference strict loader.
    if name in {
        "self_attention_res_norm.weight",
        "self_attention_res_proj.weight",
        "mlp_res_norm.weight",
        "mlp_res_proj.weight",
    }:
        return None
    if name.startswith("shared_head.head."):
        return name.replace("shared_head.head.", "lm_head.", 1)
    if name.startswith(
        ("embed_tokens.", "eh_proj.", "enorm.", "hnorm.", "shared_head.norm.")
    ):
        return "model." + name
    return "model.layers.0." + name.replace("block_sparse_moe.", "mlp.", 1)
