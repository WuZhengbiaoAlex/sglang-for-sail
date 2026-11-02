"""Kimi K3.1 external, single-layer MTP draft checkpoint."""

from typing import Iterable, Tuple

import torch
from torch import nn

from sglang.srt.configs.kimi_k3 import map_kimi_k31_mtp_weight_name
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import ReplicatedLinear
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
    get_embedding_tp_kwargs,
)
from sglang.srt.models.kimi_k3 import (
    KimiK3DecoderLayer,
    KimiK3LinearForCausalLM,
    _sp_all_gather_rows,
)
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils import BumpAllocator


class KimiK31ForCausalLMNextN(KimiK3LinearForCausalLM):
    """The export owns its embedding/head and conditions on fixed target hidden states."""

    def __init__(self, config, quant_config=None, prefix=""):
        nn.Module.__init__(self)
        self.config = config
        self.quant_config = quant_config
        self.pp_group = get_parallel().pp_group
        self.checkpoint_layer = config.mtp_checkpoint_layer
        if config.num_hidden_layers != 1 or config.is_linear_attn:
            raise ValueError("Kimi K3.1 MTP requires exactly one dense MLA draft layer")
        self.model = nn.Module()
        self.model.start_layer, self.model.end_layer = 0, 1
        self.model.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            prefix="model.embed_tokens",
            **get_embedding_tp_kwargs(),
        )
        self.model.enorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.model.hnorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.model.eh_proj = ReplicatedLinear(
            config.hidden_size * 2,
            config.hidden_size,
            bias=False,
            quant_config=None,
            prefix="model.eh_proj",
        )
        self.model.layers = nn.ModuleList(
            [
                KimiK3DecoderLayer(
                    config, 0, quant_config=quant_config, prefix="model.layers.0"
                )
            ]
        )
        self.model.shared_head = nn.Module()
        self.model.shared_head.norm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=None,
            prefix="lm_head",
            use_attn_tp_group=get_parallel().enable_dp_lm_head,
        )
        self.logits_processor = LogitsProcessor(config)

    def set_embed_and_head(self, embed, head):
        # The external MTP checkpoint contains trained copies of both tensors.
        # EagleWorker calls this hook for all NextN models; retain those copies.
        return

    @torch.no_grad()
    def forward(self, input_ids, positions, forward_batch, input_embeds=None):
        fixed_hidden = forward_batch.spec_info.hidden_states
        embeds = (
            self.model.embed_tokens(input_ids) if input_embeds is None else input_embeds
        )
        hidden = self.model.eh_proj(
            torch.cat(
                (self.model.enorm(embeds), self.model.hnorm(fixed_hidden)), dim=-1
            )
        )[0]
        allocator = BumpAllocator(
            buffer_size=2, dtype=torch.float32, device=hidden.device
        )
        decoder = self.model.layers[0]
        residual = hidden
        hidden = decoder.input_layernorm(hidden)
        hidden = decoder._run_self_attn(hidden, positions, forward_batch, allocator)
        hidden, shard_lo, _ = decoder._finish_attn_reduce(hidden, allow_scatter=True)
        if shard_lo >= 0:
            residual = residual[shard_lo : shard_lo + hidden.shape[0]]
        hidden, residual = decoder.post_attention_layernorm(hidden, residual)
        hidden = decoder.mlp(hidden, forward_batch=forward_batch)
        if hidden.shape[0] > 0:
            hidden = self.model.shared_head.norm(hidden, residual)[0]
        if shard_lo >= 0:
            hidden = _sp_all_gather_rows(hidden)
        # Supplying auxiliary hidden states lets LogitsProcessor perform the
        # same row selection as the logits in prefill, decode and graph replay.
        # Carry the target condition across all Spec3 draft steps, as in the
        # checkpoint's fixed-hidden offline evaluation.
        return self.logits_processor(
            input_ids,
            hidden,
            self.lm_head,
            forward_batch,
            aux_hidden_states=[fixed_hidden],
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        required = {
            name for name, _ in self.named_parameters() if ".experts." not in name
        }

        def remapped():
            for name, tensor in weights:
                mapped = map_kimi_k31_mtp_weight_name(name, self.checkpoint_layer)
                if mapped is not None:
                    yield mapped, tensor

        loaded = super().load_weights(remapped())
        missing = required - loaded
        if missing:
            raise ValueError(
                f"Missing Kimi K3.1 MTP non-expert weights: {sorted(missing)}"
            )
        return loaded


EntryClass = [KimiK31ForCausalLMNextN]
