"""Private ESM runtime patches for the DPLM example stack.

The example replaces upstream ESM attention with a scaled-dot-product
implementation copied from ``transformers.models.esm.modeling_esm`` (pinned
version; the 4.39.2 ESM has no native SDPA). The subclasses below couple to
4.39.2 internals (e.g. ``EsmSelfAttention.transpose_for_scores``), so the
``transformers==4.39.2`` pin in ``example/qdiffusion/requirements.txt`` is a
hard requirement, not a suggestion. Everything the example runtime never
exercises -- HF generation hooks, decoder/cross-attention machinery, contact
heads, position-embedding resizing -- is intentionally absent.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoTokenizer
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.models.esm.modeling_esm import (
    EsmAttention,
    EsmContactPredictionHead,
    EsmEmbeddings,
    EsmEncoder,
    EsmLayer,
    EsmLMHead,
    EsmPooler,
    EsmPreTrainedModel,
    EsmSelfAttention,
)


class _ModifiedEsmSelfAttention(EsmSelfAttention):
    """Custom ESM attention block using scaled-dot-product attention.

    The ``forward`` signature mirrors the pinned 4.39.2 parent exactly: 4.39's
    ``EsmAttention`` passes all seven arguments positionally. The
    encoder/cross-attention, cached-KV, and output-attentions arguments exist
    only for signature compatibility -- the SDPA core never uses them, and
    attention weights are never returned.
    """

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        del (
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )
        if head_mask is not None:
            raise NotImplementedError(
                "The SDPA ESM patch does not support attention head masking."
            )
        if self.position_embedding_type in {"relative_key", "relative_key_query"}:
            raise NotImplementedError(
                "The SDPA ESM patch does not support relative-key positions."
            )

        query_layer = self.transpose_for_scores(self.query(hidden_states))
        query_layer = query_layer * self.attention_head_size**-0.5
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

        if self.position_embedding_type == "rotary":
            query_layer, key_layer = self.rotary_embeddings(query_layer, key_layer)

        context_layer = F.scaled_dot_product_attention(
            query_layer.contiguous(),
            key_layer.contiguous(),
            value_layer.contiguous(),
            attn_mask=attention_mask,
            scale=1.0,
        )

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        return (context_layer.view(new_context_layer_shape),)


class _ModifiedEsmAttention(EsmAttention):
    """ESM attention wrapper that swaps in the modified self-attention."""

    def __init__(self, config):
        super().__init__(config)
        self.self = _ModifiedEsmSelfAttention(config)


class _ModifiedEsmLayer(EsmLayer):
    """ESM transformer layer using the modified attention implementation."""

    def __init__(self, config):
        super().__init__(config)
        self.attention = _ModifiedEsmAttention(config)


class _ModifiedEsmEncoder(EsmEncoder):
    """ESM encoder composed of modified transformer layers."""

    def __init__(self, config):
        super().__init__(config)
        self.layer = nn.ModuleList(
            [_ModifiedEsmLayer(config) for _ in range(config.num_hidden_layers)]
        )


class _ModifiedEsmModel(EsmPreTrainedModel):
    """Modified ESM backbone that accepts fused token embeddings."""

    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config
        self.embeddings = EsmEmbeddings(config)
        self.encoder = _ModifiedEsmEncoder(config)
        self.pooler = EsmPooler(config) if add_pooling_layer else None
        # Kept for strict state_dict compatibility with HF ESM checkpoints and
        # local artifacts saved from the upstream EsmForMaskedLM layout; the
        # example runtime never reads it.
        self.contact_head = EsmContactPredictionHead(
            in_features=config.num_hidden_layers * config.num_attention_heads,
            bias=True,
        )
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> BaseModelOutputWithPooling:
        """Runs the stripped ESM encoder stack.

        Args:
            input_ids: Token ids, used unless ``inputs_embeds`` is given.

            attention_mask: Padding mask for the input.

            inputs_embeds: Pre-embedded inputs used instead of ``input_ids``.

            output_hidden_states: Whether to collect per-layer states.

            return_dict: Whether to return a dataclass instead of a tuple.

        Returns:
            BaseModelOutputWithPooling: Sequence output; the pooled output is
            ``None`` when no pooler was constructed.
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        elif input_ids is not None:
            input_shape = input_ids.size()
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length), device=device)
        extended_attention_mask = self.get_extended_attention_mask(
            attention_mask, input_shape
        )

        embedding_output = self.embeddings(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]
        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
        )


class _EsmForDPLM(EsmPreTrainedModel):
    """Private masked-LM wrapper used by the example-side DPLM runtime."""

    def __init__(self, config, dropout=0.1):
        super().__init__(config)
        tokenizer = AutoTokenizer.from_pretrained(config._name_or_path)
        config.hidden_dropout_prob = dropout

        self.esm = _ModifiedEsmModel(config, add_pooling_layer=False)
        self.lm_head = EsmLMHead(config)
        self.init_weights()

        self.mask_id = tokenizer.mask_token_id
        self.pad_id = tokenizer.pad_token_id
        self.bos_id = tokenizer.cls_token_id
        self.eos_id = tokenizer.eos_token_id
        self.x_id = tokenizer._token_to_id["X"]

        self.tokenizer = tokenizer

    def get_input_embeddings(self):
        return self.esm.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.esm.set_input_embeddings(value)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ):
        """Computes masked-LM logits for the DPLM example runtime.

        Args:
            input_ids: Token ids; the padding mask is derived from ``pad_id``.

            attention_mask: Accepted for call compatibility and ignored.

            inputs_embeds: Optional pre-embedded inputs for the token
                pathway.

        Returns:
            dict: ``"logits"`` over the vocabulary plus the encoder's
            ``"last_hidden_state"``.

        Raises:
            ValueError: If ``input_ids`` is not supplied.
        """
        del attention_mask  # The runtime derives the mask from the pad id below.

        if input_ids is None:
            raise ValueError("input_ids must be provided for the DPLM example runtime.")

        attention_mask = input_ids.ne(self.pad_id)
        outputs = self.esm(
            input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        )

        sequence_output = outputs[0]
        logits = self.lm_head(sequence_output)
        return {
            "logits": logits,
            "last_hidden_state": sequence_output,
        }
