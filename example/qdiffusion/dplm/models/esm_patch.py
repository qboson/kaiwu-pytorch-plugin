"""Private ESM runtime patches for the DPLM example stack."""

from __future__ import annotations

from typing import Any, List, Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoTokenizer
from transformers.modeling_outputs import (
    BaseModelOutputWithPoolingAndCrossAttentions,
)
from transformers.models.esm.modeling_esm import (
    EsmAttention,
    EsmEncoder,
    EsmLayer,
    EsmLMHead,
    EsmPreTrainedModel,
    EsmSelfAttention,
)


class _ModifiedEsmSelfAttention(EsmSelfAttention):
    """Custom ESM attention block using scaled-dot-product attention."""

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor, ...], ...]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, ...]:
        del output_attentions
        mixed_query_layer = self.query(hidden_states)
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention and past_key_value is not None:
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

        query_layer = self.transpose_for_scores(mixed_query_layer)
        query_layer = query_layer * self.attention_head_size**-0.5

        if self.is_decoder:
            past_key_value = (key_layer, value_layer)

        if self.position_embedding_type == "rotary":
            query_layer, key_layer = self.rotary_embeddings(query_layer, key_layer)

        if self.position_embedding_type in {"relative_key", "relative_key_query"}:
            raise NotImplementedError
        if head_mask is not None:
            raise NotImplementedError

        query_layer = query_layer.contiguous()
        key_layer = key_layer.contiguous()
        value_layer = value_layer.contiguous()
        context_layer = F.scaled_dot_product_attention(
            query_layer,
            key_layer,
            value_layer,
            attn_mask=attention_mask,
            scale=1.0,
        )

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)
        outputs = (context_layer,)

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs


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
        if self.add_cross_attention:
            self.crossattention = _ModifiedEsmAttention(config)


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
        from transformers.models.esm.modeling_esm import (
            EsmContactPredictionHead,
            EsmEmbeddings,
            EsmPooler,
        )

        self.config = config
        self.embeddings = EsmEmbeddings(config)
        self.encoder = _ModifiedEsmEncoder(config)
        self.pooler = EsmPooler(config) if add_pooling_layer else None
        self.contact_head = EsmContactPredictionHead(
            in_features=config.num_hidden_layers * config.num_attention_heads,
            bias=True,
        )
        self.post_init()

    def get_position_embeddings(self):
        return self.embeddings.position_embeddings

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def resize_position_embeddings(self, new_num_position_embeddings: int):
        raise NotImplementedError(
            "Example DPLM backbones do not support resizing position embeddings."
        )

    def prepare_inputs_for_generation(
        self, *args: Any, **kwargs: Any
    ) -> dict[str, Any]:
        if not args:
            raise ValueError("input_ids must be provided for generation.")
        prepared = dict(kwargs)
        prepared["input_ids"] = args[0]
        if len(args) > 1 and args[1] is not None:
            prepared["past_key_values"] = args[1]
        attention_mask = kwargs.get("attention_mask")
        if attention_mask is not None:
            prepared["attention_mask"] = attention_mask
        return prepared

    def _reorder_cache(
        self,
        past_key_values: tuple[tuple[torch.Tensor, ...], ...],
        beam_idx: torch.Tensor,
    ) -> tuple[tuple[torch.Tensor, ...], ...]:
        return tuple(
            tuple(
                past_state.index_select(0, beam_idx.to(past_state.device))
                for past_state in layer_past
            )
            for layer_past in past_key_values
        )

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[
        Tuple[torch.Tensor, ...], BaseModelOutputWithPoolingAndCrossAttentions
    ]:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        elif input_ids is not None:
            input_shape = input_ids.size()
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        past_key_values_length = (
            past_key_values[0][0].shape[2] if past_key_values is not None else 0
        )

        if attention_mask is None:
            attention_mask = torch.ones(
                ((batch_size, seq_length + past_key_values_length)),
                device=device,
            )

        extended_attention_mask = self.get_extended_attention_mask(
            attention_mask, input_shape
        )

        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(
                encoder_attention_mask
            )
        else:
            encoder_extended_attention_mask = encoder_attention_mask

        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
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

        self.contact_head = None
        self.tokenizer = tokenizer

    def get_position_embeddings(self):
        return self.esm.get_position_embeddings()

    def get_input_embeddings(self):
        return self.esm.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.esm.set_input_embeddings(value)

    def resize_position_embeddings(self, new_num_position_embeddings: int):
        return self.esm.resize_position_embeddings(new_num_position_embeddings)

    def prepare_inputs_for_generation(
        self, *args: Any, **kwargs: Any
    ) -> dict[str, Any]:
        if not args:
            raise ValueError("input_ids must be provided for generation.")
        prepared = dict(kwargs)
        prepared["input_ids"] = args[0]
        if len(args) > 1 and args[1] is not None:
            prepared["past_key_values"] = args[1]
        attention_mask = kwargs.get("attention_mask")
        if attention_mask is not None:
            prepared["attention_mask"] = attention_mask
        return prepared

    def _reorder_cache(
        self,
        past_key_values: tuple[tuple[torch.Tensor, ...], ...],
        beam_idx: torch.Tensor,
    ) -> tuple[tuple[torch.Tensor, ...], ...]:
        return tuple(
            tuple(
                past_state.index_select(0, beam_idx.to(past_state.device))
                for past_state in layer_past
            )
            for layer_past in past_key_values
        )

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs: Any,
    ):
        del (
            attention_mask,
            position_ids,
            head_mask,
            labels,
            output_attentions,
            output_hidden_states,
            return_dict,
            kwargs,
        )

        if input_ids is None:
            raise ValueError("input_ids must be provided for the DPLM example runtime.")

        attention_mask = input_ids.ne(self.pad_id)
        outputs = self.esm(
            input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
        )

        sequence_output = outputs[0]
        logits = self.lm_head(sequence_output)
        return {
            "logits": logits,
            "last_hidden_state": sequence_output,
        }
