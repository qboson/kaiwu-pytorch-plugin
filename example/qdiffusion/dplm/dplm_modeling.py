# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""DPLM/ESM runtime helpers used by the QDiffusion examples."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

from omegaconf import OmegaConf
import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoConfig, AutoModelForMaskedLM, AutoTokenizer
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

from kaiwu.torch_plugin.qdiffusion import SequenceTokenSpec

# Config structures.


@dataclass
class DPLMNetConfig:
    """Network-level DPLM runtime configuration."""

    arch_type: str = "esm"
    name: str = "esm2_t33_650M_UR50D"
    dropout: float = 0.1
    pretrain: bool = False
    pretrained_model_name_or_path: str = ""


@dataclass
class DPLMLoRAConfig:
    """LoRA configuration preserved for compatibility with old checkpoints."""

    enable: bool = field(default=False)
    lora_rank: int = field(default=16)
    lora_dropout: float = field(default=0.1)
    lora_target_module: str = field(default="")
    modules_to_save: str = field(default="")


@dataclass
class DPLMConfig:
    """Wrapper configuration for the example DPLM backbone."""

    num_diffusion_timesteps: int = field(default=500)
    lora: DPLMLoRAConfig = field(default_factory=DPLMLoRAConfig)
    net: DPLMNetConfig = field(default_factory=DPLMNetConfig)
    gradient_ckpt: bool = field(default=False)
    rdm_couple: bool = field(default=False)


# Config and network loading helpers.


def load_yaml_config(fpath: str):
    """Loads one OmegaConf YAML file and resolves interpolations.

    Args:
        fpath: YAML config path.

    Returns:
        The resolved OmegaConf configuration object.
    """
    cfg = OmegaConf.load(fpath)
    OmegaConf.resolve(cfg)
    return cfg


def get_net_class(dplm_type: str):
    """Returns the supported masked-LM class for one DPLM type.

    Args:
        dplm_type: Runtime type tag stored in the checkpoint config.

    Returns:
        The masked-LM class used for the requested DPLM type.

    Raises:
        ValueError: If the DPLM type is unsupported by these examples.
    """
    if dplm_type != "dplm_esm":
        raise ValueError(f"Unsupported dplm_type for DPLM examples: {dplm_type}")
    return _EsmForDPLM


def get_net(cfg):
    """Builds one underlying masked-LM network from the example config.

    Args:
        cfg: Resolved DPLM example configuration.

    Returns:
        One configured masked-LM network.

    Raises:
        NotImplementedError: If the network architecture type is unsupported.
    """
    if cfg.net.arch_type != "esm":
        raise NotImplementedError(
            f"Unsupported arch_type for DPLM examples: {cfg.net.arch_type}"
        )

    config = AutoConfig.from_pretrained(cfg.net.name)
    net = _EsmForDPLM(config, dropout=cfg.net.dropout)

    if cfg.net.pretrain:
        pretrained_model_name_or_path = cfg.net.pretrained_model_name_or_path
        is_local = os.path.exists(pretrained_model_name_or_path)
        if is_local:
            pretrained_state_dict = torch.load(
                pretrained_model_name_or_path, map_location="cpu"
            )["state_dict"]
            new_pretrained_state_dict = OrderedDict()
            for key, value in pretrained_state_dict.items():
                new_pretrained_state_dict[key[10:]] = value
            net.load_state_dict(new_pretrained_state_dict, strict=True)
        else:
            pretrained_net = AutoModelForMaskedLM.from_pretrained(
                pretrained_model_name_or_path
            )
            net.load_state_dict(pretrained_net.state_dict(), strict=True)
            del pretrained_net

    return net


# Backbone and adapter wrappers.


class DPLMBackbone(nn.Module):
    """Minimal DPLM backbone wrapper used by the examples."""

    _default_cfg = DPLMConfig()

    def __init__(self, cfg: Any | None = None, net: nn.Module | None = None):
        """Initializes the example-side DPLM backbone wrapper.

        Args:
            cfg: Optional wrapper configuration or overrides.
            net: Optional prebuilt masked-LM network.
        """
        super().__init__()
        self._update_cfg(cfg or {})

        self.net = get_net(self.cfg) if net is None else net
        self.tokenizer = self.net.tokenizer

        self.mask_id = self.net.mask_id
        self.pad_id = self.net.pad_id
        self.bos_id = self.net.bos_id
        self.eos_id = self.net.eos_id
        self.x_id = self.net.x_id

        if self.cfg.gradient_ckpt:
            self.net.supports_gradient_checkpointing = True
            self.net.gradient_checkpointing_enable()

    @classmethod
    def from_pretrained(
        cls,
        net_name,
        cfg_override=None,
        net_override=None,
        from_huggingface=True,
    ):
        """Loads one DPLM backbone wrapper from a checkpoint or model id.

        Args:
            net_name: Hugging Face model id or local checkpoint path.
            cfg_override: Optional wrapper-config overrides.
            net_override: Optional keyword overrides forwarded to the network loader.
            from_huggingface: Whether to load through the Hugging Face API.

        Returns:
            One configured example-side DPLM backbone wrapper.
        """
        cfg_override = cfg_override or {}
        net_override = net_override or {}

        if not from_huggingface:
            cfg_path = Path(net_name).parents[1]
            cfg_path = Path(cfg_path, ".hydra", "config.yaml")
            cfg = load_yaml_config(str(cfg_path)).model
            cfg.net.pretrain = False
            cfg.pop("_target_")
            model = cls(cfg)

            pretrained_state_dict = torch.load(
                net_name, map_location=torch.device("cpu")
            )["state_dict"]
            new_pretrained_state_dict = OrderedDict()
            for key, value in pretrained_state_dict.items():
                new_pretrained_state_dict[key[6:]] = value

            model.load_state_dict(new_pretrained_state_dict, strict=False)
            return model

        dplm_type = AutoConfig.from_pretrained(net_name).dplm_type
        net_class = get_net_class(dplm_type)
        net = net_class.from_pretrained(net_name, **net_override)
        return cls(cfg=cfg_override, net=net)

    def _update_cfg(self, cfg):
        """Merges runtime config overrides onto the default DPLM config.

        Args:
            cfg: Runtime config overrides.
        """
        self.cfg = OmegaConf.merge(self._default_cfg, cfg)

    def forward(self, input_ids, return_last_hidden_state=False, **kwargs):
        """Runs the wrapped DPLM network.

        Args:
            input_ids: Input token ids.
            return_last_hidden_state: Whether to return hidden states together
                with logits.
            **kwargs: Unused compatibility keyword arguments.

        Returns:
            Either logits alone or a tuple ``(logits, last_hidden_state)``.
        """
        del kwargs
        outputs = self.net(input_ids=input_ids)
        logits = outputs["logits"]
        if return_last_hidden_state:
            return logits, outputs["last_hidden_state"]
        return logits


def build_dplm_token_spec(backbone: DPLMBackbone) -> SequenceTokenSpec:
    """Builds one generic token spec from a DPLM backbone wrapper.

    Args:
        backbone: Example-side DPLM backbone wrapper.

    Returns:
        One generic token-spec object for ``QDiffusion``.
    """
    return SequenceTokenSpec(
        mask_id=backbone.mask_id,
        pad_id=backbone.pad_id,
        bos_id=backbone.bos_id,
        eos_id=backbone.eos_id,
        x_id=backbone.x_id,
        tokenizer=backbone.tokenizer,
    )


class DPLMEnergyAdapter:
    """Adapter that exposes generic energy hooks over one DPLM backbone."""

    def __init__(self, backbone: DPLMBackbone) -> None:
        """Initializes the example-side DPLM energy adapter.

        Args:
            backbone: Wrapped DPLM backbone used for embedding and encoding.
        """
        self.backbone = backbone

    @property
    def hidden_size(self) -> int:
        """Returns the hidden size of the wrapped backbone."""
        return int(self.backbone.net.config.hidden_size)

    def embed_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """Embeds discrete token ids using the wrapped DPLM network."""
        return self.backbone.net.get_input_embeddings()(tokens)

    def encode_conditioned(
        self,
        input_ids: torch.Tensor,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Encodes fused embeddings and returns token-level hidden states."""
        outputs = self.backbone.net(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        )
        return outputs["last_hidden_state"]


# Private ESM implementation details.


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
        """Initializes the modified ESM attention block.

        Args:
            config: Hugging Face ESM configuration object.
        """
        super().__init__(config)
        self.self = _ModifiedEsmSelfAttention(config)


class _ModifiedEsmLayer(EsmLayer):
    """ESM transformer layer using the modified attention implementation."""

    def __init__(self, config):
        """Initializes the modified ESM transformer layer.

        Args:
            config: Hugging Face ESM configuration object.
        """
        super().__init__(config)
        self.attention = _ModifiedEsmAttention(config)
        if self.add_cross_attention:
            self.crossattention = _ModifiedEsmAttention(config)


class _ModifiedEsmEncoder(EsmEncoder):
    """ESM encoder composed of modified transformer layers."""

    def __init__(self, config):
        """Initializes the modified ESM encoder.

        Args:
            config: Hugging Face ESM configuration object.
        """
        super().__init__(config)
        self.layer = nn.ModuleList(
            [_ModifiedEsmLayer(config) for _ in range(config.num_hidden_layers)]
        )


class _ModifiedEsmModel(EsmPreTrainedModel):
    """Modified ESM backbone that accepts fused token embeddings."""

    def __init__(self, config, add_pooling_layer=True):
        """Initializes the modified ESM model.

        Args:
            config: Hugging Face ESM configuration object.
            add_pooling_layer: Whether to include the optional pooler.
        """
        super().__init__(config)
        from transformers.models.esm.modeling_esm import (
            EsmEmbeddings,
            EsmContactPredictionHead,
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
        """Returns the positional embedding module required by transformers."""
        return self.embeddings.position_embeddings

    def get_input_embeddings(self):
        """Returns the token embedding layer used by the modified ESM model."""
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        """Replaces the token embedding layer on the modified ESM model.

        Args:
            value: Replacement embedding module.
        """
        self.embeddings.word_embeddings = value

    def resize_position_embeddings(self, new_num_position_embeddings: int):
        """Raises because the example ESM layout stays fixed.

        Args:
            new_num_position_embeddings: Requested position-embedding size.

        Raises:
            NotImplementedError: Always, because example ESM position embeddings
                are intentionally fixed.
        """
        raise NotImplementedError(
            "Example DPLM backbones do not support resizing position embeddings."
        )

    def prepare_inputs_for_generation(
        self, *args: Any, **kwargs: Any
    ) -> dict[str, Any]:
        """Builds a minimal generation-compatible input dictionary.

        Args:
            *args: Positional generation arguments, with ``input_ids`` first.
            **kwargs: Keyword generation arguments such as ``attention_mask``.

        Returns:
            A minimal generation input dictionary.

        Raises:
            ValueError: If ``input_ids`` is missing.
        """
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
        """Reorders cached states during beam-style generation helpers.

        Args:
            past_key_values: Cached attention states.
            beam_idx: Beam-selection indices.

        Returns:
            Reordered cached attention states.
        """
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
        """Runs the modified ESM backbone with optional fused embeddings.

        Args:
            input_ids: Optional token ids.
            attention_mask: Optional attention mask.
            position_ids: Optional position ids.
            head_mask: Optional attention-head mask.
            inputs_embeds: Optional precomputed input embeddings.
            encoder_hidden_states: Optional encoder-side hidden states.
            encoder_attention_mask: Optional encoder-side attention mask.
            past_key_values: Optional cached attention states.
            use_cache: Whether to return updated cache entries.
            output_attentions: Whether to return attention tensors.
            output_hidden_states: Whether to return hidden states.
            return_dict: Whether to return Hugging Face model-output objects.

        Returns:
            Either a tuple or a Hugging Face model-output object containing
            token-level hidden states.

        Raises:
            ValueError: If both ``input_ids`` and ``inputs_embeds`` are missing.
        """
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
        """Initializes the example-side masked-LM wrapper.

        Args:
            config: Hugging Face ESM configuration object.
            dropout: Hidden-dropout override applied before model creation.
        """
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
        """Delegates positional embedding access to the wrapped ESM."""
        return self.esm.get_position_embeddings()

    def get_input_embeddings(self):
        """Delegates token embedding access to the wrapped ESM."""
        return self.esm.get_input_embeddings()

    def set_input_embeddings(self, value):
        """Delegates token embedding replacement to the wrapped ESM.

        Args:
            value: Replacement embedding module.
        """
        self.esm.set_input_embeddings(value)

    def resize_position_embeddings(self, new_num_position_embeddings: int):
        """Delegates the unsupported resize operation to the wrapped ESM.

        Args:
            new_num_position_embeddings: Requested position-embedding size.

        Returns:
            The wrapped ESM resize result.
        """
        return self.esm.resize_position_embeddings(new_num_position_embeddings)

    def prepare_inputs_for_generation(
        self, *args: Any, **kwargs: Any
    ) -> dict[str, Any]:
        """Builds the minimal generation input payload expected by transformers.

        Args:
            *args: Positional generation arguments, with ``input_ids`` first.
            **kwargs: Keyword generation arguments such as ``attention_mask``.

        Returns:
            A minimal generation input dictionary.

        Raises:
            ValueError: If ``input_ids`` is missing.
        """
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
        """Reorders cached states during beam-style generation helpers.

        Args:
            past_key_values: Cached attention states.
            beam_idx: Beam-selection indices.

        Returns:
            Reordered cached attention states.
        """
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
        """Runs the private masked-LM backbone and returns logits plus hidden states.

        Args:
            input_ids: Optional token ids.
            attention_mask: Optional attention mask.
            position_ids: Optional position ids.
            head_mask: Optional attention-head mask.
            inputs_embeds: Optional precomputed input embeddings.
            encoder_hidden_states: Optional encoder-side hidden states.
            encoder_attention_mask: Optional encoder-side attention mask.
            labels: Optional masked-LM labels.
            output_attentions: Whether to return attention tensors.
            output_hidden_states: Whether to return hidden states.
            return_dict: Whether to return Hugging Face model-output objects.
            **kwargs: Unused compatibility keyword arguments.

        Returns:
            A dictionary containing logits and token-level hidden states.

        Raises:
            ValueError: If ``input_ids`` is missing.
        """
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
