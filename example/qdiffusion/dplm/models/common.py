"""Shared feature-encoding helpers for DPLM-conditioned energy models."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .backbone import DPLMBackbone


class DPLMFeatureEncoder(nn.Module):
    """Sequence-feature encoder backed by one DPLM backbone."""

    def __init__(self, backbone: DPLMBackbone) -> None:
        super().__init__()
        self.backbone = backbone

    @property
    def hidden_size(self) -> int:
        return int(self.backbone.net.config.hidden_size)

    @property
    def tokenizer(self) -> Any:
        return self.backbone.tokenizer

    def embed_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.backbone.net.get_input_embeddings()(tokens)

    def encode_tokens(
        self,
        tokens: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # This is the plain sequence-encoding path used before any BM-specific
        # projection happens.
        outputs = self.backbone.net(input_ids=tokens, attention_mask=attention_mask)
        return outputs["last_hidden_state"]

    def encode_conditioned(
        self,
        input_ids: torch.Tensor,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        outputs = self.backbone.net(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        )
        return outputs["last_hidden_state"]


def masked_mean_pool(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Pool valid tokens, accumulating low-precision states in float32.

    Zero-weight padding is excluded even if its hidden state is nonfinite.
    Active nonfinite states still propagate, and the result keeps the input dtype.
    """
    # All downstream rerankers expect one fixed-size sequence feature, so we
    # collapse token states here while ignoring padding positions.
    accumulation_dtype = (torch.float32 if hidden_states.dtype in
                          (torch.float16, torch.bfloat16) else hidden_states.dtype)
    mask = attention_mask.to(accumulation_dtype).unsqueeze(-1)
    states = hidden_states.to(accumulation_dtype)
    states = states.masked_fill((mask == 0) & ~torch.isfinite(states), 0)
    denominator = mask.sum(dim=1).clamp_min(1.0)
    return ((states * mask).sum(dim=1) / denominator).to(hidden_states.dtype)
