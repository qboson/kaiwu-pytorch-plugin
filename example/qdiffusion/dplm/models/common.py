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
    """Pools token states with one padding-aware mean."""
    # All downstream rerankers expect one fixed-size sequence feature, so we
    # collapse token states here while ignoring padding positions.
    mask = attention_mask.to(hidden_states.dtype).unsqueeze(-1)
    denominator = mask.sum(dim=1).clamp_min(1.0)
    return (hidden_states * mask).sum(dim=1) / denominator


def build_conditioned_features(
    encoder: DPLMFeatureEncoder,
    noisy_tokens: torch.Tensor,
    candidate_tokens: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Builds one concatenated noisy/candidate sequence representation."""
    # The energy model scores one (noisy state, candidate state) pair, so we
    # encode both sides separately and concatenate them into one conditioned
    # feature vector before projecting into BM space.
    noisy_hidden = encoder.encode_tokens(noisy_tokens, attention_mask=attention_mask)
    candidate_hidden = encoder.encode_tokens(
        candidate_tokens,
        attention_mask=attention_mask,
    )
    noisy_features = masked_mean_pool(noisy_hidden, attention_mask)
    candidate_features = masked_mean_pool(candidate_hidden, attention_mask)
    return torch.cat([noisy_features, candidate_features], dim=-1)
