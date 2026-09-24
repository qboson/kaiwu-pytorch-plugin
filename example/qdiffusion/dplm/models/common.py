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
        """Returns the backbone hidden width."""
        return int(self.backbone.net.config.hidden_size)

    @property
    def tokenizer(self) -> Any:
        """Returns the backbone tokenizer."""
        return self.backbone.tokenizer

    def embed_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """Embeds token ids with the backbone input embedding table.

        Args:
            tokens: Token ids shaped ``[batch, sequence]``.

        Returns:
            torch.Tensor: Embeddings shaped ``[batch, sequence, hidden]``.
        """
        return self.backbone.net.get_input_embeddings()(tokens)

    def encode_tokens(
        self,
        tokens: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encodes token ids into backbone hidden states.

        Args:
            tokens: Token ids shaped ``[batch, sequence]``.

            attention_mask: Optional padding mask.

        Returns:
            torch.Tensor: Last hidden states shaped
            ``[batch, sequence, hidden]``.
        """
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
        """Encodes a fused token/embedding input into hidden states.

        Args:
            input_ids: Token ids providing the token pathway.

            inputs_embeds: Embeddings providing the value pathway.

            attention_mask: Padding mask.

        Returns:
            torch.Tensor: Last hidden states shaped
            ``[batch, sequence, hidden]``.
        """
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
