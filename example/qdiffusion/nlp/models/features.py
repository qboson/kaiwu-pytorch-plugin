"""Conditioned feature encoder shared by the BM energy model."""

from __future__ import annotations

import torch
from torch import nn

from .proposal import MDLMBackbone


class ConditionedFeatureEncoder(nn.Module):
    """Encode a noisy/candidate token pair into one sequence feature.

    Args:
        encoder: Proposal-compatible backbone that exposes conditioned token
            encoding helpers.
        pooling_mode: Sequence pooling strategy, either ``"mean"`` or
            ``"attention"``.

    Attributes:
        input_projection: Learned projection from concatenated noisy and
            candidate embeddings back to the backbone hidden width.
        output_layer: Hidden-width copy of the proposal output layer.
        pool_attention: Optional learned scalar attention scorer.
    """

    def __init__(
        self,
        encoder: MDLMBackbone,
        *,
        pooling_mode: str = "mean",
    ) -> None:
        super().__init__()
        if pooling_mode not in {"mean", "attention"}:
            raise ValueError("pooling_mode must be 'mean' or 'attention'.")
        self.pooling_mode = pooling_mode
        self.input_projection = nn.Linear(
            2 * encoder.hidden_size,
            encoder.hidden_size,
        )
        self.output_layer = encoder.build_conditioned_output_layer()
        self.pool_attention = (
            nn.Linear(encoder.hidden_size, 1, bias=False)
            if pooling_mode == "attention"
            else None
        )

    def forward(
        self,
        encoder: MDLMBackbone,
        noisy_tokens: torch.Tensor,
        candidate_tokens: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return a pooled representation for each sequence pair.

        Args:
            encoder: Backbone used to run conditioned token encoding.
            noisy_tokens: Noisy diffusion state with shape ``(B, L)``.
            candidate_tokens: Candidate clean sequence with shape ``(B, L)``.
            attention_mask: Optional valid-token mask with shape ``(B, L)``.

        Returns:
            Pooled features with shape ``(B, D)``.

        Raises:
            ValueError: If an attention mask has the wrong shape or contains a
                sequence with no valid tokens.
        """

        hidden_states = encoder.encode_conditioned_tokens(
            noisy_tokens,
            candidate_tokens,
            input_projection=self.input_projection,
            output_layer=self.output_layer,
        )
        # Mean pooling matches the lightweight path. Attention pooling lets the
        # energy model learn which token positions matter for sequence quality.
        if self.pool_attention is None:
            return hidden_states.mean(dim=1)

        attention_inputs = hidden_states.to(self.pool_attention.weight.dtype)
        scores = self.pool_attention(attention_inputs).squeeze(-1).float()
        if attention_mask is not None:
            if attention_mask.shape != scores.shape:
                raise ValueError(
                    "attention_mask must match the conditioned sequence shape."
                )
            valid_tokens = attention_mask.bool()
            if not valid_tokens.any(dim=1).all():
                raise ValueError("Every sequence must contain a valid token.")
            scores = scores.masked_fill(~valid_tokens, -torch.inf)
        weights = torch.softmax(scores, dim=1).to(hidden_states.dtype)
        return (hidden_states * weights.unsqueeze(-1)).sum(dim=1)
