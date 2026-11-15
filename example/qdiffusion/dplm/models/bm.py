"""BM-backed conditioned rerankers for DPLM examples."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from kaiwu.torch_plugin import EnergyModel

from .common import DPLMFeatureEncoder, masked_mean_pool
from .sampler import build_bm_sampler


def _parameter_device(module: nn.Module, fallback: torch.Tensor) -> torch.device:
    """Return the device of ``module``'s first parameter.

    The device is reread on every call. A parameter-free module falls back to
    ``fallback.device`` so an empty encoder does not raise ``StopIteration``.
    """
    parameter = next(module.parameters(), None)
    if parameter is None:
        return fallback.device
    return parameter.device


class BMConditionedEnergyModel(EnergyModel):
    """Conditioned BM reranker backed by one DPLM feature encoder."""

    def __init__(
        self,
        encoder: DPLMFeatureEncoder,
        bm_num_visible: int,
        bm_num_hidden: int,
        sampler: Any | None = None,
        sampler_type: str = "sa",
        sampler_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.sampler_type = sampler_type
        self.sampler_kwargs = dict(sampler_kwargs or {})
        bm_sampler = sampler or build_bm_sampler(
            sampler_type=sampler_type,
            sampler_kwargs=self.sampler_kwargs,
        )
        super().__init__(
            bm_num_visible=bm_num_visible,
            bm_num_hidden=bm_num_hidden,
            sampler=bm_sampler,
        )
        self.encoder = encoder
        self.feature_projector = nn.Linear(2 * encoder.hidden_size, bm_num_visible)

    def build_conditioned_features(
        self,
        noisy_tokens: torch.Tensor,
        candidate_tokens: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Pool noisy and candidate tokens into one BM-space feature.

        Parent ``_apply`` moves the encoder without consulting
        ``energy_bm.device``. Tokens and the mask follow the live encoder
        parameters. Callers are not updated in place, and dtypes stay put.
        """
        encoder_device = _parameter_device(self.encoder, noisy_tokens)
        noisy_tokens = noisy_tokens.to(device=encoder_device)
        candidate_tokens = candidate_tokens.to(device=encoder_device)
        attention_mask = attention_mask.to(device=encoder_device)
        noisy_hidden = self.encoder.encode_tokens(
            noisy_tokens,
            attention_mask=attention_mask,
        )
        candidate_hidden = self.encoder.encode_tokens(
            candidate_tokens,
            attention_mask=attention_mask,
        )
        noisy_features = masked_mean_pool(noisy_hidden, attention_mask)
        candidate_features = masked_mean_pool(candidate_hidden, attention_mask)
        return torch.cat([noisy_features, candidate_features], dim=-1)

    def build_visible_logits(
        self,
        noisy_tokens: torch.Tensor,
        candidate_tokens: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Project pooled features on the projector parameter device.

        The projector can sit on a different device from the encoder, so the
        pooled features are aligned here rather than to a cached device.
        """
        conditioned_features = self.build_conditioned_features(
            noisy_tokens=noisy_tokens,
            candidate_tokens=candidate_tokens,
            attention_mask=attention_mask,
        )
        projector_device = _parameter_device(
            self.feature_projector,
            conditioned_features,
        )
        conditioned_features = conditioned_features.to(device=projector_device)
        return self.feature_projector(conditioned_features)

    def score_conditioned(
        self,
        noisy_tokens: torch.Tensor,
        candidate_tokens: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Scores candidate sequences under the conditioned BM energy model."""
        visible_logits = self.build_visible_logits(
            noisy_tokens=noisy_tokens,
            candidate_tokens=candidate_tokens,
            attention_mask=attention_mask,
        )
        return self.score_visible_logits(visible_logits)
