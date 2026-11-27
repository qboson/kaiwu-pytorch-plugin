"""BM-backed conditioned rerankers for DPLM examples."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from transformers.models.esm import modeling_esm
from torch.nn.modules import module as module_hooks

from kaiwu.torch_plugin import EnergyModel

from .backbone import DPLMBackbone
from .common import DPLMFeatureEncoder, masked_mean_pool
from . import esm_patch
from .esm_patch import _EsmForDPLM
from .sampler import build_bm_sampler


# Only the standard ESM module graph has the row-independent inference contract.
# Missing/new Transformers classes and custom/adapted child modules fall back.
_CONTEXT_REUSE_MODULE_TYPES = frozenset(
    module_type for module_type in (
        DPLMFeatureEncoder, DPLMBackbone, _EsmForDPLM,
        esm_patch._ModifiedEsmModel, esm_patch._ModifiedEsmEncoder,
        esm_patch._ModifiedEsmLayer, esm_patch._ModifiedEsmAttention,
        esm_patch._ModifiedEsmSelfAttention,
        nn.Embedding, nn.Dropout, nn.ModuleList, nn.Linear, nn.LayerNorm, nn.Sigmoid,
        *(getattr(modeling_esm, name, None) for name in (
            'EsmEmbeddings', 'EsmSelfOutput', 'EsmIntermediate', 'EsmOutput',
            'EsmContactPredictionHead', 'EsmLMHead', 'RotaryEmbedding',
        )),
    ) if module_type is not None
)


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

    def _encode_noisy_contexts(self, noisy_tokens, candidate_tokens, attention_mask):
        """Reuse repeated contexts only for the standard deterministic encoder.

        Custom backbones, hooks, compiled modules and training/gradient calls retain
        independent rows. Reuse lasts for this call, so changed weights or inputs
        are always re-encoded.
        """
        encoder = self.encoder
        reusable = (
            type(noisy_tokens) is torch.Tensor
            and type(candidate_tokens) is torch.Tensor
            and type(attention_mask) is torch.Tensor
            and noisy_tokens.ndim == candidate_tokens.ndim == attention_mask.ndim == 2
            and noisy_tokens.shape == candidate_tokens.shape == attention_mask.shape
            and noisy_tokens.dtype in (torch.int32, torch.int64)
            and candidate_tokens.dtype in (torch.int32, torch.int64)
            and attention_mask.dtype == torch.bool
            and noisy_tokens.size(0) > 1 and noisy_tokens.size(1) > 0
            and not self.training
            and not torch.is_grad_enabled()
            and not module_hooks._global_forward_hooks
            and not module_hooks._global_forward_pre_hooks
            and type(encoder) is DPLMFeatureEncoder
            and type(encoder.backbone) is DPLMBackbone
            and type(encoder.backbone.net) is _EsmForDPLM
            and 'encode_tokens' not in vars(encoder)
        )
        if reusable:
            reusable = not any(
                type(module) not in _CONTEXT_REUSE_MODULE_TYPES
                or module.training or module._forward_hooks or module._forward_pre_hooks
                or 'forward' in vars(module)
                or module._compiled_call_impl is not None
                for module in encoder.modules()
            )
        if reusable:
            length = noisy_tokens.size(1)
            # Integer token ids plus a boolean mask form an exact, lossless key.
            keys = torch.cat((noisy_tokens.long(), attention_mask.long()), dim=1)
            contexts, inverse = torch.unique(keys, dim=0, return_inverse=True)
            if contexts.size(0) < noisy_tokens.size(0):
                unique_tokens = contexts[:, :length].to(noisy_tokens.dtype)
                unique_mask = contexts[:, length:].bool()
                hidden = encoder.encode_tokens(unique_tokens, attention_mask=unique_mask)
                return hidden[inverse]
        return encoder.encode_tokens(noisy_tokens, attention_mask=attention_mask)

    def build_conditioned_features(
        self,
        noisy_tokens: torch.Tensor,
        candidate_tokens: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        # Keep this helper explicit because it is the semantic bridge between
        # sequence modeling and BM scoring: everything after this point operates
        # on BM-space states rather than protein tokens.
        noisy_hidden = self._encode_noisy_contexts(
            noisy_tokens, candidate_tokens, attention_mask
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
        # The projector turns one concatenated sequence feature into the visible
        # part of the BM state before discretization/sampling steps.
        conditioned_features = self.build_conditioned_features(
            noisy_tokens=noisy_tokens,
            candidate_tokens=candidate_tokens,
            attention_mask=attention_mask,
        )
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
