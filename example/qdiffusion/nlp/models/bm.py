"""BM energy model for natural-language diffusion candidates."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from kaiwu.torch_plugin import EnergyModel

from .features import ConditionedFeatureEncoder
from .proposal import MDLMBackbone
from .sampler import build_bm_sampler


class VisibleTransform(nn.Module):
    """Preserve continuous visible values and checkpoint compatibility.

    Args:
        num_visible: Number of BM visible variables.
        mode: Visible transform name. The maintained path accepts only
            ``"identity"``.
    """

    def __init__(self, num_visible: int, mode: str = "identity") -> None:
        super().__init__()
        if mode != "identity":
            raise ValueError("The BM-only example supports identity visibles.")
        self.mode = mode
        self.normalizer = nn.LayerNorm(
            num_visible,
            elementwise_affine=False,
        )
        self.scale = nn.Parameter(torch.ones(()), requires_grad=False)

    def forward(self, visible_logits: torch.Tensor) -> torch.Tensor:
        """Return continuous visible conditions accepted by Kaiwu SDK.

        Args:
            visible_logits: Projected visible values with shape ``(B, V)``.

        Returns:
            The unchanged floating-point visible values.
        """

        return visible_logits


class BMEnergyModel(EnergyModel):
    """Score noisy/candidate sequence pairs with a conditioned BM.

    Args:
        encoder: Trainable energy-side text backbone.
        bm_num_visible: Number of projected BM visible variables.
        bm_num_hidden: Number of BM hidden variables.
        sampler: Optional preconstructed hidden-state sampler.
        sampler_type: Hidden-state sampler type. The maintained path uses SA.
        sampler_kwargs: Keyword arguments forwarded to the sampler.
        visible_transform: Visible transform name; must be ``"identity"``.
        pooling_mode: Conditioned sequence pooling strategy.

    Attributes:
        conditioned_encoder: Joint noisy/candidate feature encoder.
        feature_projector: Projection from text features to BM visibles.
        energy_bm: Kaiwu SDK Boltzmann machine created by ``EnergyModel``.
    """

    energy_type = "bm"
    feature_mode = "edlm_pair"

    def __init__(
        self,
        encoder: MDLMBackbone,
        bm_num_visible: int,
        bm_num_hidden: int,
        *,
        sampler: Any | None = None,
        sampler_type: str = "sa",
        sampler_kwargs: dict[str, Any] | None = None,
        visible_transform: str = "identity",
        pooling_mode: str = "attention",
    ) -> None:
        if pooling_mode not in {"mean", "attention"}:
            raise ValueError("pooling_mode must be 'mean' or 'attention'.")
        self.pooling_mode = pooling_mode
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
        self.conditioned_encoder = ConditionedFeatureEncoder(
            encoder,
            pooling_mode=pooling_mode,
        )
        self.feature_projector = nn.Linear(
            encoder.hidden_size,
            bm_num_visible,
        )
        self.visible_transform = VisibleTransform(
            bm_num_visible,
            visible_transform,
        )

    def discretize_visible_state(
        self,
        visible_logits: torch.Tensor,
    ) -> torch.Tensor:
        """Keep projected features continuous when conditioning the BM.

        Args:
            visible_logits: Projected visible values with shape ``(B, V)``.

        Returns:
            Continuous values passed directly to the BM sampler.
        """

        # Kaiwu's conditional sampler accepts real-valued conditions. Avoiding
        # sigmoid or hard thresholding preserves projector precision.
        return self.visible_transform(visible_logits)

    def build_visible_logits(
        self,
        noisy_tokens: torch.Tensor,
        candidate_tokens: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Project noisy/candidate sequence pairs into BM visible values.

        Args:
            noisy_tokens: Noisy diffusion states with shape ``(B, L)``.
            candidate_tokens: Candidate clean tokens with shape ``(B, L)``.
            attention_mask: Valid-token mask with shape ``(B, L)``.

        Returns:
            Continuous BM visible values with shape ``(B, V)``.
        """

        features = self.conditioned_encoder(
            self.encoder,
            noisy_tokens,
            candidate_tokens,
            attention_mask,
        )
        return self.feature_projector(
            features.to(self.feature_projector.weight.dtype)
        )

    def score_conditioned(
        self,
        noisy_tokens: torch.Tensor,
        candidate_tokens: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return one BM energy for every sequence pair.

        Args:
            noisy_tokens: Noisy diffusion states with shape ``(B, L)``.
            candidate_tokens: Candidate clean tokens with shape ``(B, L)``.
            attention_mask: Valid-token mask with shape ``(B, L)``.

        Returns:
            Sequence energies with shape ``(B, 1)``.
        """

        return self.score_visible_logits(
            self.build_visible_logits(
                noisy_tokens,
                candidate_tokens,
                attention_mask,
            )
        )

    def score_candidates_conditioned(
        self,
        noisy_tokens: torch.Tensor,
        candidate_tokens: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Score all candidates in one flattened batch.

        Args:
            noisy_tokens: Noisy states with shape ``(B, L)``.
            candidate_tokens: Candidate pool with shape ``(B, K, L)``.
            attention_mask: Candidate valid-token mask with shape
                ``(B, K, L)``.

        Returns:
            Candidate energies with shape ``(B, K)``.
        """

        batch_size, num_candidates, sequence_length = candidate_tokens.shape
        # Repeat each noisy state K times so the conditioned encoder can process
        # all candidate pairs with one vectorized forward call.
        flat_noisy_tokens = (
            noisy_tokens.unsqueeze(1)
            .expand(-1, num_candidates, -1)
            .reshape(batch_size * num_candidates, sequence_length)
        )
        flat_candidate_tokens = candidate_tokens.reshape(
            batch_size * num_candidates,
            sequence_length,
        )
        flat_attention_mask = attention_mask.reshape(
            batch_size * num_candidates,
            sequence_length,
        )
        energy = self.score_conditioned(
            flat_noisy_tokens,
            flat_candidate_tokens,
            flat_attention_mask,
        )
        return energy.view(batch_size, num_candidates)

    def checkpoint_metadata(self) -> dict[str, Any]:
        """Return architecture metadata needed to rebuild the BM.

        Returns:
            JSON-serializable BM architecture and sampler configuration.
        """

        return {
            "bm_num_visible": self.bm_num_visible,
            "bm_num_hidden": self.bm_num_hidden,
            "sampler_type": self.sampler_type,
            "sampler_kwargs": self.sampler_kwargs,
            "scoring_mode": "sampler",
            "visible_transform": self.visible_transform.mode,
            "feature_mode": self.feature_mode,
            "pooling_mode": self.pooling_mode,
        }

    def compact_state_dict(self) -> dict[str, Any]:
        """Return trainable BM-side modules without proposal weights.

        Returns:
            Nested state dict for the conditioned encoder, projector, visible
            transform, and BM parameters.
        """

        return {
            "conditioned_encoder": self.conditioned_encoder.state_dict(),
            "feature_projector": self.feature_projector.state_dict(),
            "visible_transform": self.visible_transform.state_dict(),
            "energy_bm": self.energy_bm.state_dict(),
        }

    def load_compact_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore BM-side modules from a compact state dict.

        Args:
            state_dict: State produced by :meth:`compact_state_dict` or a
                compatible legacy BM checkpoint.
        """

        self.conditioned_encoder.load_state_dict(
            state_dict["conditioned_encoder"]
        )
        self.feature_projector.load_state_dict(
            state_dict["feature_projector"]
        )
        if "visible_transform" in state_dict:
            self.visible_transform.load_state_dict(
                state_dict["visible_transform"]
            )
        self.energy_bm.load_state_dict(state_dict["energy_bm"])
