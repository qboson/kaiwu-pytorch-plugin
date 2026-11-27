"""Concrete BM-guided QDiffusion generator for the NLP example."""

from __future__ import annotations

from typing import Any

import torch

from kaiwu.torch_plugin import QDiffusion, QDiffusionConfig

from ..sampling import BMGuidedSampler
from .bm import BMEnergyModel
from .proposal import MDLMBackbone, build_mdlm_token_spec


class BMTextGenerator(QDiffusion):
    """Combine a frozen text proposal with a conditioned BM energy model.

    Args:
        proposal: Frozen model that predicts clean-token proposal
            distributions.
        energy_backbone: Separate backbone used by the trainable BM scorer.
        bm_num_visible: Number of BM visible variables.
        bm_num_hidden: Number of BM hidden variables.
        bm_sampler: Optional preconstructed SA sampler.
        bm_sampler_kwargs: Keyword arguments used to construct SA.
        pooling_mode: Conditioned feature pooling strategy.
        num_candidates: Number of proposal candidates scored per guided step.
        energy_temperature: Temperature applied to negative candidate energy.
        dtype: QDiffusion computation dtype.
        device: Target device. Defaults to the proposal parameter device.
    """

    def __init__(
        self,
        proposal: MDLMBackbone,
        energy_backbone: MDLMBackbone,
        *,
        bm_num_visible: int = 768,
        bm_num_hidden: int = 256,
        bm_sampler: Any | None = None,
        bm_sampler_kwargs: dict[str, Any] | None = None,
        pooling_mode: str = "attention",
        num_candidates: int = 2,
        energy_temperature: float = 1.0,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str | None = None,
    ) -> None:
        resolved_device = (
            torch.device(device)
            if device is not None
            else next(proposal.parameters()).device
        )
        energy_model = BMEnergyModel(
            encoder=energy_backbone,
            bm_num_visible=bm_num_visible,
            bm_num_hidden=bm_num_hidden,
            sampler=bm_sampler,
            sampler_type="sa",
            sampler_kwargs=bm_sampler_kwargs,
            visible_transform="identity",
            pooling_mode=pooling_mode,
        )
        # QDiffusion owns the generic proposal/energy composition. NLP-specific
        # code only supplies adapters and the token metadata.
        super().__init__(
            proposal_model=proposal,
            energy_model=energy_model,
            token_spec=build_mdlm_token_spec(proposal),
            config=QDiffusionConfig(
                num_candidates=num_candidates,
                energy_temperature=energy_temperature,
                disable_resample=True,
            ),
            dtype=dtype,
            device=resolved_device,
            freeze_proposal=True,
        )
        self.last_sampling_stats: dict[str, Any] = {}
        self._normalize_bm_precision()

    def _normalize_bm_precision(self) -> None:
        """Keep SDK BM parameters in float32 for NumPy/SA conversion.

        Raises:
            TypeError: If the generator was constructed without a BM energy
                model.
        """

        energy_model = self.energy_model
        if not isinstance(energy_model, BMEnergyModel):
            raise TypeError("BMTextGenerator requires a BMEnergyModel.")
        energy_model.feature_projector.to(dtype=torch.float32)
        bm_device = energy_model.energy_bm.linear_bias.device
        torch.nn.Module.to(
            energy_model.energy_bm,
            device=bm_device,
            dtype=torch.float32,
        )
        energy_model.energy_bm.device = bm_device
        energy_model.energy_bm.dtype = torch.float32

    @torch.no_grad()
    def generate(
        self,
        input_tokens: torch.Tensor,
        *,
        max_steps: int,
        importance_start_t: float = 1.0,
        importance_end_t: float = 0.0,
        remask_ratio: float = 0.1,
    ) -> torch.Tensor:
        """Generate tokens with BM candidate selection at guided steps.

        Args:
            input_tokens: Initial token canvas with shape ``(B, L)``.
            max_steps: Number of reverse-diffusion steps.
            importance_start_t: Largest normalized timestep using BM guidance.
            importance_end_t: Smallest normalized timestep using BM guidance.
            remask_ratio: Fraction of low-confidence revealed tokens eligible
                for skeptical remasking.

        Returns:
            Fully denoised token IDs with shape ``(B, L)``.
        """

        sampler = BMGuidedSampler(
            self.proposal_model,
            energy_model=self.energy_model,
            mask_id=self.mask_id,
            num_candidates=self.config.num_candidates,
            energy_temperature=self.config.energy_temperature,
            importance_start_t=importance_start_t,
            importance_end_t=importance_end_t,
            remask_ratio=remask_ratio,
        )
        output = sampler.sample(input_tokens, num_steps=max_steps)
        self.last_sampling_stats = dict(sampler.last_stats)
        return output
