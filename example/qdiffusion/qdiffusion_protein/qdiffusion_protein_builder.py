# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Factory helpers for the ``qdiffusion_protein`` example case."""

from __future__ import annotations

from typing import Any

import torch

from kaiwu.torch_plugin import QDiffusion, QDiffusionConfig

try:
    from .models import (
        BMConditionedEnergyModel,
        DPLMFeatureEncoder,
    )
    from .models.backbone import DPLMBackbone, build_dplm_token_spec
except ImportError:  # pragma: no cover - direct script-path compatibility
    from models import (
        BMConditionedEnergyModel,
        DPLMFeatureEncoder,
    )
    from models.backbone import DPLMBackbone, build_dplm_token_spec

# Backbone loading helpers.


def load_dplm_backbone(
    model_name_or_path: str,
    *,
    cfg_override: dict[str, Any] | None = None,
    net_override: dict[str, Any] | None = None,
    from_huggingface: bool = True,
) -> DPLMBackbone:
    """Loads one DPLM backbone wrapper for example-only usage.

    Args:
        model_name_or_path: Hugging Face model id or local checkpoint path.
        cfg_override: Optional wrapper-config overrides.
        net_override: Optional keyword overrides forwarded to the network loader.
        from_huggingface: Whether to treat ``model_name_or_path`` as a Hugging
            Face identifier instead of a local training artifact.

    Returns:
        DPLMBackbone: One configured example-side DPLM backbone wrapper.
    """
    return DPLMBackbone.from_pretrained(
        model_name_or_path,
        cfg_override=cfg_override,
        net_override=net_override,
        from_huggingface=from_huggingface,
    )


# Generic QDiffusion construction helpers.


def _build_energy_components(
    *,
    energy_backbone: DPLMBackbone,
    bm_num_visible: int,
    bm_num_hidden: int,
    bm_sampler: Any | None,
    bm_sampler_type: str,
    bm_sampler_kwargs: dict[str, Any] | None,
) -> tuple[Any, Any]:
    """Builds the BM energy model plus the QDiffusion-facing adapter object."""
    # The example keeps a single energy path: DPLM feature encoding followed by
    # one sampler-backed BM reranker. The BM model already exposes the generic
    # QDiffusion hooks directly, so no extra wrapper layer is needed.
    energy_encoder = DPLMFeatureEncoder(energy_backbone)
    energy_model = BMConditionedEnergyModel(
        encoder=energy_encoder,
        bm_num_visible=bm_num_visible,
        bm_num_hidden=bm_num_hidden,
        sampler=bm_sampler,
        sampler_type=bm_sampler_type,
        sampler_kwargs=bm_sampler_kwargs,
    )
    return energy_model, energy_model


def build_qdiffusion_protein(
    proposal_ckpt: str,
    energy_ckpt: str,
    *,
    bm_num_visible: int = 256,
    bm_num_hidden: int = 128,
    bm_sampler: Any | None = None,
    bm_sampler_type: str = "sa",
    bm_sampler_kwargs: dict[str, Any] | None = None,
    num_candidates: int = 1,
    proposal_temperature: float = 0.0,
    proposal_noise_scale: float = 1.0,
    energy_temperature: float = 1.0,
    disable_resample: bool = False,
    resample_ratio: float = 0.25,
    resample_top_p: float = 0.95,
    freeze_proposal: bool = True,
    proposal_cfg_override: dict[str, Any] | None = None,
    energy_cfg_override: dict[str, Any] | None = None,
    proposal_net_override: dict[str, Any] | None = None,
    energy_net_override: dict[str, Any] | None = None,
    from_huggingface: bool = True,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> QDiffusion:
    """Builds one generic ``QDiffusion`` instance for the protein example case.

    Args:
        proposal_ckpt: Proposal backbone checkpoint or model id.
        energy_ckpt: Energy backbone checkpoint or model id.
        bm_num_visible: Visible-state size for the BM reranker.
        bm_num_hidden: Hidden-state size for the BM reranker.
        bm_sampler: Optional pre-built sampler object used by BM sampling mode.
        bm_sampler_type: Sampler family used when ``bm_sampler`` is omitted.
        bm_sampler_kwargs: Optional keyword arguments used when creating the
            BM sampler internally. For ``"cim"``, these correspond to
            ``CIMOptimizer`` arguments such as ``task_name``, ``wait``,
            ``interval``, ``project_no``, ``task_mode``, and
            ``sample_number``, plus optional ``PrecisionReducer`` controls.
        num_candidates: Number of proposal candidates sampled per decode step.
        proposal_temperature: Temperature used for proposal-side sampling.
        proposal_noise_scale: Gumbel noise scale used during proposal sampling.
        energy_temperature: Temperature used for energy-based reranking.
        disable_resample: Whether to disable repetition-collapse resampling.
        resample_ratio: Frequency threshold that triggers resampling.
        resample_top_p: Top-p cutoff used during resampling.
        freeze_proposal: Whether to freeze proposal-model parameters.
        proposal_cfg_override: Optional config overrides for the proposal wrapper.
        energy_cfg_override: Optional config overrides for the energy wrapper.
        proposal_net_override: Optional network overrides for the proposal loader.
        energy_net_override: Optional network overrides for the energy loader.
        from_huggingface: Whether checkpoints should be loaded from Hugging Face.
        dtype: Floating-point dtype tracked by the resulting ``QDiffusion``.
        device: Optional target device for the resulting ``QDiffusion``.

    Returns:
        QDiffusion: One generic ``QDiffusion`` instance backed by DPLM adapters.
    """
    proposal_model = load_dplm_backbone(
        proposal_ckpt,
        cfg_override=proposal_cfg_override,
        net_override=proposal_net_override,
        from_huggingface=from_huggingface,
    )
    energy_backbone = load_dplm_backbone(
        energy_ckpt,
        cfg_override=energy_cfg_override,
        net_override=energy_net_override,
        from_huggingface=from_huggingface,
    )
    energy_model, energy_adapter = _build_energy_components(
        energy_backbone=energy_backbone,
        bm_num_visible=bm_num_visible,
        bm_num_hidden=bm_num_hidden,
        bm_sampler=bm_sampler,
        bm_sampler_type=bm_sampler_type,
        bm_sampler_kwargs=bm_sampler_kwargs,
    )

    token_spec = build_dplm_token_spec(proposal_model)
    # ``QDiffusionConfig`` only stores generic generation/training knobs; the
    # DPLM- and BM-specific assembly has already been done above.
    config = QDiffusionConfig(
        num_candidates=num_candidates,
        proposal_temperature=proposal_temperature,
        proposal_noise_scale=proposal_noise_scale,
        energy_temperature=energy_temperature,
        disable_resample=disable_resample,
        resample_ratio=resample_ratio,
        resample_top_p=resample_top_p,
    )
    return QDiffusion(
        proposal_model=proposal_model,
        energy_model=energy_model,
        token_spec=token_spec,
        energy_adapter=energy_adapter,
        config=config,
        dtype=dtype,
        device=device,
        freeze_proposal=freeze_proposal,
    )


def build_dplm_qdiffusion(*args, **kwargs) -> QDiffusion:
    """Backward-compatible alias for the old builder name."""
    return build_qdiffusion_protein(*args, **kwargs)
