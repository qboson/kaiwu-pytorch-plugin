# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Factory helpers that adapt DPLM backbones into the generic QDiffusion API."""

from __future__ import annotations

from typing import Any

import torch

from kaiwu.torch_plugin import QDiffusion, QDiffusionConfig

from dplm.dplm_modeling import (
    BMConditionedEnergyAdapter,
    BMConditionedEnergyModel,
    DPLMBackbone,
    DPLMFeatureEncoder,
    RBMConditionedEnergyAdapter,
    RBMConditionedEnergyModel,
    build_dplm_token_spec,
)

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


def build_dplm_qdiffusion(
    proposal_ckpt: str,
    energy_ckpt: str,
    *,
    energy_model_type: str = "rbm",
    rbm_num_visible: int = 256,
    rbm_num_hidden: int = 128,
    bm_num_visible: int | None = None,
    bm_num_hidden: int | None = None,
    feature_pooling: str = "mean",
    bm_mean_field_steps: int = 3,
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
    """Builds one generic ``QDiffusion`` instance from DPLM checkpoints.

    Args:
        proposal_ckpt: Proposal backbone checkpoint or model id.
        energy_ckpt: Energy backbone checkpoint or model id.
        energy_model_type: Energy-model backend type. Supported values are
            ``"rbm"`` and ``"bm"``.
        rbm_num_visible: Visible-state size for the RBM reranker.
        rbm_num_hidden: Hidden-state size for the RBM reranker.
        bm_num_visible: Optional visible-state size for the BM reranker. When
            omitted, reuse ``rbm_num_visible`` for convenience.
        bm_num_hidden: Optional hidden-state size for the BM reranker. When
            omitted, reuse ``rbm_num_hidden`` for convenience.
        feature_pooling: Sequence pooling strategy used before RBM scoring.
        bm_mean_field_steps: Number of mean-field hidden-state refinement steps
            used by the BM reranker.
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

    Raises:
        ValueError: If the requested energy backend is unsupported.
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
    energy_encoder = DPLMFeatureEncoder(energy_backbone)
    if energy_model_type == "rbm":
        energy_model = RBMConditionedEnergyModel(
            encoder=energy_encoder,
            rbm_num_visible=rbm_num_visible,
            rbm_num_hidden=rbm_num_hidden,
            feature_pooling=feature_pooling,
        )
        energy_adapter = RBMConditionedEnergyAdapter(energy_model)
    elif energy_model_type == "bm":
        energy_model = BMConditionedEnergyModel(
            encoder=energy_encoder,
            bm_num_visible=bm_num_visible or rbm_num_visible,
            bm_num_hidden=bm_num_hidden or rbm_num_hidden,
            feature_pooling=feature_pooling,
            mean_field_steps=bm_mean_field_steps,
        )
        energy_adapter = BMConditionedEnergyAdapter(energy_model)
    else:
        raise ValueError(
            f"Unsupported DPLM example energy_model_type: {energy_model_type}"
        )

    token_spec = build_dplm_token_spec(proposal_model)
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
