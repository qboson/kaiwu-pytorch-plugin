# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Factory helpers for the ``dplm`` example case."""

from __future__ import annotations

from typing import Any

import torch

from kaiwu.torch_plugin import QDiffusion, QDiffusionConfig

try:
    from ..models import (
        BMConditionedEnergyModel,
        DPLMFeatureEncoder,
    )
    from ..models.backbone import DPLMBackbone, build_dplm_token_spec
except ImportError:  # pragma: no cover - direct script-path compatibility
    from models import (
        BMConditionedEnergyModel,
        DPLMFeatureEncoder,
    )
    from models.backbone import DPLMBackbone, build_dplm_token_spec


def load_dplm_backbone(
    model_name_or_path: str,
    *,
    cfg_override: dict[str, Any] | None = None,
    net_override: dict[str, Any] | None = None,
    from_huggingface: bool = True,
) -> DPLMBackbone:
    """Loads one DPLM backbone wrapper for example-only usage."""
    return DPLMBackbone.from_pretrained(
        model_name_or_path,
        cfg_override=cfg_override,
        net_override=net_override,
        from_huggingface=from_huggingface,
    )


def _build_energy_model(
    *,
    energy_backbone: DPLMBackbone,
    bm_num_visible: int,
    bm_num_hidden: int,
    bm_sampler: Any | None,
    bm_sampler_type: str,
    bm_sampler_kwargs: dict[str, Any] | None,
) -> BMConditionedEnergyModel:
    energy_encoder = DPLMFeatureEncoder(energy_backbone)
    return BMConditionedEnergyModel(
        encoder=energy_encoder,
        bm_num_visible=bm_num_visible,
        bm_num_hidden=bm_num_hidden,
        sampler=bm_sampler,
        sampler_type=bm_sampler_type,
        sampler_kwargs=bm_sampler_kwargs,
    )


def build_qdiffusion(
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
    """Builds one generic ``QDiffusion`` instance for the protein example case."""
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
    energy_model = _build_energy_model(
        energy_backbone=energy_backbone,
        bm_num_visible=bm_num_visible,
        bm_num_hidden=bm_num_hidden,
        bm_sampler=bm_sampler,
        bm_sampler_type=bm_sampler_type,
        bm_sampler_kwargs=bm_sampler_kwargs,
    )

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
        token_spec=build_dplm_token_spec(proposal_model),
        config=config,
        dtype=dtype,
        device=device,
        freeze_proposal=freeze_proposal,
    )


def build_dplm_qdiffusion(*args, **kwargs) -> QDiffusion:
    """Backward-compatible alias for the old builder name."""
    return build_qdiffusion(*args, **kwargs)


