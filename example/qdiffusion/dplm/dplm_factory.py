# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Factory helpers that adapt DPLM backbones into the generic QDiffusion API."""

from __future__ import annotations

from typing import Any

import torch

from kaiwu.torch_plugin import QDiffusion, QDiffusionConfig

from dplm_runtime import (
    DPLMBackbone,
    DPLMEnergyAdapter,
    build_dplm_token_spec,
)


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


def build_dplm_qdiffusion(
    proposal_ckpt: str,
    energy_ckpt: str,
    *,
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
    """Builds one generic ``QDiffusion`` instance from DPLM checkpoints."""
    proposal_model = load_dplm_backbone(
        proposal_ckpt,
        cfg_override=proposal_cfg_override,
        net_override=proposal_net_override,
        from_huggingface=from_huggingface,
    )
    energy_model = load_dplm_backbone(
        energy_ckpt,
        cfg_override=energy_cfg_override,
        net_override=energy_net_override,
        from_huggingface=from_huggingface,
    )
    token_spec = build_dplm_token_spec(proposal_model)
    energy_adapter = DPLMEnergyAdapter(energy_model)
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
