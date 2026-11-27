"""Validation helpers for QDiffusion generation settings."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .qdiffusion import QDiffusionConfig


def validate_generation_config(config: "QDiffusionConfig") -> None:
    """Reject generation settings that would otherwise fail mid-decode.

    Args:
        config: Delivery configuration to validate.

    Raises:
        ValueError: If a threshold is out of range or the decoding strategy is
            not one of the supported four-field forms.
    """
    ratio = float(config.resample_ratio)
    if not 0.0 <= ratio <= 1.0:
        raise ValueError("resample_ratio must be in [0, 1]")

    top_p = float(config.resample_top_p)
    if not 0.0 < top_p <= 1.0:
        raise ValueError("resample_top_p must be in (0, 1]")

    if float(config.proposal_noise_scale) < 0.0:
        raise ValueError("proposal_noise_scale must be non-negative")

    strategy = str(config.decoding_strategy).split("-")
    if len(strategy) != 4:
        raise ValueError("decoding_strategy must contain four '-'-separated fields")
    _, condition, topk_mode, schedule = strategy
    if condition not in {"cond", "uncond"}:
        raise ValueError("decoding_strategy condition must be 'cond' or 'uncond'")
    if topk_mode != "deterministic" and not topk_mode.startswith("stochastic"):
        raise ValueError(
            "decoding_strategy top-k mode must be 'deterministic' "
            "or start with 'stochastic'"
        )
    if schedule not in {"linear", "cosine"}:
        raise ValueError("decoding_strategy schedule must be 'linear' or 'cosine'")
