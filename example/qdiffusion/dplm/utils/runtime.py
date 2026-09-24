"""Runtime helpers for Q-Diffusion DPLM example workflows."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch

# Version tag for the compact energy-side checkpoint payload. Bump this when
# the payload layout changes and teach ``load_trained_energy_weights`` to
# migrate or reject older formats.
CHECKPOINT_FORMAT = "dplm-energy-compact-v1"

_REQUIRED_STATE_KEYS = ("energy_encoder", "feature_projector", "energy_bm")


def seed_torch(seed: int) -> None:
    """Seeds Torch CPU/CUDA RNGs for reproducible generation steps."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def encode_sequence(
    generator, sequence: str, max_length: int | None = None
) -> torch.Tensor:
    """Tokenizes one protein sequence to ``[1, seq_len]`` on the generator device."""
    if max_length is not None:
        sequence = sequence[:max_length]
    encoded = generator.tokenizer(
        sequence,
        return_tensors="pt",
        add_special_tokens=True,
    )
    return encoded["input_ids"].to(generator.device)


def summarize_trainable_parameters(generator) -> dict[str, int]:
    """Counts total and trainable parameters."""
    total = 0
    trainable = 0
    for parameter in generator.parameters():
        count = parameter.numel()
        total += count
        if parameter.requires_grad:
            trainable += count
    return {"total_parameters": total, "trainable_parameters": trainable}


def _energy_backend_state(generator) -> dict[str, Any]:
    """Collects the BM checkpoint payload."""
    return {
        "energy_bm": generator.energy_model.energy_bm.state_dict(),
    }


def _energy_backend_metadata(generator) -> dict[str, Any]:
    """Collects lightweight metadata required to rebuild the BM backend."""
    return {
        "bm_sampler_type": getattr(generator.energy_model, "sampler_type", None),
        "bm_sampler_kwargs": getattr(generator.energy_model, "sampler_kwargs", {}),
    }


def save_checkpoint(
    output_dir: Path,
    name: str,
    *,
    generator,
    epoch: int,
    metric: float,
) -> Path:
    """Saves a compact checkpoint containing only energy-side weights.

    The payload carries ``checkpoint_format`` and is written atomically so a
    crash mid-epoch never leaves a truncated ``best`` checkpoint behind.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / name
    temporary_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
    torch.save(
        {
            "checkpoint_format": CHECKPOINT_FORMAT,
            "epoch": epoch,
            "metric": metric,
            "metadata": _energy_backend_metadata(generator),
            "state_dict": {
                # Proposal weights are intentionally omitted because the current
                # example treats proposal DPLM as a frozen upstream component.
                "energy_encoder": generator.energy_model.encoder.backbone.state_dict(),
                "feature_projector": generator.energy_model.feature_projector.state_dict(),
                **_energy_backend_state(generator),
            },
        },
        temporary_path,
    )
    os.replace(temporary_path, checkpoint_path)
    return checkpoint_path


def load_trained_energy_weights(
    generator, checkpoint_path: str, device: str
) -> dict[str, Any]:
    """Loads a compact energy-side checkpoint into one generator.

    Raises:
        ValueError: If the file is not a dictionary payload, was not written by
            this checkpoint format, or is missing required state entries.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint payload must be a dictionary")
    checkpoint_format = checkpoint.get("checkpoint_format")
    if checkpoint_format != CHECKPOINT_FORMAT:
        raise ValueError(
            f"unsupported checkpoint format: {checkpoint_format!r} "
            f"(expected {CHECKPOINT_FORMAT!r})"
        )
    state_dict = checkpoint["state_dict"]
    missing = [key for key in _REQUIRED_STATE_KEYS if key not in state_dict]
    if missing:
        raise ValueError(f"checkpoint state_dict is missing entries: {missing}")
    # Rebuild exactly the energy-side modules we saved during training so rerun
    # and evaluation use the same scorer weights as the best checkpoint.
    generator.energy_model.encoder.backbone.load_state_dict(state_dict["energy_encoder"])
    generator.energy_model.feature_projector.load_state_dict(
        state_dict["feature_projector"]
    )
    generator.energy_model.energy_bm.load_state_dict(state_dict["energy_bm"])
    return checkpoint.get("metadata", {})
