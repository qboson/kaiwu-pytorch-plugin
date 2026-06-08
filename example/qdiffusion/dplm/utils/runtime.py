"""Runtime helpers for Q-Diffusion DPLM example workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


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
    """Saves a compact checkpoint containing only energy-side weights."""
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / name
    torch.save(
        {
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
        checkpoint_path,
    )
    return checkpoint_path


def load_trained_energy_weights(
    generator, checkpoint_path: str, device: str
) -> dict[str, Any]:
    """Loads a compact energy-side checkpoint into one generator."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["state_dict"]
    # Rebuild exactly the energy-side modules we saved during training so rerun
    # and evaluation use the same scorer weights as the best checkpoint.
    generator.energy_model.encoder.backbone.load_state_dict(state_dict["energy_encoder"])
    generator.energy_model.feature_projector.load_state_dict(
        state_dict["feature_projector"]
    )
    generator.energy_model.energy_bm.load_state_dict(state_dict["energy_bm"])
    return checkpoint.get("metadata", {})
