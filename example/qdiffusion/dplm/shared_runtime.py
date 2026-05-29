"""Shared runtime helpers for Q-Diffusion DPLM example workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


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
    """Collects the energy-backend-specific checkpoint payload."""
    if hasattr(generator.energy_model, "energy_rbm"):
        return {"energy_rbm": generator.energy_model.energy_rbm.state_dict()}
    return {
        "energy_bm": generator.energy_model.energy_bm.state_dict(),
        "hidden_init": generator.energy_model.hidden_init.state_dict(),
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
            "state_dict": {
                "energy_encoder": generator.energy_model.encoder.backbone.state_dict(),
                "feature_projector": generator.energy_model.feature_projector.state_dict(),
                **_energy_backend_state(generator),
                "energy_head": generator.energy_head.state_dict(),
                "vocab_proj": generator.vocab_proj.state_dict(),
            },
        },
        checkpoint_path,
    )
    return checkpoint_path


def load_trained_energy_weights(generator, checkpoint_path: str, device: str) -> None:
    """Loads a compact energy-side checkpoint into one generator."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["state_dict"]
    generator.energy_model.encoder.backbone.load_state_dict(state_dict["energy_encoder"])
    generator.energy_model.feature_projector.load_state_dict(
        state_dict["feature_projector"]
    )
    if hasattr(generator.energy_model, "energy_rbm"):
        generator.energy_model.energy_rbm.load_state_dict(state_dict["energy_rbm"])
    else:
        generator.energy_model.energy_bm.load_state_dict(state_dict["energy_bm"])
        generator.energy_model.hidden_init.load_state_dict(state_dict["hidden_init"])
    generator.energy_head.load_state_dict(state_dict["energy_head"])
    generator.vocab_proj.load_state_dict(state_dict["vocab_proj"])
