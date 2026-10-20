"""Versioned checkpoint I/O for the contextual Nemotron energy model."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from ..common.runtime import atomic_torch_save
from .energy import CHECKPOINT_FORMAT, ContextualEnergyModel, model_from_config


def checkpoint_payload(
    model: ContextualEnergyModel,
    *,
    epoch: int,
    energy_mean: float,
    energy_std: float,
    energy_lambda: float,
    metrics: dict[str, Any],
    run_config: dict[str, Any],
    training_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a versioned checkpoint payload.

    Args:
        model: Trained contextual energy model.
        epoch: Completed training epoch.
        energy_mean: Mean energy measured on the training reference set.
        energy_std: Standard deviation used to normalize energy at inference.
        energy_lambda: Residual-energy weight selected for inference.
        metrics: Metrics recorded for this checkpoint.
        run_config: Immutable training configuration.
        training_state: Optional optimizer, scheduler, and history state.

    Returns:
        Serializable checkpoint payload.

    Raises:
        ValueError: If ``energy_std`` is not positive.
    """

    if energy_std <= 0:
        raise ValueError("energy_std must be positive")
    payload = {
        "checkpoint_format": CHECKPOINT_FORMAT,
        "training_method": "outcome_pairwise_plus_nce_regularizer",
        "epoch": int(epoch),
        "model_config": model.get_config(),
        "model_state": model.compact_state_dict(),
        "energy_normalization": {
            "mean": float(energy_mean),
            "std": float(energy_std),
        },
        "energy_lambda": float(energy_lambda),
        "metrics": metrics,
        "run_config": run_config,
    }
    if training_state is not None:
        payload["training_state"] = training_state
    return payload


def save_checkpoint(path: str | Path, payload: dict[str, Any]) -> None:
    """Atomically save a checkpoint payload.

    Args:
        path: Destination checkpoint path.
        payload: Serialized checkpoint returned by ``checkpoint_payload``.
    """

    atomic_torch_save(Path(path), payload)


def load_checkpoint(
    path: str | Path,
    *,
    device: str | torch.device,
    sampler: Any | None = None,
) -> tuple[ContextualEnergyModel, dict[str, Any]]:
    """Load and validate a trusted local contextual-energy checkpoint.

    Args:
        path: Local checkpoint path.
        device: Device on which to rebuild the energy model.
        sampler: Optional Kaiwu sampler override.

    Returns:
        Rebuilt energy model and its complete checkpoint payload.

    Raises:
        ValueError: If the payload does not match this checkpoint format.
    """

    payload = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("checkpoint payload must be a dictionary")
    if payload.get("checkpoint_format") != CHECKPOINT_FORMAT:
        raise ValueError(
            f"unsupported checkpoint format: {payload.get('checkpoint_format')}"
        )
    required = {
        "model_config",
        "model_state",
        "energy_normalization",
        "energy_lambda",
        "training_method",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"checkpoint is missing fields: {missing}")
    normalization = payload["energy_normalization"]
    if float(normalization["std"]) <= 0:
        raise ValueError("checkpoint energy std must be positive")
    model = model_from_config(payload["model_config"], device=device, sampler=sampler)
    model.load_compact_state_dict(payload["model_state"])
    model.eval()
    return model, payload
