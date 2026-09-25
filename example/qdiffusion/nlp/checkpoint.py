"""Compact checkpoint helpers for the trainable NLP BM."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def save_bm_checkpoint(
    generator,
    path: Path,
    *,
    epoch: int,
    metric: float,
    extra_metadata: dict[str, Any] | None = None,
    ema_state_dict: dict[str, torch.Tensor] | None = None,
) -> None:
    """Save a compact BM checkpoint.

    The frozen proposal model is intentionally omitted. Trainable parameters
    from the energy-side encoder are retained so a checkpoint can reconstruct
    the complete scoring path.

    Args:
        generator: Object that owns the ``energy_model`` to serialize.
        path: Destination checkpoint path.
        epoch: Completed optimizer step stored for resume bookkeeping.
        metric: Latest training metric associated with the checkpoint.
        extra_metadata: Additional run configuration to persist.
        ema_state_dict: Optional exponential-moving-average parameter state.

    Raises:
        ValueError: If ``generator`` does not contain an energy model.
    """

    energy_model = generator.energy_model
    if energy_model is None:
        raise ValueError("The generator does not contain a BM energy model.")
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "energy_type": getattr(energy_model, "energy_type", "bm"),
        **energy_model.checkpoint_metadata(),
    }
    metadata.update(extra_metadata or {})
    # Only parameters that participated in optimization are copied. This keeps
    # a frozen proposal out of the payload while preserving a trainable
    # energy-side backbone.
    trainable_encoder_state = {
        name: parameter.detach().cpu()
        for name, parameter in energy_model.encoder.named_parameters()
        if parameter.requires_grad
    }
    compact_state = energy_model.compact_state_dict()
    compact_state["energy_encoder_trainable"] = trainable_encoder_state
    payload = {
        "epoch": epoch,
        "metric": metric,
        "metadata": metadata,
        "state_dict": compact_state,
    }
    if ema_state_dict is not None:
        payload["ema_state_dict"] = {
            name: value.detach().cpu()
            for name, value in ema_state_dict.items()
        }
    torch.save(payload, path)


def read_bm_checkpoint(
    path: Path,
    *,
    map_location: torch.device | str = "cpu",
) -> dict[str, Any]:
    """Load and validate a compact NLP BM checkpoint.

    Args:
        path: Checkpoint path produced by :func:`save_bm_checkpoint`.
        map_location: Device mapping forwarded to :func:`torch.load`.

    Returns:
        Validated checkpoint payload.

    Raises:
        ValueError: If required metadata or state-dict fields are missing.
    """

    checkpoint = torch.load(path, map_location=map_location)
    if "metadata" not in checkpoint or "state_dict" not in checkpoint:
        raise ValueError(f"Invalid NLP BM checkpoint: {path}")
    return checkpoint


def load_bm_weights(
    generator,
    checkpoint: dict[str, Any],
    *,
    use_ema: bool = True,
) -> None:
    """Restore BM parameters.

    Args:
        generator: QDiffusion generator that owns the target energy model.
        checkpoint: Payload returned by :func:`read_bm_checkpoint`.
        use_ema: Whether to apply EMA weights after restoring raw weights.

    Raises:
        ValueError: If the generator has no BM model or the checkpoint contains
            a different energy type.
    """

    energy_model = generator.energy_model
    if energy_model is None:
        raise ValueError("The generator does not contain a BM energy model.")
    state_dict = checkpoint["state_dict"]
    checkpoint_type = checkpoint["metadata"].get("energy_type", "bm")
    if checkpoint_type != "bm":
        raise ValueError("The checkpoint does not contain BM energy weights.")
    energy_model.load_compact_state_dict(state_dict)
    # Old and new checkpoints both store the energy-side backbone separately
    # from the compact head state. ``strict=False`` keeps this compatible with
    # checkpoints that froze part of that backbone.
    if state_dict.get("energy_encoder_trainable"):
        energy_model.encoder.load_state_dict(
            state_dict["energy_encoder_trainable"],
            strict=False,
        )
    # EMA is applied last because it is keyed by full module parameter names,
    # whereas ``compact_state_dict`` is grouped by submodule.
    if use_ema and checkpoint.get("ema_state_dict"):
        energy_model.load_state_dict(
            checkpoint["ema_state_dict"],
            strict=False,
        )
