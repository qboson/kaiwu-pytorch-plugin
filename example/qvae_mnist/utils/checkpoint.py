"""Checkpoint writing for the MNIST training pipeline."""

from pathlib import Path
from tempfile import NamedTemporaryFile

import torch


def save_checkpoint(state, destination):
    """Replace the destination only after serialization closes successfully."""
    destination = Path(destination)
    temporary_path = None
    try:
        with NamedTemporaryFile(
            dir=destination.parent, prefix=f".{destination.name}.", delete=False
        ) as temporary:
            temporary_path = Path(temporary.name)
            torch.save(state, temporary)
        # Both paths are in the same directory. Close the file before renaming
        # so replacement works on Windows as well.
        temporary_path.replace(destination)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
