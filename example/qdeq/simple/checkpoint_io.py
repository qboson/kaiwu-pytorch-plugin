"""Checkpoint file I/O for the simple QDEQ examples."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def load_checkpoint(path: Path | str) -> Any:
    """Loads a checkpoint onto CPU, with guidance when training has not run."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run simple_train_example.py to create a checkpoint."
        )
    return torch.load(path, map_location="cpu")
