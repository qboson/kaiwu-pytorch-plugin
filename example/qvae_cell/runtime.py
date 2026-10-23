"""Device selection for the single-cell training entry point."""

import torch


def resolve_device(requested):
    """Keep the requested device, falling back when its GPU backend is unavailable."""
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    if device.type == "mps" and not torch.backends.mps.is_available():
        return torch.device("cpu")
    return device
