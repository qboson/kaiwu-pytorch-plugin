"""Utility helpers for Q-Diffusion DPLM example workflows."""

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .dplm_builder import build_dplm_qdiffusion, build_qdiffusion

__all__ = [
    "build_dplm_qdiffusion",
    "build_qdiffusion",
]


def __getattr__(name):
    """Load model factories only when requested, not for pure I/O helpers."""
    if name in __all__:
        builder = import_module(".dplm_builder", __name__)
        value = getattr(builder, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
