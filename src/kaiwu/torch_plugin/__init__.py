# -*- coding: utf-8 -*-
"""Kaiwu PyTorch plugin public API.

The module exposes the stable top-level training and generation classes for
end users. Optional heavy modules are loaded lazily through ``__getattr__`` so
legacy imports keep working even when extra dependencies are not installed.
"""

from importlib import import_module
from typing import TYPE_CHECKING, Any

from .dbn import UnsupervisedDBN
from .full_boltzmann_machine import BoltzmannMachine
from .qvae import QVAE
from .restricted_boltzmann_machine import RestrictedBoltzmannMachine

if TYPE_CHECKING:
    from .qdiffusion import QDiffusion, QDiffusionConfig

__version__ = "0.1.0"

__all__ = [
    "RestrictedBoltzmannMachine",
    "BoltzmannMachine",
    "QVAE",
    "UnsupervisedDBN",
    "QDiffusion",
    "QDiffusionConfig",
]


def __getattr__(name: str) -> Any:
    """Lazily loads optional heavy modules without breaking legacy imports.

    Args:
        name: Public attribute name requested from this module.

    Returns:
        The lazily imported public object associated with ``name``.

    Raises:
        AttributeError: If ``name`` is not a supported public export.
    """
    if name in {"QDiffusion", "QDiffusionConfig"}:
        module = import_module(".qdiffusion", __name__)
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
