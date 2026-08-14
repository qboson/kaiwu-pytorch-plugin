"""MAIFS: Model-Agnostic Ising Feature Selection."""

from __future__ import annotations

from .plugin import FeatureSelectionWrapper
from .qubo import QuadraticLinearSolver


__all__ = [
    "FeatureSelectionWrapper",
    "QuadraticLinearSolver",
]
