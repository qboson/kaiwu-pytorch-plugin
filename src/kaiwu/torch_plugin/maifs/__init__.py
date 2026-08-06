"""MAIFS: Model-Agnostic Ising Feature Selection."""

from __future__ import annotations


def __getattr__(name: str):
    """按需加载公开对象，避免导入包时立即加载可选依赖。"""
    if name == "FeatureSelectionWrapper":
        from .plugin import FeatureSelectionWrapper

        return FeatureSelectionWrapper

    if name == "AVAILABLE_SOLVERS":
        from .qubo import AVAILABLE_SOLVERS

        return AVAILABLE_SOLVERS

    if name == "QuadraticLinearSolver":
        from .qubo import QuadraticLinearSolver

        return QuadraticLinearSolver

    if name == "solve_qubo":
        from .qubo import solve_qubo

        return solve_qubo

    raise AttributeError(f"module 'maifs' has no attribute {name!r}")


__all__ = [
    "FeatureSelectionWrapper",
    "QuadraticLinearSolver",
]
