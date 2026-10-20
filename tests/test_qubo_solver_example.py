# -*- coding: utf-8 -*-
"""Tests for the runnable QUBO solver example."""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

EXAMPLE_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../example/qubo_solver")
)
if EXAMPLE_ROOT not in sys.path:
    sys.path.insert(0, EXAMPLE_ROOT)

from run_qubo_solver import (  # noqa: E402  # pylint: disable=wrong-import-position
    build_max_cut_qubo,
    qubo_objective,
    solve_demo,
)


def test_solve_demo_matches_brute_force() -> None:
    """The bundled local-search backend finds the cycle Max-Cut optimum."""
    result = solve_demo(num_nodes=5)

    assert result["best_state"].shape == (5,)
    assert set(np.unique(result["best_state"])).issubset({0, 1})
    assert result["ising_matrix"].shape == (6, 6)
    assert result["best_value"] <= result["initial_value"] + 1e-9
    assert result["brute_force_value"] <= result["initial_value"] + 1e-9
    assert np.isclose(result["best_value"], result["brute_force_value"])


def test_qubo_objective_matches_cut_penalty() -> None:
    """The QUBO value equals minus the number of cut edges."""
    quadratic, linear = build_max_cut_qubo(
        3,
        ((0, 1), (1, 2), (2, 0)),
    )
    state = np.array([1, 0, 0], dtype=int)

    assert np.isclose(qubo_objective(quadratic, linear, state), -2.0)


def test_build_max_cut_qubo_rejects_invalid_edges() -> None:
    """Self-loops and out-of-range endpoints are rejected early."""
    with pytest.raises(ValueError):
        build_max_cut_qubo(3, ((0, 0),))
    with pytest.raises(ValueError):
        build_max_cut_qubo(3, ((0, 3),))
    with pytest.raises(ValueError):
        build_max_cut_qubo(0, ())


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
