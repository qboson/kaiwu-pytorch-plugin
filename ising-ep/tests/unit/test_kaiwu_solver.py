"""Unit tests for the Kaiwu boundary, driven by a controlled fake SDK.

The real SDK is never imported here. A fake ``kaiwu`` module is injected into
``sys.modules`` so the official call chain, the candidate shape and the
candidate-selection rule can be asserted without a license.
"""

# Test functions document intent in their summary; pytest fixtures and
# parametrised values do not benefit from Args sections.
# pylint: disable=missing-param-doc,missing-return-doc
# The fake SDK deliberately mirrors the real ranking rule, which the solver's
# 1.3.1 fallback also implements.
# pylint: disable=duplicate-code

from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from ising_ep.kaiwu_solver import AUXILIARY_VARIABLE, KaiwuSolver, init_license

from tests.conftest import PRIMARY_ANNEAL


class _FakeQuboModel:
    """Stand-in for ``kaiwu.core.QuboModel`` that only remembers its matrix."""

    def __init__(self, matrix: np.ndarray) -> None:
        self.matrix = np.asarray(matrix, dtype=np.float64)


class _FakeIsingModel:
    """Stand-in for ``kaiwu.core.IsingModel`` with one trailing auxiliary spin."""

    def __init__(
        self,
        qubo_model: _FakeQuboModel,
        *,
        variables: dict | None = None,
        matrix: np.ndarray | None = None,
    ) -> None:
        n = int(qubo_model.matrix.shape[0])
        if matrix is None:
            matrix = np.full((n + 1, n + 1), 0.25, dtype=np.float64)
        self._matrix = np.asarray(matrix, dtype=np.float64)
        if variables is None:
            variables = {f"b[{i}]": i for i in range(n)}
            variables[AUXILIARY_VARIABLE] = n
        self._variables = variables

    def get_matrix(self) -> np.ndarray:
        """Return the Ising coupling matrix including the auxiliary spin."""
        return self._matrix

    def get_variables(self) -> dict:
        """Return the name-to-index map in solver order."""
        return dict(self._variables)

    def get_bias(self) -> float:
        """Return the scalar Ising bias."""
        return 0.0


class _FakeSdk:
    """Records the official call sequence and returns canned candidates."""

    def __init__(
        self,
        candidates: np.ndarray,
        *,
        variables: dict | None = None,
        ising_matrix: np.ndarray | None = None,
    ) -> None:
        self.candidates = np.asarray(candidates)
        self.variables = variables
        self.ising_matrix = ising_matrix
        self.calls: list[str] = []
        self.optimizer_kwargs: dict[str, Any] = {}
        self.solve_kwargs: dict[str, Any] = {}
        self.solve_inits: list[Any] = []
        self.empty_when_initialized = False
        self.qubo_matrix: np.ndarray = np.empty(0)
        self.license_calls: list[dict] = []

    def as_module(self) -> SimpleNamespace:
        """Return an object shaped like the ``kaiwu`` top-level module."""
        sdk = self

        def qubo_matrix_to_qubo_model(qubo_mat: np.ndarray) -> _FakeQuboModel:
            sdk.calls.append("qubo_matrix_to_qubo_model")
            sdk.qubo_matrix = np.asarray(qubo_mat, dtype=np.float64)
            return _FakeQuboModel(qubo_mat)

        def qubo_model_to_ising_model(qubo_model: _FakeQuboModel) -> _FakeIsingModel:
            sdk.calls.append("qubo_model_to_ising_model")
            return _FakeIsingModel(
                qubo_model, variables=sdk.variables, matrix=sdk.ising_matrix
            )

        def get_sorted_solutions(
            matrix: np.ndarray,
            solutions: np.ndarray,
            bias: float = 0.0,
            negtail_ff: bool = True,
            sort_solutions: bool = True,
        ) -> tuple[np.ndarray, np.ndarray]:
            """Mirror ``kaiwu.core.get_sorted_solutions`` for the injected SDK."""
            sdk.calls.append("get_sorted_solutions")
            rows = np.atleast_2d(np.asarray(solutions, dtype=np.float64))
            coupling = np.asarray(matrix, dtype=np.float64)
            hamilton = -np.einsum("ij,ij->i", rows.dot(coupling), rows) + bias
            if sort_solutions:
                order = np.argsort(hamilton)
                rows = rows[order]
                hamilton = hamilton[order]
            if negtail_ff:
                rows = rows * rows[:, [-1]]
            return rows, hamilton

        class _FakeOptimizer:
            """Stand-in for ``SimulatedAnnealingOptimizer``."""

            def __init__(self, **kwargs: Any) -> None:
                """Record the constructor keyword arguments."""
                sdk.calls.append("SimulatedAnnealingOptimizer")
                sdk.optimizer_kwargs = dict(kwargs)

            def single_process_solve(self, **kwargs: Any) -> np.ndarray:
                """Record the solve keyword arguments and return the candidates."""
                sdk.calls.append("single_process_solve")
                sdk.solve_kwargs = dict(kwargs)
                sdk.solve_inits.append(kwargs.get("init_solution"))
                if sdk.empty_when_initialized and kwargs.get("init_solution") is not None:
                    return np.array([])
                return sdk.candidates

        def _init(**kwargs: Any) -> None:
            """Record a license activation call."""
            sdk.license_calls.append(dict(kwargs))

        return SimpleNamespace(
            core=SimpleNamespace(
                qubo_matrix_to_qubo_model=qubo_matrix_to_qubo_model,
                qubo_model_to_ising_model=qubo_model_to_ising_model,
                get_sorted_solutions=get_sorted_solutions,
            ),
            classical=SimpleNamespace(SimulatedAnnealingOptimizer=_FakeOptimizer),
            license=SimpleNamespace(init=_init),
        )

    def as_legacy_module(self) -> SimpleNamespace:
        """Return a Kaiwu 1.3.1 layout: converters live off ``core`` and there is no sorter."""
        modern = self.as_module()
        return SimpleNamespace(
            core=SimpleNamespace(),
            qubo=SimpleNamespace(
                qubo_matrix_to_qubo_model=modern.core.qubo_matrix_to_qubo_model
            ),
            conversion=SimpleNamespace(
                qubo_model_to_ising_model=modern.core.qubo_model_to_ising_model
            ),
            classical=modern.classical,
            license=modern.license,
        )


@pytest.fixture(name="two_variable_qubo")
def fixture_two_variable_qubo() -> torch.Tensor:
    """Return a small upper-triangular 2-variable QUBO matrix."""
    return torch.tensor([[-1.0, 2.0], [0.0, -1.0]], dtype=torch.float64)


def _install(monkeypatch: pytest.MonkeyPatch, sdk: _FakeSdk) -> None:
    monkeypatch.setitem(sys.modules, "kaiwu", sdk.as_module())


def test_solve_calls_official_conversion_chain_in_order(
    monkeypatch: pytest.MonkeyPatch, two_variable_qubo: torch.Tensor
) -> None:
    """QUBO matrix reaches the annealer through the two documented converters."""
    sdk = _FakeSdk(np.array([[1, -1, 1]]))
    _install(monkeypatch, sdk)

    KaiwuSolver().solve(two_variable_qubo, seed=7)

    assert sdk.calls == [
        "qubo_matrix_to_qubo_model",
        "qubo_model_to_ising_model",
        "SimulatedAnnealingOptimizer",
        "single_process_solve",
        "get_sorted_solutions",
    ]
    # The annealer receives the Ising matrix produced by the SDK, not the QUBO.
    np.testing.assert_array_equal(
        sdk.solve_kwargs["ising_matrix"], np.full((3, 3), 0.25, dtype=np.float64)
    )
    np.testing.assert_array_equal(sdk.qubo_matrix, two_variable_qubo.numpy())


def test_solve_passes_init_solution_and_seed_through_unchanged(
    monkeypatch: pytest.MonkeyPatch, two_variable_qubo: torch.Tensor
) -> None:
    """A warm start and its paired seed reach ``single_process_solve`` verbatim."""
    sdk = _FakeSdk(np.array([[1, -1, 1]]))
    _install(monkeypatch, sdk)
    warm = np.array([-1.0, 1.0, -1.0])

    KaiwuSolver().solve(two_variable_qubo, seed=1234, init_solution=warm)

    np.testing.assert_array_equal(sdk.solve_kwargs["init_solution"], warm)
    assert sdk.solve_kwargs["rand_seed"] == 1234


def test_empty_warm_start_falls_back_to_a_cold_start(
    monkeypatch: pytest.MonkeyPatch, two_variable_qubo: torch.Tensor
) -> None:
    """An empty Kaiwu pool is retried from the opposite gauge, then cold."""
    sdk = _FakeSdk(np.array([[1.0, -1.0, 1.0]]))
    sdk.empty_when_initialized = True
    _install(monkeypatch, sdk)

    candidates = KaiwuSolver().solve(
        two_variable_qubo, seed=5, init_solution=np.array([1.0, -1.0, 1.0])
    )

    assert [init is None for init in sdk.solve_inits] == [False, False, True]
    assert candidates.shape == (1, 3)
    assert sdk.calls.count("single_process_solve") == 3


def test_solve_returns_one_row_per_candidate_including_auxiliary(
    monkeypatch: pytest.MonkeyPatch, two_variable_qubo: torch.Tensor
) -> None:
    """A single flat SDK return is reshaped to one candidate row of ``n + 1``."""
    sdk = _FakeSdk(np.array([1, -1, 1]))
    _install(monkeypatch, sdk)

    candidates = KaiwuSolver().solve(two_variable_qubo, seed=0)

    assert candidates.shape == (1, 3)
    assert candidates.dtype == np.float64


def test_anneal_settings_map_to_official_optimizer_kwargs(
    monkeypatch: pytest.MonkeyPatch, two_variable_qubo: torch.Tensor
) -> None:
    """Registered annealing settings are forwarded with the paired seed."""
    sdk = _FakeSdk(np.array([[1, -1, 1]]))
    _install(monkeypatch, sdk)

    KaiwuSolver(PRIMARY_ANNEAL).solve(two_variable_qubo, seed=42)

    assert sdk.optimizer_kwargs == {
        "initial_temperature": 1.25,
        "alpha": 0.95,
        "cutoff_temperature": 0.001,
        "iterations_per_t": 10,
        "size_limit": 8,
        "rand_seed": 42,
        "process_num": 1,
    }


def test_solve_rejects_a_layout_without_a_trailing_auxiliary_spin(
    monkeypatch: pytest.MonkeyPatch, two_variable_qubo: torch.Tensor
) -> None:
    """The one contract check fires when the aux spin is not the last variable."""
    sdk = _FakeSdk(
        np.array([[1, -1, 1]]),
        variables={AUXILIARY_VARIABLE: 0, "b[0]": 1, "b[1]": 2},
    )
    _install(monkeypatch, sdk)

    with pytest.raises(RuntimeError, match="auxiliary"):
        KaiwuSolver().solve(two_variable_qubo, seed=0)


def test_solve_fixes_a_negative_auxiliary_spin(
    monkeypatch: pytest.MonkeyPatch, two_variable_qubo: torch.Tensor
) -> None:
    """The negative-tail flip turns a trailing ``-1`` into ``+1``."""
    sdk = _FakeSdk(np.array([[1.0, -1.0, -1.0]]))
    _install(monkeypatch, sdk)

    candidates = KaiwuSolver().solve(two_variable_qubo, seed=0)

    np.testing.assert_array_equal(candidates, np.array([[-1.0, 1.0, 1.0]]))


def test_solve_returns_the_lowest_ising_objective_first(
    monkeypatch: pytest.MonkeyPatch, two_variable_qubo: torch.Tensor
) -> None:
    """Candidates come back sorted by the Ising objective, best first."""
    coupling = np.array(
        [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float64
    )
    # s^T M s is +2 for [1, 1, 1] (objective -2) and -2 for [1, -1, -1]
    # (objective +2). After the flip the higher row becomes [-1, 1, 1].
    sdk = _FakeSdk(
        np.array([[1.0, -1.0, -1.0], [1.0, 1.0, 1.0]]),
        ising_matrix=coupling,
    )
    _install(monkeypatch, sdk)

    candidates = KaiwuSolver().solve(two_variable_qubo, seed=0)

    np.testing.assert_array_equal(
        candidates, np.array([[1.0, 1.0, 1.0], [-1.0, 1.0, 1.0]])
    )
    assert set(np.unique(candidates[:, -1]).tolist()) == {1.0}


def test_legacy_sdk_layout_selects_the_same_candidate(
    monkeypatch: pytest.MonkeyPatch, two_variable_qubo: torch.Tensor
) -> None:
    """Kaiwu 1.3.1's module layout reaches the local sorter and the same row."""
    coupling = np.array(
        [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float64
    )
    rows = np.array([[1.0, -1.0, -1.0], [1.0, 1.0, 1.0]])
    modern = _FakeSdk(rows, ising_matrix=coupling)
    _install(monkeypatch, modern)
    modern_result = KaiwuSolver().solve(two_variable_qubo, seed=0)

    legacy = _FakeSdk(rows, ising_matrix=coupling)
    monkeypatch.setitem(sys.modules, "kaiwu", legacy.as_legacy_module())
    legacy_result = KaiwuSolver().solve(two_variable_qubo, seed=0)

    np.testing.assert_array_equal(modern_result, legacy_result)
    assert "get_sorted_solutions" not in legacy.calls
    assert legacy.calls[:4] == [
        "qubo_matrix_to_qubo_model",
        "qubo_model_to_ising_model",
        "SimulatedAnnealingOptimizer",
        "single_process_solve",
    ]


def test_init_license_forwards_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    """Credentials are handed to the official license entry point unchanged."""
    sdk = _FakeSdk(np.array([[1, -1, 1]]))
    module = sdk.as_module()
    monkeypatch.setitem(sys.modules, "kaiwu", module)

    init_license("user-1", "code-1")

    assert sdk.license_calls == [{"user_id": "user-1", "sdk_code": "code-1"}]
