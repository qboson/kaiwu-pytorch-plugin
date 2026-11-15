"""Optional checks against the real installed Kaiwu SDK.

These skip automatically when the enterprise wheel is absent. They verify the
assumptions the wrapper relies on: the auxiliary spin is appended last, the
return is a +-1 candidate block, the anneal is seed-reproducible, warm starts
are accepted, and decoding recovers a true minimizer.
"""

# Test functions document intent in their summary; pytest fixtures do not
# benefit from Args sections.
# pylint: disable=missing-param-doc,missing-return-doc

from __future__ import annotations

import numpy as np
import pytest
import torch

# exc_type=ImportError also skips when the wheel is present but unusable, for
# example a license or binary problem. Without it, pytest 9.1 turns that case
# into a hard failure instead of a skip.
kaiwu = pytest.importorskip(
    "kaiwu", reason="Kaiwu SDK is not installed", exc_type=ImportError
)

# pylint: disable=wrong-import-position
from ising_ep.kaiwu_solver import AUXILIARY_VARIABLE, AnnealSettings, KaiwuSolver, _sdk_api
from ising_ep.model import IsingEPModel

from tests.conftest import iter_physical_states, qubo_score

pytestmark = pytest.mark.integration

_SETTINGS = AnnealSettings(
    initial_temperature=1.25,
    alpha=0.95,
    cutoff_temperature=0.001,
    iterations_per_t=10,
    candidate_count=6,
)


@pytest.fixture(name="qubo")
def fixture_qubo() -> tuple[torch.Tensor, float]:
    """Return a 1-hidden/1-output EP QUBO, i.e. two physical spins."""
    model = IsingEPModel(n_in=1, hidden_sizes=1, n_classes=1, spins_per_class=1, seed=0)
    with torch.no_grad():
        model.weights[0].zero_()
        model.weights[1].copy_(torch.tensor([[0.3]], dtype=torch.float64))
        model.biases[0].copy_(torch.tensor([0.2], dtype=torch.float64))
        model.biases[1].copy_(torch.tensor([-0.3], dtype=torch.float64))
    return model.qubo(torch.zeros(1), torch.zeros(1), 0.25)


@pytest.fixture(name="licensed_annealer", scope="module")
def fixture_licensed_annealer() -> None:
    """Skip the anneal checks when this Kaiwu build asks for a license first.

    Kaiwu 1.4 prompts for a user id before ``single_process_solve`` returns.
    With stdin closed that prompt becomes ``EOF when reading a line``.

    Raises:
        ValueError: If the optimizer fails for a reason unrelated to the prompt.
    """
    optimizer = kaiwu.classical.SimulatedAnnealingOptimizer(
        initial_temperature=1.0,
        alpha=0.5,
        cutoff_temperature=0.5,
        iterations_per_t=1,
        size_limit=1,
        rand_seed=0,
        process_num=1,
    )
    try:
        optimizer.single_process_solve(
            ising_matrix=np.zeros((2, 2)), init_solution=None, rand_seed=0
        )
    except ValueError as error:
        # A closed stdin turns the license prompt into EOF. Under pytest the
        # same prompt is reported as a captured-stdin ValueError instead.
        message = str(error)
        if "EOF" in message or "stdin" in message:
            pytest.skip("this Kaiwu build asks for a license before annealing")
        raise


def test_ising_conversion_appends_one_trailing_auxiliary_spin() -> None:
    """The documented converters add ``__spin__`` as the final variable."""
    matrix = np.array([[-1.0, 2.0], [0.0, -1.0]])
    to_model, to_ising, _sorter = _sdk_api(kaiwu)
    ising = to_ising(to_model(matrix))
    variables = ising.get_variables()

    assert variables[AUXILIARY_VARIABLE] == 2
    assert np.asarray(ising.get_matrix()).shape == (3, 3)


@pytest.mark.usefixtures("licensed_annealer")
def test_solve_returns_a_spin_candidate_block(qubo: tuple[torch.Tensor, float]) -> None:
    """Every candidate row is +-1 and carries the auxiliary column."""
    candidates = KaiwuSolver(_SETTINGS).solve(qubo[0], seed=11)

    assert candidates.shape == (6, 3)
    assert set(np.unique(candidates).tolist()) <= {-1.0, 1.0}


@pytest.mark.usefixtures("licensed_annealer")
def test_solve_is_reproducible_for_an_identical_seed(qubo: tuple[torch.Tensor, float]) -> None:
    """Paired free/nudged solves rely on seed-level reproducibility."""
    solver = KaiwuSolver(_SETTINGS)

    first = solver.solve(qubo[0], seed=11)
    second = solver.solve(qubo[0], seed=11)

    np.testing.assert_array_equal(first, second)


@pytest.mark.usefixtures("licensed_annealer")
def test_solve_accepts_a_full_solver_row_as_warm_start(qubo: tuple[torch.Tensor, float]) -> None:
    """The nudged phase warm-starts from a complete free-phase solver row."""
    solver = KaiwuSolver(_SETTINGS)
    free = solver.solve(qubo[0], seed=11)

    warmed = solver.solve(qubo[0], seed=11, init_solution=free[0])

    assert warmed.shape == free.shape


@pytest.mark.usefixtures("licensed_annealer")
def test_best_row_is_gauge_fixed_and_minimizes_the_qubo(qubo: tuple[torch.Tensor, float]) -> None:
    """Kaiwu's sort and negative-tail flip recover an exact minimizer."""
    matrix, constant = qubo
    candidates = KaiwuSolver(_SETTINGS).solve(matrix, seed=11)

    assert set(np.unique(candidates[:, -1]).tolist()) == {1.0}

    def energy_of(spins: np.ndarray) -> float:
        return qubo_score(matrix, constant, spins)

    chosen = candidates[0, :-1]
    brute_force = min(energy_of(state) for state in iter_physical_states(2))

    assert energy_of(chosen) == pytest.approx(brute_force, abs=1e-9)
