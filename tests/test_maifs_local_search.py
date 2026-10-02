"""Independent energy and numerical controls for local Ising search."""

from decimal import Decimal, localcontext

import numpy as np
import pytest

from kaiwu.torch_plugin.maifs.qubo import _solve_ising_local_search


def full_scan_reference(matrix, initial, max_iter):
    """Greedy reference using complete candidate energies, not local fields."""
    spins = np.r_[2 * initial - 1, 1].astype(int)
    weights = np.triu(matrix)
    for _ in range(max_iter):
        current = float(np.sum(weights * spins[:, None] * spins[None, :]))
        best_value, best_state = current, None
        for index in range(len(spins)):
            candidate = spins.copy()
            candidate[index] *= -1
            value = float(np.sum(weights * candidate[:, None] * candidate[None, :]))
            if value < best_value:
                best_value, best_state = value, candidate
        if best_state is None:
            break
        spins = best_state
    return spins.reshape(1, -1)


@pytest.mark.parametrize("size", [1, 2, 4, 8, 16, 32])
@pytest.mark.parametrize("seed", range(20))
def test_integer_and_float_states_match_full_scan(size, seed):
    """Preserve state choices across seeds, asymmetry, diagonals and limits."""
    rng = np.random.default_rng(seed)
    for integer in (True, False):
        matrix = (rng.integers(-10, 11, (size + 1, size + 1)).astype(float)
                  if integer else rng.normal(size=(size + 1, size + 1)))
        initial = rng.integers(0, 2, size=size)
        original_matrix, original_initial = matrix.copy(), initial.copy()
        for limit in (1, 2, 10, 100):
            result = _solve_ising_local_search(matrix, initial, limit)
            expected = full_scan_reference(matrix, initial, limit)
            np.testing.assert_array_equal(result, expected)
        np.testing.assert_array_equal(matrix, original_matrix)
        np.testing.assert_array_equal(initial, original_initial)


def test_search_does_not_materialize_candidate_outer_products(monkeypatch):
    """Candidate evaluation must not build a square tensor per spin."""
    def reject_outer(*args, **kwargs):
        raise AssertionError("full candidate outer product allocated")
    monkeypatch.setattr(np, "outer", reject_outer)
    result = _solve_ising_local_search(np.array([[0., 2.], [2., 0.]]), max_iter=1)
    np.testing.assert_array_equal(result, [[-1, 1]])


def test_diagonal_constants_do_not_hide_an_improving_flip():
    """Large state-independent diagonals must not cancel a small energy gain."""
    matrix = np.array([[1e20, 1.], [1., 1e20]])
    result = _solve_ising_local_search(matrix, max_iter=1)
    np.testing.assert_array_equal(result, [[-1, 1]])


def test_tied_candidates_choose_first_index():
    matrix = np.ones((3, 3)) - np.eye(3)
    np.testing.assert_array_equal(
        _solve_ising_local_search(matrix, max_iter=1), [[-1, 1, 1]],
    )


def test_auxiliary_spin_can_be_the_best_flip():
    matrix = np.array([[0., -3., 2.], [-3., 0., 2.], [2., 2., 0.]])
    np.testing.assert_array_equal(
        _solve_ising_local_search(matrix, max_iter=1), [[1, 1, -1]],
    )


def decimal_energy(matrix, spins):
    """Exactly sum finite binary64 coefficients for a small independent control."""
    with localcontext() as context:
        context.prec = 2048
        return sum(
            (Decimal.from_float(float(matrix[i, j])) * int(spins[i]) * int(spins[j])
             for i in range(len(spins)) for j in range(i + 1, len(spins))),
            Decimal(0),
        )


@pytest.mark.parametrize("scale", [1., 1e8, 1e16])
def test_nearly_cancelled_fields_end_at_a_true_local_minimum(scale):
    matrix = np.array([
        [0., scale, 1., -scale], [0., 0., -scale, 0.],
        [0., 0., 0., scale], [0., 0., 0., 0.],
    ])
    spins = _solve_ising_local_search(matrix, max_iter=100)[0]
    value = decimal_energy(matrix, spins)
    for index in range(len(spins)):
        candidate = spins.copy()
        candidate[index] *= -1
        assert decimal_energy(matrix, candidate) >= value


def test_no_couplings_preserves_initial_state_and_ignores_lower_triangle():
    matrix = np.diag([100., -200., 300.])
    matrix[2, 0] = -1e9
    np.testing.assert_array_equal(
        _solve_ising_local_search(matrix, np.array([0, 1])), [[-1, 1, 1]],
    )


def test_empty_matrix_preserves_empty_solution():
    assert _solve_ising_local_search(np.empty((0, 0))).shape == (1, 0)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_coefficients_are_rejected(value):
    with pytest.raises(ValueError, match="finite"):
        _solve_ising_local_search(np.array([[0., value], [0., 0.]]))
