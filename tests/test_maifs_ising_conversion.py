"""QUBO conversion parity, independent energies and vectorization regressions."""
import itertools

import numpy as np
import pytest

from kaiwu.torch_plugin.maifs import qubo


def legacy(matrix):
    """Independent copy of the original ordered scalar conversion."""
    matrix = np.asarray(matrix, dtype=float)
    n = matrix.shape[0]
    result = np.zeros((n + 1, n + 1))
    for row in range(n):
        result[row, n] += 0.5 * float(matrix[row, row])
        for col in range(row + 1, n):
            pair = 0.25 * float(matrix[row, col])
            result[row, col] += pair
            result[row, n] += pair
            result[col, n] += pair
    return result


@pytest.mark.parametrize("seed", range(10))
@pytest.mark.parametrize("size", [0, 1, 2, 7, 64])
@pytest.mark.parametrize("kind", ["integer", "float", "scaled"])
def test_exact_ordered_conversion(seed, size, kind):
    rng = np.random.default_rng(seed)
    if kind == "integer":
        matrix = rng.integers(-100, 101, size=(size, size))
    elif kind == "float":
        matrix = rng.normal(size=(size, size))
    else:
        matrix = rng.normal(size=(size, size)) * np.exp2(rng.integers(-500, 501, size=(size, size)))
    saved = matrix.copy()
    actual = qubo.QuadraticLinearSolver.qubo_matrix_to_ising_matrix(matrix)
    reference = legacy(matrix)
    np.testing.assert_array_equal(actual, reference)
    np.testing.assert_array_equal(np.signbit(actual), np.signbit(reference))
    np.testing.assert_array_equal(matrix, saved)
    assert actual.dtype == np.float64
    assert not np.shares_memory(actual, matrix)


@pytest.mark.parametrize("layout", ["transpose", "slice", "fortran", "readonly", "negative_stride"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int64])
def test_input_layout_and_dtype(layout, dtype):
    matrix = np.arange(64, dtype=dtype).reshape(8, 8) - 20
    if layout == "transpose":
        matrix = matrix.T
    elif layout == "slice":
        matrix = matrix[::2, ::2]
    elif layout == "fortran":
        matrix = np.asfortranarray(matrix)
    elif layout == "readonly":
        matrix.flags.writeable = False
    else:
        matrix = matrix[::-1, ::-1]
    np.testing.assert_array_equal(qubo.QuadraticLinearSolver.qubo_matrix_to_ising_matrix(matrix), legacy(matrix))


@pytest.mark.parametrize("size", [0, 1, 2, 5])
@pytest.mark.parametrize("seed", [7, 17, 27])
def test_independent_qubo_energy_difference_and_gauges(size, seed):
    matrix = np.random.default_rng(seed).normal(size=(size, size))
    converted = qubo.QuadraticLinearSolver.qubo_matrix_to_ising_matrix(matrix)
    reference = np.ones(size + 1) @ converted @ np.ones(size + 1)
    base = np.ones(size) @ np.triu(matrix) @ np.ones(size)
    for bits in itertools.product((0, 1), repeat=size):
        binary = np.asarray(bits)
        expected = binary @ np.triu(matrix) @ binary - base
        for gauge in (-1, 1):
            spins = np.r_[2 * binary - 1, 1] * gauge
            np.testing.assert_allclose(spins @ converted @ spins - reference, expected, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("size", [0, 1, 3, 6])
def test_solve_nonsymmetric_quadratic_and_linear_terms(size, seed):
    rng = np.random.default_rng(seed)
    quadratic = rng.normal(size=(size, size))
    linear = rng.normal(size=size)
    symmetric = 0.5 * (quadratic + quadratic.T)
    upper = np.triu(symmetric, 1)
    np.fill_diagonal(upper, linear + 0.5 * np.diag(symmetric))
    actual = qubo.QuadraticLinearSolver().solve(quadratic, linear)
    np.testing.assert_array_equal(actual, legacy(upper))
    zero = np.r_[np.full(size, -1), 1]
    for bits in itertools.product((0, 1), repeat=size):
        x = np.asarray(bits)
        spins = np.r_[2*x-1, 1]
        expected = 0.5 * x @ quadratic @ x + linear @ x
        np.testing.assert_allclose(spins @ actual @ spins - zero @ actual @ zero, expected, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("scenario", ["cancellation", "overflow", "subnormal", "signed_zero"])
def test_numerical_boundaries_keep_scalar_order_and_warning_behavior(scenario):
    matrix = np.zeros((6, 6))
    if scenario == "cancellation":
        matrix[0, 3], matrix[1, 3], matrix[2, 3] = 4e20, 4.0, -4e20
        matrix[3, 3], matrix[3, 4], matrix[3, 5] = 6.0, 8.0, -12.0
    elif scenario == "overflow":
        matrix[:3, 3:] = np.finfo(float).max
        matrix[3, 4:] = -np.finfo(float).max
        matrix[3, 3] = np.finfo(float).max
    elif scenario == "subnormal":
        matrix[:] = -np.nextafter(0., 1.)
    else:
        matrix[:] = -0.0
    with np.errstate(over="ignore", invalid="ignore"):
        expected = legacy(matrix)
        actual = qubo.QuadraticLinearSolver.qubo_matrix_to_ising_matrix(matrix)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(np.signbit(actual), np.signbit(expected))
    if scenario == "cancellation":
        assert actual[3, 6] == 2.0


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("position", [(0, 0), (0, 1), (1, 0)])
def test_nonfinite_values_rejected_even_below_diagonal(value, position):
    matrix = np.eye(2)
    matrix[position] = value
    with pytest.raises(ValueError, match="finite"):
        qubo.QuadraticLinearSolver.qubo_matrix_to_ising_matrix(matrix)


@pytest.mark.parametrize("matrix", [np.ones(3), np.ones((2, 3)), np.ones((2, 2, 1))])
def test_invalid_shape_rejected(matrix):
    with pytest.raises(ValueError, match="square"):
        qubo.QuadraticLinearSolver.qubo_matrix_to_ising_matrix(matrix)


def test_ordered_cumsum_path_used(monkeypatch):
    calls = []
    original = np.cumsum
    def record(array, *args, **kwargs):
        calls.append((kwargs.get("axis"), kwargs.get("out") is not None))
        return original(array, *args, **kwargs)
    monkeypatch.setattr(qubo.np, "cumsum", record)
    matrix = np.arange(16).reshape(4, 4)
    np.testing.assert_array_equal(qubo.QuadraticLinearSolver.qubo_matrix_to_ising_matrix(matrix), legacy(matrix))
    assert calls == [(0, True), (1, True)]


@pytest.mark.parametrize("operation", [legacy, qubo.QuadraticLinearSolver.qubo_matrix_to_ising_matrix])
def test_caller_overflow_policy_honored(operation):
    matrix = np.full((8, 8), np.finfo(float).max)
    with np.errstate(over="raise"):
        with pytest.raises(FloatingPointError):
            operation(matrix)


@pytest.mark.parametrize("operation", [legacy, qubo.QuadraticLinearSolver.qubo_matrix_to_ising_matrix])
def test_scalar_multiplication_underflow_is_silent(operation):
    matrix = np.full((2, 2), -np.nextafter(0., 1.))
    with np.errstate(under="raise"):
        np.testing.assert_array_equal(operation(matrix), np.zeros((3, 3)))


@pytest.mark.parametrize("size", [1, 3, 7])
@pytest.mark.parametrize("seed", [7, 17, 27])
@pytest.mark.parametrize("max_iter", [1, 10])
def test_local_solver_result_unchanged(monkeypatch, size, seed, max_iter):
    rng = np.random.default_rng(seed)
    quadratic = rng.normal(size=(size, size))
    linear = rng.normal(size=size)
    initial = rng.integers(0, 2, size=size)
    actual = qubo.solve_qubo(quadratic, linear, initial, max_iter=max_iter)
    monkeypatch.setattr(qubo.QuadraticLinearSolver, "qubo_matrix_to_ising_matrix", staticmethod(legacy))
    expected = qubo.solve_qubo(quadratic, linear, initial, max_iter=max_iter)
    np.testing.assert_array_equal(actual, expected)
