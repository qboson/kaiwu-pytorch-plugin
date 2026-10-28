"""MAIFS SDK submissions must minimize the original binary QUBO objective."""
from itertools import product
import inspect

import numpy as np
import pytest
import torch

import kaiwu as kw
from kaiwu.torch_plugin.maifs import qubo
from kaiwu.torch_plugin.maifs.plugin import FeatureSelectionWrapper


MATH_CASES = [
    pytest.param(np.zeros((3, 3)), np.array([-3., 1., 2.]), id="linear"),
    pytest.param(np.array([[1., 0.], [2., -1.]]), np.array([-3., 4.]),
                 id="diagonal-and-asymmetric-interaction"),
    pytest.param(np.array([[1., .5, -.25], [1.5, -2., 0.], [.75, -1., .25]]),
                 np.array([.25, -1., 2.]), id="mixed-three-variable"),
]


def binary_states(size):
    return np.array(list(product((0, 1), repeat=size)))


def polynomial(states, coefficients, linear):
    """Independent scalar x^T A x + l^T x, including both off-diagonal entries."""
    size = len(linear)
    return np.array([
        sum(coefficients[i, j] * bits[i] * bits[j]
            for i, j in product(range(size), repeat=2))
        + sum(linear[i] * bits[i] for i in range(size))
        for bits in states
    ])


def both_gauges(states):
    spins = np.column_stack((2 * states - 1, np.ones(len(states), dtype=int)))
    return np.concatenate((spins, -spins))


def sdk_minimum(matrix):
    """Enumerate using the actual SDK energy; no annealing algorithm is emulated."""
    spins = np.array(list(product((-1, 1), repeat=len(matrix))))
    energies = kw.common.hamiltonian(matrix, spins)
    return spins[[int(np.argmin(energies))]]


@pytest.fixture(autouse=True)
def preserve_caller_rng():
    state = np.random.get_state()
    yield
    np.random.set_state(state)


@pytest.fixture
def offline_sa(monkeypatch):
    """Keep the real SDK constructor, solve dispatcher and flip-energy method."""
    optimizer_class = kw.classical.SimulatedAnnealingOptimizer
    calls = []

    def single_process(instance, ising_matrix=None, init_solution=None, rand_seed=None):
        assert type(instance) is optimizer_class
        instance.set_matrix(ising_matrix)
        calls.append(instance)
        return sdk_minimum(instance.matrix)

    # Replace only licensed annealing execution, with legal exact energy solutions.
    monkeypatch.setattr(optimizer_class, "single_process_solve", single_process)
    return calls


@pytest.fixture
def offline_cim(monkeypatch):
    """Keep real SDK precision/split/restore; replace only cloud submission."""
    signature = inspect.signature(kw.cim.CIMOptimizer)
    state = {"plans": [], "inputs": [], "submitted": [], "restored": []}
    original_search = qubo.PrecisionSplitExplorer.search
    original_restore = qubo.PrecisionSplitExplorer.restore_solution

    def search(instance, matrix):
        state["inputs"].append(np.array(matrix, copy=True))
        plan = original_search(instance, matrix)
        state["plans"].append(plan)
        return plan

    def restore(instance, solution, vote=False):
        restored = original_restore(instance, solution, vote)
        state["restored"].append(np.array(restored, copy=True))
        return restored

    class OfflineCIM:
        def __init__(self, **kwargs):
            signature.bind(**kwargs)

        def solve(self, matrix):
            state["submitted"].append(np.array(matrix, copy=True))
            return sdk_minimum(matrix)

    monkeypatch.setattr(kw.cim, "CIMOptimizer", OfflineCIM)
    monkeypatch.setattr(qubo.PrecisionSplitExplorer, "search", search)
    monkeypatch.setattr(qubo.PrecisionSplitExplorer, "restore_solution", restore)
    monkeypatch.setattr(kw.common.CheckpointManager, "save_dir", None)
    return state


def cim_options(path, target=4, precision=4):
    return {"save_dir": path, "cleanup_records": False, "target_precision": target,
            "max_precision": precision, "max_bits": 32}


@pytest.mark.parametrize("coefficients, linear", MATH_CASES)
def test_public_symbolic_conversion_keeps_its_positive_upper_triangular_convention(
    coefficients, linear
):
    states = binary_states(len(linear))
    expected = polynomial(states, coefficients, linear)
    # solve() takes the Hessian-form quadratic coefficient: 1/2 x^T Q x + l^T x.
    matrix = qubo.QuadraticLinearSolver().solve(2 * coefficients, linear)
    assert np.count_nonzero(np.tril(matrix)) == 0
    spins = both_gauges(states)
    positive_energy = -kw.common.hamiltonian(matrix, spins)
    np.testing.assert_allclose(positive_energy - positive_energy[0],
                               np.tile(expected - expected[0], 2), rtol=0, atol=1e-12)
    upper = np.triu(coefficients + coefficients.T, 1)
    np.fill_diagonal(upper, np.diag(coefficients) + linear)
    np.testing.assert_array_equal(
        qubo.QuadraticLinearSolver.qubo_matrix_to_ising_matrix(upper), matrix)


@pytest.mark.parametrize("coefficients, linear", MATH_CASES)
def test_sa_submission_sdk_energy_matches_original_qubo_for_all_states_and_gauges(
    offline_sa, coefficients, linear
):
    states = binary_states(len(linear))
    expected = polynomial(states, coefficients, linear)
    result = qubo.solve_qubo(2 * coefficients, linear, states[0], solver="sa")
    matrix = offline_sa[-1].matrix
    actual = kw.common.hamiltonian(matrix, both_gauges(states))
    np.testing.assert_allclose(actual - actual[0], np.tile(expected - expected[0], 2),
                               rtol=0, atol=1e-12)
    assert polynomial([result], coefficients, linear)[0] == pytest.approx(expected.min())


@pytest.mark.parametrize("coefficients, linear", MATH_CASES)
def test_actual_sdk_flip_delta_matches_submitted_energy_changes(
    offline_sa, coefficients, linear
):
    states = binary_states(len(linear))
    qubo.solve_qubo(2 * coefficients, linear, states[0], solver="sa")
    optimizer = offline_sa[-1]
    np.testing.assert_array_equal(optimizer.matrix, optimizer.matrix.T)
    for spins in both_gauges(states):
        before = kw.common.hamiltonian(optimizer.matrix, spins[None])[0]
        for index in range(len(spins)):
            flipped = spins.copy()
            flipped[index] *= -1
            difference = kw.common.hamiltonian(optimizer.matrix, flipped[None])[0] - before
            assert optimizer._dlt_h_single_flip(spins, index) == pytest.approx(difference)


@pytest.mark.parametrize("coefficients, linear, target, precision, split", [
    (np.zeros((2, 2)), np.array([-2., 4.]), 4, 4, False),
    (np.array([[1., 0.], [2., -1.]]), np.array([-3., 4.]), 4, 4, False),
    (np.zeros((2, 2)), np.array([-2., 8.]), 2, 4, True),
], ids=["linear", "interactions", "real-variable-splitting"])
def test_cim_preprocessing_and_restoration_preserve_qubo_energy_ordering(
    offline_cim, tmp_path, coefficients, linear, target, precision, split
):
    states = binary_states(len(linear))
    expected = polynomial(states, coefficients, linear)
    result = qubo.solve_qubo(2 * coefficients, linear, states[0], solver="kaiwu_cim",
                             **cim_options(tmp_path, target, precision))
    matrix = offline_cim["inputs"][-1]
    actual = kw.common.hamiltonian(matrix, both_gauges(states))
    np.testing.assert_allclose(actual - actual[0], np.tile(expected - expected[0], 2),
                               rtol=0, atol=1e-12)
    plan = offline_cim["plans"][-1]
    submitted = offline_cim["submitted"][-1]
    np.testing.assert_array_equal(submitted, plan.split_matrix)
    assert submitted.dtype.kind in "iu"
    np.testing.assert_array_equal(submitted, submitted.T)
    assert (plan.split_size > len(linear) + 1) == split
    expanded = np.array([kw.preprocess.construct_split_solution(spins, plan.last_var_idx)
                         for spins in both_gauges(states)])
    split_energy = kw.common.hamiltonian(submitted, expanded)
    scale = float(plan.precision_info["multiplier"])
    assert scale > 0 and np.isfinite(scale)
    np.testing.assert_allclose(split_energy - split_energy[0],
                               scale * np.tile(expected - expected[0], 2),
                               rtol=0, atol=1e-12)
    assert offline_cim["restored"][-1].shape == (len(linear) + 1,)
    assert polynomial([result], coefficients, linear)[0] == pytest.approx(expected.min())


@pytest.mark.parametrize("backend", ["local_search", "sa", "kaiwu_cim"])
def test_real_feature_mask_update_selects_lower_loss_and_regularization(
    offline_sa, offline_cim, tmp_path, backend
):
    model = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[1., 0.]]))
    kwargs = cim_options(tmp_path, 8, 8) if backend == "kaiwu_cim" else {}
    selector = FeatureSelectionWrapper(model, 2, lambda_reg=.125, solver=backend,
                                      solver_kwargs=kwargs, min_selected_features=0)
    states = binary_states(2)
    # The true scalar loss is quadratic, so its MAIFS Taylor model is exact here.
    objectives = [.5 * (a - 1) ** 2 + .125 * (a + b) for a, b in states]
    result = selector.update_mask([(torch.eye(2), torch.tensor([[1.], [0.]]))],
                                   torch.nn.MSELoss())
    np.testing.assert_array_equal(result, states[int(np.argmin(objectives))])
    np.testing.assert_array_equal(selector.get_support(), result.astype(bool))


@pytest.mark.parametrize("coefficients, linear", MATH_CASES[:2])
def test_local_search_keeps_correct_existing_results_from_every_initial_state(
    coefficients, linear
):
    states = binary_states(len(linear))
    expected = polynomial(states, coefficients, linear)
    for initial in states:
        result = qubo.solve_qubo(2 * coefficients, linear, initial, solver="local_search")
        assert polynomial([result], coefficients, linear)[0] == pytest.approx(expected.min())
