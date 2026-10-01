"""MAIFS must reject raw non-binary values before integer conversion."""
import importlib
from decimal import Decimal
from itertools import product
from pathlib import Path
import sys
import warnings

import numpy as np
import pytest
import torch
from torch import nn


@pytest.fixture
def maifs_modules():
    """Select real worktree modules after collection's existing Kaiwu reloads."""
    source = Path(__file__).resolve().parents[1] / "src"
    sys.path.insert(0, str(source))
    import kaiwu
    kaiwu.__path__.insert(0, str(source / "kaiwu"))
    qubo = importlib.import_module("kaiwu.torch_plugin.maifs.qubo")
    plugin = importlib.import_module("kaiwu.torch_plugin.maifs.plugin")
    for module, name in ((qubo, "qubo.py"), (plugin, "plugin.py")):
        assert Path(module.__file__).resolve() == (
            source / "kaiwu/torch_plugin/maifs" / name
        ).resolve()
    return qubo, plugin


@pytest.fixture
def offline_sa(maifs_modules, monkeypatch):
    """Keep the real SDK constructor and replace only authorized solve execution."""
    qubo, _ = maifs_modules
    optimizer_class = qubo.kw.classical.SimulatedAnnealingOptimizer
    state = {"result": np.array([[1, -1, 1]]), "calls": []}
    original_rng = np.random.get_state()

    def solve(instance, matrix):
        assert type(instance) is optimizer_class
        assert matrix.shape == (3, 3)
        state["calls"].append(instance)
        return state["result"]

    monkeypatch.setattr(optimizer_class, "solve", solve)
    yield state
    np.random.set_state(original_rng)


INVALID_INITIAL = [
    pytest.param([.9, 1.9], id="positive-fractions"),
    pytest.param([-.2, 1.], id="negative-fraction"),
    pytest.param([np.nan, 1.], id="nan"),
    pytest.param([np.inf, 1.], id="infinity"),
    pytest.param(["0", "1"], id="numeric-strings"),
    pytest.param(np.array(["0", "1"], dtype=object), id="object-strings"),
    pytest.param(np.array([None, 1], dtype=object), id="object-none"),
]


@pytest.mark.parametrize("initial", INVALID_INITIAL)
@pytest.mark.parametrize("backend", ["local_search", "sa"])
def test_public_solver_rejects_original_nonbinary_values(
    maifs_modules, offline_sa, initial, backend
):
    qubo, _ = maifs_modules
    with pytest.raises(ValueError, match="initial_state must contain only binary 0/1"):
        qubo.solve_qubo(np.zeros((2, 2)), np.zeros(2), initial, solver=backend)
    assert offline_sa["calls"] == []


@pytest.mark.parametrize("initial", [[], [1], [[0, 1]], 0])
def test_public_solver_retains_shape_validation(maifs_modules, initial):
    qubo, _ = maifs_modules
    with pytest.raises(ValueError, match="initial_state must match"):
        qubo.solve_qubo(np.zeros((2, 2)), np.zeros(2), initial)


@pytest.mark.parametrize("initial", INVALID_INITIAL[:2] + INVALID_INITIAL[4:5])
@pytest.mark.parametrize("backend", ["_solve_ising_local_search", "_solve_ising_sa"])
def test_backend_entry_points_validate_before_casting(
    maifs_modules, offline_sa, initial, backend
):
    qubo, _ = maifs_modules
    with pytest.raises(ValueError, match="initial_binary must contain only 0/1"):
        getattr(qubo, backend)(np.zeros((3, 3)), initial_binary=initial)
    assert offline_sa["calls"] == []


@pytest.mark.parametrize("dtype", [bool, np.int32, np.float64, np.complex128, object])
@pytest.mark.parametrize("backend", ["local_search", "sa"])
def test_exact_numeric_binary_states_remain_compatible(
    maifs_modules, offline_sa, dtype, backend
):
    qubo, _ = maifs_modules
    initial = np.array([0, 1], dtype=dtype)
    original = initial.copy()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        selected = qubo.solve_qubo(np.zeros((2, 2)), np.zeros(2), initial, solver=backend)
    np.testing.assert_array_equal(selected, [0, 1] if backend == "local_search" else [1, 0])
    np.testing.assert_array_equal(initial, original)
    assert np.issubdtype(selected.dtype, np.integer)


@pytest.mark.parametrize("spins", list(product((-1, 1), repeat=3)))
@pytest.mark.parametrize("dtype", [np.int32, np.float64, np.complex128])
def test_legal_sa_solutions_decode_both_gauges(maifs_modules, offline_sa, spins, dtype):
    qubo, _ = maifs_modules
    offline_sa["result"] = np.array(spins, dtype=dtype)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        selected = qubo.solve_qubo(np.zeros((2, 2)), np.zeros(2), [0., 1.], solver="sa")
    # A bit is active exactly when its spin agrees with the auxiliary spin.
    expected = [int(spin == spins[-1]) for spin in spins[:-1]]
    np.testing.assert_array_equal(selected, expected)
    assert len(offline_sa["calls"]) == 1


@pytest.mark.parametrize("spins", [
    np.array([[True, True, True]]),
    np.array([[-1., 1., -1.]], dtype=object),
])
def test_numeric_object_and_boolean_spins_remain_compatible(maifs_modules, offline_sa, spins):
    qubo, _ = maifs_modules
    offline_sa["result"] = spins
    selected = qubo.solve_qubo(np.zeros((2, 2)), np.zeros(2), [0, 1], solver="sa")
    np.testing.assert_array_equal(selected, [1, 1] if spins.dtype == bool else [1, 0])


def test_numeric_objects_are_validated_by_value(maifs_modules, offline_sa):
    qubo, _ = maifs_modules
    initial = np.array([Decimal(0), Decimal(1)], dtype=object)
    offline_sa["result"] = np.array([[Decimal(-1), Decimal(1), Decimal(-1)]], dtype=object)
    selected = qubo.solve_qubo(np.zeros((2, 2)), np.zeros(2), initial, solver="sa")
    np.testing.assert_array_equal(selected, [1, 0])


INVALID_SPINS = [
    pytest.param([[1.9, -1.9, 1.9]], id="fractional-spins"),
    pytest.param([[1., -1., .999999]], id="near-unit-auxiliary"),
    pytest.param([[0., -1., 1.]], id="zero"),
    pytest.param([[np.nan, -1., 1.]], id="nan"),
    pytest.param([[np.inf, -1., 1.]], id="infinity"),
    pytest.param([["1", "-1", "1"]], id="numeric-strings"),
    pytest.param(np.array([["1", "-1", "1"]], dtype=object), id="object-strings"),
    pytest.param(np.array([[None, -1, 1]], dtype=object), id="object-none"),
]


@pytest.mark.parametrize("spins", INVALID_SPINS)
def test_public_sa_rejects_raw_invalid_spins(maifs_modules, offline_sa, spins):
    qubo, _ = maifs_modules
    offline_sa["result"] = spins
    with pytest.raises(RuntimeError, match="spin solution must contain only -1/1"):
        qubo.solve_qubo(np.zeros((2, 2)), np.zeros(2), [1, 1], solver="sa")
    assert len(offline_sa["calls"]) == 1


@pytest.mark.parametrize("spins", [[], np.empty((0, 3)), [[1, -1]], np.ones((1, 1, 3))])
def test_public_sa_retains_solution_shape_validation(maifs_modules, offline_sa, spins):
    qubo, _ = maifs_modules
    offline_sa["result"] = spins
    with pytest.raises(RuntimeError):
        qubo.solve_qubo(np.zeros((2, 2)), np.zeros(2), [1, 1], solver="sa")


@pytest.mark.parametrize("spins", INVALID_SPINS[:1] + INVALID_SPINS[6:7])
def test_real_wrapper_preserves_mask_when_backend_returns_invalid_values(
    maifs_modules, offline_sa, spins
):
    _, plugin = maifs_modules
    offline_sa["result"] = spins
    selector = plugin.FeatureSelectionWrapper(nn.Linear(2, 1, bias=False), feature_dim=2, solver="sa")
    initial = selector.mask.detach().clone()
    inputs = torch.tensor([[1., 0.], [0., 1.], [1., 1.]])
    with pytest.raises(RuntimeError, match="spin solution must contain only -1/1"):
        selector.update_mask([(inputs, torch.zeros(3, 1))], nn.MSELoss())
    torch.testing.assert_close(selector.mask, initial)
    assert len(offline_sa["calls"]) == 1


def test_real_wrapper_accepts_legal_float_spins(maifs_modules, offline_sa):
    _, plugin = maifs_modules
    offline_sa["result"] = np.array([[-1., 1., -1.]])
    selector = plugin.FeatureSelectionWrapper(nn.Linear(2, 1, bias=False), feature_dim=2, solver="sa")
    inputs = torch.tensor([[1., 0.], [0., 1.], [1., 1.]])
    selected = selector.update_mask([(inputs, torch.zeros(3, 1))], nn.MSELoss())
    np.testing.assert_array_equal(selected, [1, 0])
    np.testing.assert_array_equal(selector.get_support(), [True, False])


def test_zero_variable_qubo_retains_empty_binary_result(maifs_modules):
    qubo, _ = maifs_modules
    selected = qubo.solve_qubo(np.empty((0, 0)), np.empty(0), np.empty(0))
    assert selected.shape == (0,)
    assert np.issubdtype(selected.dtype, np.integer)
