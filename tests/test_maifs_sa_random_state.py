"""MAIFS must seed the SDK optimizer without reseeding caller training streams."""
from itertools import product

import numpy as np
import pytest
import torch

from kaiwu.classical import SimulatedAnnealingOptimizer
from kaiwu.torch_plugin.maifs import qubo
from kaiwu.torch_plugin.maifs.plugin import FeatureSelectionWrapper


@pytest.fixture
def offline_sa(monkeypatch):
    """Keep real SDK construction/dispatch; replace license-gated SA execution."""
    calls = []
    state = {"result": np.array([[1, -1, 1]]), "error": None, "calls": calls}
    original_rng = np.random.get_state()

    def single_process(instance, ising_matrix=None, init_solution=None, rand_seed=None):
        assert type(instance) is SimulatedAnnealingOptimizer
        assert ising_matrix.shape == (3, 3)
        # Observe the SDK-dispatched seed with a local generator, without emulating
        # annealing. The recorded legal batch independently encodes mask [1, 0].
        calls.append({
            "seed": rand_seed,
            "draws": np.random.default_rng(rand_seed).normal(size=8),
            "size_limit": instance._size_limit,
            "alpha": instance._alpha,
        })
        if state["error"] is not None:
            raise state["error"]
        return state["result"]

    monkeypatch.setattr(SimulatedAnnealingOptimizer, "single_process_solve", single_process)
    yield state
    np.random.set_state(original_rng)


def solve(**kwargs):
    return qubo.solve_qubo(np.zeros((2, 2)), np.array([-2., 3.]),
                           np.ones(2, dtype=int), solver="sa", **kwargs)


def make_selector(**kwargs):
    model = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[1., 0.]]))
    return FeatureSelectionWrapper(model, 2, lambda_reg=.1, solver="sa",
                                   solver_kwargs=kwargs)


def update(selector):
    return selector.update_mask([(torch.eye(2), torch.tensor([[1.], [0.]]))],
                                torch.nn.MSELoss())


def assert_same_rng_state(actual, expected):
    assert actual[0] == expected[0]
    np.testing.assert_array_equal(actual[1], expected[1])
    assert actual[2:] == expected[2:]


@pytest.mark.parametrize("kwargs, expected", [({}, 0), ({"random_state": 17}, 17),
                                            ({"random_state": np.int64(23)}, 23)])
def test_random_state_reaches_the_real_sdk_solve_dispatch(offline_sa, kwargs, expected):
    solve(**kwargs)
    assert offline_sa["calls"][0]["seed"] == expected


def test_direct_sa_helper_also_forwards_its_seed(offline_sa):
    matrix = np.array([[0., 0., -1.], [0., 0., 1.5], [0., 0., 0.]])
    qubo._solve_ising_sa(matrix, random_state=17)
    assert offline_sa["calls"][0]["seed"] == 17


@pytest.mark.parametrize("entry", ["solve_qubo", "update_mask"])
@pytest.mark.parametrize("outcome", ["success", "none", "exception"])
def test_training_rng_continues_unchanged_after_success_or_failure(
    offline_sa, entry, outcome
):
    if outcome == "none":
        offline_sa["result"] = None
    elif outcome == "exception":
        offline_sa["error"] = LookupError("offline execution failed")
    selector = make_selector(random_state=17)
    original_mask = selector.mask.clone()
    np.random.seed(128)
    np.random.normal()  # Keep a cached Gaussian draw as well as the MT19937 position.
    before = np.random.get_state()
    oracle_rng = np.random.RandomState()
    oracle_rng.set_state(before)
    expected_uniform = oracle_rng.random_sample(5)
    expected_normal = oracle_rng.normal(size=4)
    call = (lambda: solve(random_state=17)) if entry == "solve_qubo" else (
        lambda: update(selector))
    if outcome == "success":
        np.testing.assert_array_equal(call(), [1, 0])
    else:
        with pytest.raises(RuntimeError):
            call()
        torch.testing.assert_close(selector.mask, original_mask)
    assert len(offline_sa["calls"]) == 1
    assert_same_rng_state(np.random.get_state(), before)
    np.testing.assert_array_equal(np.random.random_sample(5), expected_uniform)
    np.testing.assert_array_equal(np.random.normal(size=4), expected_normal)


def test_explicit_sdk_seed_and_schedule_override_wrapper_defaults(offline_sa):
    kwargs = {"random_state": 17, "rand_seed": 29, "size_limit": 2, "alpha": .8}
    original = dict(kwargs)
    solve(**kwargs)
    call = offline_sa["calls"][0]
    assert call["seed"] == 29
    assert call["size_limit"] == 2
    assert call["alpha"] == .8
    np.testing.assert_array_equal(call["draws"], np.random.default_rng(29).normal(size=8))
    assert kwargs == original


def test_explicit_none_preserves_sdk_native_seed_selection(offline_sa):
    # SDK 1.3.1 chooses a seed from the caller's NumPy RNG when rand_seed=None.
    # Retain that explicitly requested native behavior, with no extra MAIFS reseed.
    np.random.seed(128)
    oracle_rng = np.random.RandomState(128)
    expected_seed = oracle_rng.randint(2**30)
    solve(random_state=17, rand_seed=None)
    assert offline_sa["calls"][0]["seed"] == expected_seed
    assert_same_rng_state(np.random.get_state(), oracle_rng.get_state())


def test_same_seed_repeats_local_draws_independent_of_caller_random_activity(offline_sa):
    np.random.seed(128)
    solve(random_state=17)
    np.random.normal(size=23)
    np.random.randint(1000, size=5)
    solve(random_state=17)
    first, second = offline_sa["calls"]
    assert first["seed"] == second["seed"] == 17
    np.testing.assert_array_equal(first["draws"], second["draws"])


def test_seed_forwarding_preserves_real_qubo_and_mask_update_results(offline_sa):
    # These scalar binary objectives are independent of MAIFS Ising conversion.
    candidates = list(product((0, 1), repeat=2))
    qubo_values = [-2 * a + 3 * b for a, b in candidates]
    expected = candidates[int(np.argmin(qubo_values))]
    result = solve(random_state=17)
    np.testing.assert_array_equal(result, expected)
    selector = make_selector(random_state=17)
    mask_values = [.5 * (a - 1) ** 2 + .1 * (a + b) for a, b in candidates]
    expected_mask = candidates[int(np.argmin(mask_values))]
    result = update(selector)
    np.testing.assert_array_equal(result, expected_mask)
    np.testing.assert_array_equal(selector.get_support(), np.array(expected_mask, dtype=bool))
    assert [call["seed"] for call in offline_sa["calls"]] == [17, 17]
