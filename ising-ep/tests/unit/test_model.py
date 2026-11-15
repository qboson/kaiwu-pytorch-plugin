"""Tests for the Ising EP model: initialization, energy, QUBO and prediction."""

# Test functions document intent in their summary; parametrised values do not
# benefit from Args sections.
# pylint: disable=missing-param-doc,missing-return-doc

from __future__ import annotations

import itertools

import numpy as np
import pytest
import torch

from ising_ep.model import IsingEPModel

from tests.conftest import iter_physical_states, qubo_score, set_parameter, to_numpy


def test_same_seed_reproduces_parameters_and_shapes() -> None:
    """One seed fully determines the initial parameters."""
    first = IsingEPModel(n_in=5, hidden_sizes=3, n_classes=2, spins_per_class=2, seed=7)
    second = IsingEPModel(n_in=5, hidden_sizes=3, n_classes=2, spins_per_class=2, seed=7)
    other = IsingEPModel(n_in=5, hidden_sizes=3, n_classes=2, spins_per_class=2, seed=8)

    left = first.parameter_snapshot()
    right = second.parameter_snapshot()
    for key, shape in (
        ("weights.0", (5, 3)),
        ("weights.1", (3, 4)),
        ("biases.0", (3,)),
        ("biases.1", (4,)),
    ):
        np.testing.assert_allclose(to_numpy(left[key]), to_numpy(right[key]))
        assert tuple(left[key].shape) == shape
        assert left[key].dtype == torch.float64
    assert not np.array_equal(
        to_numpy(left["weights.0"]), to_numpy(other.parameter_snapshot()["weights.0"])
    )
    assert first.layer_sizes == (5, 3, 4)
    assert first.n_hidden == 3


def test_integer_hidden_size_is_a_single_layer() -> None:
    """An integer width is the one-hidden-layer network."""
    model = IsingEPModel(n_in=4, hidden_sizes=8, n_classes=2, spins_per_class=2, seed=1)

    assert model.hidden_sizes == (8,)
    assert model.layer_sizes == (4, 8, 4)


def test_couplings_start_within_the_inverse_sqrt_fan_in_range() -> None:
    """Initialization stays inside ``+-1/sqrt(fan_in)`` for each coupling."""
    model = IsingEPModel(n_in=64, hidden_sizes=16, n_classes=2, spins_per_class=2, seed=3)
    params = model.parameter_snapshot()

    assert to_numpy(params["weights.0"]).max() <= 1.0 / np.sqrt(64)
    assert to_numpy(params["weights.0"]).min() >= -1.0 / np.sqrt(64)
    assert to_numpy(params["weights.1"]).max() <= 1.0 / np.sqrt(16)
    assert to_numpy(params["weights.1"]).min() >= -1.0 / np.sqrt(16)
    np.testing.assert_array_equal(to_numpy(params["biases.0"]), np.zeros(16))


def test_biases_are_learnable_parameters() -> None:
    """Every weight and bias participates in gradient descent."""
    model = IsingEPModel(n_in=3, hidden_sizes=2, n_classes=2, spins_per_class=2)

    assert {name for name, _ in model.named_parameters()} == {
        "weights.0",
        "weights.1",
        "biases.0",
        "biases.1",
    }


def test_target_spins_encode_one_class_block_per_label() -> None:
    """A label maps to a +1 block on a -1 background, one sample at a time."""
    model = IsingEPModel(n_in=2, hidden_sizes=2, n_classes=10, spins_per_class=4)

    for label, start in ((0, 0), (3, 12), (9, 36)):
        encoded = to_numpy(model.target_spins(label))
        assert encoded.shape == (40,)
        np.testing.assert_array_equal(encoded[start : start + 4], 1.0)
        assert encoded.sum() == pytest.approx(-32.0)


def test_target_spins_rejects_a_label_outside_the_class_range() -> None:
    """Labels must name an existing class."""
    model = IsingEPModel(n_in=2, hidden_sizes=2, n_classes=3, spins_per_class=2)

    with pytest.raises(ValueError, match=r"label must be in \[0, 3\)"):
        model.target_spins(3)


def test_energy_matches_the_written_out_bilinear_form() -> None:
    """``energy`` equals the closed-form EP energy for one hidden layer."""
    model = IsingEPModel(n_in=4, hidden_sizes=3, n_classes=2, spins_per_class=2, seed=2)
    x = np.array([0.2, -0.5, 0.1, 0.8])
    hidden = np.array([1.0, -1.0, 1.0])
    output = np.array([1.0, -1.0, -1.0, 1.0])
    params = {name: to_numpy(value) for name, value in model.parameter_snapshot().items()}

    expected = float(
        -x @ params["weights.0"] @ hidden
        - hidden @ params["weights.1"] @ output
        - params["biases.0"] @ hidden
        - params["biases.1"] @ output
    )

    assert float(model.energy(x, [hidden, output])) == pytest.approx(expected, abs=1e-12)


def test_stacked_energy_sums_every_adjacent_layer() -> None:
    """Two hidden layers add one extra bilinear term and one extra bias."""
    model = IsingEPModel(n_in=3, hidden_sizes=(2, 2), n_classes=2, spins_per_class=1, seed=4)
    x = np.array([0.2, -0.1, 0.5])
    first = np.array([1.0, -1.0])
    second = np.array([-1.0, 1.0])
    output = np.array([1.0, -1.0])
    params = {name: to_numpy(value) for name, value in model.parameter_snapshot().items()}

    expected = float(
        -x @ params["weights.0"] @ first
        - first @ params["weights.1"] @ second
        - second @ params["weights.2"] @ output
        - params["biases.0"] @ first
        - params["biases.1"] @ second
        - params["biases.2"] @ output
    )

    assert float(model.energy(x, [first, second, output])) == pytest.approx(expected, abs=1e-12)
    assert model.layer_sizes == (3, 2, 2, 2)


def test_energy_returns_a_scalar_that_tracks_parameter_gradients() -> None:
    """The energy stays differentiable in the parameters, not in the spins."""
    model = IsingEPModel(n_in=3, hidden_sizes=2, n_classes=2, spins_per_class=2, seed=5)

    value = model.energy(np.ones(3), [np.ones(2), np.ones(4)])
    assert value.ndim == 0
    value.backward()

    # dE/dW = -s_prev (x) s, so every entry is -1 for all-ones inputs.
    np.testing.assert_allclose(to_numpy(model.weights[0].grad), -np.ones((3, 2)))
    np.testing.assert_allclose(to_numpy(model.weights[1].grad), -np.ones((2, 4)))
    np.testing.assert_allclose(to_numpy(model.biases[0].grad), -np.ones(2))
    np.testing.assert_allclose(to_numpy(model.biases[1].grad), -np.ones(4))


def test_middle_layer_gradient_is_the_adjacent_outer_product() -> None:
    """Each stacked weight's gradient is ``-s_left (x) s_right``."""
    model = IsingEPModel(n_in=2, hidden_sizes=(2, 3), n_classes=2, spins_per_class=1, seed=5)
    first = np.array([1.0, -1.0])
    second = np.array([1.0, 1.0, -1.0])
    output = np.array([-1.0, 1.0])

    model.energy(np.ones(2), [first, second, output]).backward()

    np.testing.assert_allclose(to_numpy(model.weights[1].grad), -np.outer(first, second))
    np.testing.assert_allclose(to_numpy(model.weights[2].grad), -np.outer(second, output))


def test_cost_is_the_linear_form_of_the_squared_output_error() -> None:
    """``C = n_output - target . output`` equals ``0.5 * ||o - y||**2``."""
    model = IsingEPModel(n_in=2, hidden_sizes=2, n_classes=2, spins_per_class=2)
    target = model.target_spins(0)
    output = torch.tensor([1.0, 1.0, 1.0, -1.0], dtype=torch.float64)

    observed = float(model.cost(output, target))

    assert observed == pytest.approx(2.0)
    assert observed == pytest.approx(
        0.5 * float(((output - target) ** 2).sum())
    )


def test_cost_is_zero_when_the_output_equals_the_target() -> None:
    """A perfect output costs nothing."""
    model = IsingEPModel(n_in=2, hidden_sizes=2, n_classes=3, spins_per_class=2)
    target = model.target_spins(2)

    assert float(model.cost(target, target)) == pytest.approx(0.0)


def test_fields_match_the_reference_linear_formulas() -> None:
    """Local fields follow ``x W + b`` and ``b_out + beta * target``."""
    model = IsingEPModel(n_in=3, hidden_sizes=2, n_classes=2, spins_per_class=2, seed=3)
    params = {name: to_numpy(value) for name, value in model.parameter_snapshot().items()}
    x = np.array([0.2, -0.4, 0.7])
    target = to_numpy(model.target_spins(1))

    fields = model.fields(x, target, beta=0.6)
    hidden_field, output_field = fields[0], fields[1]

    np.testing.assert_allclose(
        to_numpy(hidden_field), params["weights.0"].T @ x + params["biases.0"]
    )
    np.testing.assert_allclose(to_numpy(output_field), params["biases.1"] + 0.6 * target)


def test_middle_field_is_only_the_bias() -> None:
    """A layer with spin neighbours on both sides carries no clamped input."""
    model = IsingEPModel(n_in=2, hidden_sizes=(2, 3), n_classes=1, spins_per_class=2, seed=3)
    params = {name: to_numpy(value) for name, value in model.parameter_snapshot().items()}

    fields = model.fields(np.ones(2), model.target_spins(0), beta=0.4)

    np.testing.assert_allclose(to_numpy(fields[1]), params["biases.1"])
    assert len(fields) == 3


def test_free_phase_output_field_drops_the_target() -> None:
    """With beta = 0 the output field carries no label information."""
    model = IsingEPModel(n_in=2, hidden_sizes=2, n_classes=2, spins_per_class=2, seed=1)

    fields = model.fields(np.ones(2), model.target_spins(0), beta=0.0)

    np.testing.assert_allclose(
        to_numpy(fields[-1]), to_numpy(model.parameter_snapshot()["biases.1"])
    )


def test_prediction_uses_block_sums_then_local_field_tie_break() -> None:
    """Class scores are block sums; ties fall through to the local field."""
    model = IsingEPModel(n_in=2, hidden_sizes=2, n_classes=4, spins_per_class=2, seed=4)
    set_parameter(model.weights[1], np.zeros((2, 8)))
    set_parameter(model.biases[1], np.zeros(8))
    hidden = np.zeros(2)

    tied_output = np.array([1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0])
    prediction, tied = model.predict([hidden, tied_output])
    assert (prediction, tied) == (0, True)

    clear_output = np.array([-1.0, -1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0])
    prediction, tied = model.predict([hidden, clear_output])
    assert (prediction, tied) == (1, False)


def test_prediction_tie_break_prefers_the_stronger_local_field() -> None:
    """Among tied blocks the larger output local field wins."""
    model = IsingEPModel(n_in=2, hidden_sizes=1, n_classes=2, spins_per_class=1, seed=4)
    set_parameter(model.weights[1], np.zeros((1, 2)))
    set_parameter(model.biases[1], np.array([0.0, 5.0]))

    prediction, tied = model.predict([np.zeros(1), np.array([1.0, 1.0])])

    assert (prediction, tied) == (1, True)


def test_parameter_snapshot_is_independent_of_later_updates() -> None:
    """A snapshot keeps its values after the live weights move."""
    model = IsingEPModel(n_in=2, hidden_sizes=2, n_classes=1, spins_per_class=2, seed=9)
    snapshot = model.parameter_snapshot()

    with torch.no_grad():
        model.weights[0].add_(1.0)

    np.testing.assert_allclose(
        to_numpy(snapshot["weights.0"]),
        to_numpy(model.parameter_snapshot()["weights.0"]) - 1.0,
    )
    assert snapshot["weights.0"].data_ptr() != model.weights[0].data_ptr()


def test_default_init_gains_match_explicit_ones() -> None:
    """Omitting ``init_gains`` is the same as passing a 1 for every connection."""
    omitted = IsingEPModel(n_in=5, hidden_sizes=3, n_classes=2, spins_per_class=2, seed=7)
    explicit = IsingEPModel(
        n_in=5,
        hidden_sizes=3,
        n_classes=2,
        spins_per_class=2,
        seed=7,
        init_gains=(1.0, 1.0),
    )

    left = omitted.parameter_snapshot()
    right = explicit.parameter_snapshot()
    for key in left:
        np.testing.assert_allclose(to_numpy(left[key]), to_numpy(right[key]))
    assert omitted.init_gains == (1.0, 1.0)


def test_init_gains_scale_the_uniform_range() -> None:
    """Each connection's draw stays inside ``+-gain/sqrt(fan_in)``."""
    model = IsingEPModel(
        n_in=64,
        hidden_sizes=16,
        n_classes=2,
        spins_per_class=2,
        seed=3,
        init_gains=(0.5, 2.0),
    )
    params = model.parameter_snapshot()

    assert to_numpy(params["weights.0"]).max() <= 0.5 / np.sqrt(64)
    assert to_numpy(params["weights.0"]).min() >= -0.5 / np.sqrt(64)
    assert to_numpy(params["weights.1"]).max() <= 2.0 / np.sqrt(16)
    assert to_numpy(params["weights.1"]).min() >= -2.0 / np.sqrt(16)


def test_init_gains_rejects_the_wrong_length_or_a_non_positive_value() -> None:
    """The gain list must match the number of connections and stay positive."""
    with pytest.raises(ValueError, match="init_gains must have length 2"):
        IsingEPModel(n_in=4, hidden_sizes=3, n_classes=2, spins_per_class=1, init_gains=(1.0,))
    with pytest.raises(ValueError, match="strictly positive"):
        IsingEPModel(
            n_in=4,
            hidden_sizes=3,
            n_classes=2,
            spins_per_class=1,
            init_gains=(1.0, 0.0),
        )


def _reference_objective(
    x: np.ndarray,
    parts: list[np.ndarray],
    target: np.ndarray,
    params: dict[str, np.ndarray],
    beta: float,
) -> float:
    """Evaluate ``F = E + beta * C`` straight from the stacked EP definition."""
    previous = x
    energy = 0.0
    for index, current in enumerate(parts):
        energy -= previous @ params[f"weights.{index}"] @ current
        energy -= params[f"biases.{index}"] @ current
        previous = current
    output = parts[-1]
    cost = output.size - target @ output
    return float(energy + beta * cost)


def _model_with_fields(
    coupling: np.ndarray,
    hidden_field: np.ndarray,
    output_field: np.ndarray,
) -> IsingEPModel:
    """Return a model whose fields equal the arguments at a zero input.

    The input coupling is zero. Callers pass a zero target, so the hidden
    field is the first bias and the output field is the last bias, and
    ``beta`` only changes the QUBO offset.
    """
    n_hidden, n_output = np.asarray(coupling).shape
    model = IsingEPModel(
        n_in=1,
        hidden_sizes=int(n_hidden),
        n_classes=int(n_output),
        spins_per_class=1,
        seed=0,
    )
    with torch.no_grad():
        model.weights[0].zero_()
        model.weights[1].copy_(torch.as_tensor(coupling, dtype=torch.float64))
        model.biases[0].copy_(torch.as_tensor(hidden_field, dtype=torch.float64))
        model.biases[1].copy_(torch.as_tensor(output_field, dtype=torch.float64))
    return model


def test_qubo_matches_explicit_closed_form() -> None:
    """The QUBO matrix and offset match the hand-expanded closed form."""
    coupling = np.array([[1.0, 2.0], [3.0, 4.0]])
    hidden_field = np.array([0.5, -1.0])
    output_field = np.array([2.0, -0.5])
    beta = 0.3
    model = _model_with_fields(coupling, hidden_field, output_field)

    matrix, constant = model.qubo(torch.zeros(1), torch.zeros(model.n_output), beta)

    # Hidden diag: 2*V.sum(axis=1) - 2*H_h = [2*3-1, 2*7+2] = [5, 16]
    # Output diag: 2*V.sum(axis=0) - 2*H_o = [2*4-4, 2*6+1] = [4, 13]
    # Cross block: -4*V
    # Offset: beta*n_out + H_h.sum() + H_o.sum() - V.sum() = 0.6 - 0.5 + 1.5 - 10
    expected = np.array(
        [
            [5.0, 0.0, -4.0, -8.0],
            [0.0, 16.0, -12.0, -16.0],
            [0.0, 0.0, 4.0, 0.0],
            [0.0, 0.0, 0.0, 13.0],
        ]
    )
    np.testing.assert_allclose(to_numpy(matrix), expected)
    assert constant == pytest.approx(-8.4)


def test_qubo_returns_matrix_and_offset_without_a_wrapper_type() -> None:
    """The model hands back a plain tensor and float, not a problem object."""
    model = _model_with_fields(np.ones((2, 3)), np.zeros(2), np.zeros(3))

    matrix, constant = model.qubo(torch.zeros(1), torch.zeros(model.n_output), 0.0)

    assert matrix.shape == (5, 5)
    assert isinstance(constant, float)


@pytest.mark.parametrize("beta", [0.0, 0.7])
def test_ep_and_qubo_energies_differ_only_by_a_constant(beta: float) -> None:
    """Over all 64 states the two energies agree up to one shared offset."""
    model = IsingEPModel(n_in=3, hidden_sizes=2, n_classes=2, spins_per_class=2, seed=11)
    params = {name: to_numpy(value) for name, value in model.parameter_snapshot().items()}
    x = np.array([0.1, 0.7, -0.2])
    target = to_numpy(model.target_spins(1)).reshape(-1)

    matrix, constant = model.qubo(x, target, beta)

    ep_values = []
    qubo_values = []
    width = model.n_hidden + model.n_output
    for spins in iter_physical_states(width):
        parts = [to_numpy(block) for block in model.split_state(spins)]
        ep_values.append(_reference_objective(x, parts, target, params, beta))
        qubo_values.append(qubo_score(matrix, constant, spins))

    ep_array = np.asarray(ep_values)
    qubo_array = np.asarray(qubo_values)
    assert ep_array.size == 64
    np.testing.assert_allclose(ep_array, qubo_array, atol=1e-10)
    assert set(np.flatnonzero(ep_array <= ep_array.min() + 1e-10)) == set(
        np.flatnonzero(qubo_array <= qubo_array.min() + 1e-10)
    )


def test_two_hidden_layers_match_the_qubo_on_every_state() -> None:
    """A second hidden layer stays inside the same QUBO identity."""
    model = IsingEPModel(n_in=2, hidden_sizes=(2, 2), n_classes=2, spins_per_class=1, seed=3)
    params = {name: to_numpy(value) for name, value in model.parameter_snapshot().items()}
    x = np.array([0.4, -0.3])
    target = to_numpy(model.target_spins(0)).reshape(-1)
    beta = 0.5
    matrix, constant = model.qubo(x, target, beta)

    ep_values = []
    qubo_values = []
    for spins in iter_physical_states(sum(model.spin_widths)):
        parts = [to_numpy(block) for block in model.split_state(spins)]
        ep_values.append(_reference_objective(x, parts, target, params, beta))
        qubo_values.append(qubo_score(matrix, constant, spins))

    np.testing.assert_allclose(np.asarray(ep_values), np.asarray(qubo_values), atol=1e-10)


def test_split_state_splits_on_every_layer_boundary() -> None:
    """Blocks follow the hidden widths and then the output."""
    model = IsingEPModel(n_in=1, hidden_sizes=(1, 1), n_classes=1, spins_per_class=1, seed=0)
    parts = model.split_state(np.array([1.0, -1.0, -1.0]))
    first, second, output = parts[0], parts[1], parts[2]

    np.testing.assert_array_equal(to_numpy(first), [1.0])
    np.testing.assert_array_equal(to_numpy(second), [-1.0])
    np.testing.assert_array_equal(to_numpy(output), [-1.0])


def test_iter_physical_states_enumerates_every_assignment() -> None:
    """The helper yields all 2**n spin assignments without repeats."""
    states = [tuple(to_numpy(s).tolist()) for s in iter_physical_states(3)]

    assert len(states) == 8
    assert len(set(states)) == 8
    assert set(states) == set(itertools.product((-1.0, 1.0), repeat=3))
