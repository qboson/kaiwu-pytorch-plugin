"""Independent numerical oracles for the mask-to-QUBO approximation."""

from itertools import product

import numpy as np
import pytest
import torch
from torch import nn

from kaiwu.torch_plugin import FeatureSelectionWrapper
from kaiwu.torch_plugin.maifs import plugin


INPUTS = np.array([[.2, -.7, 1.1], [1.3, .4, -.5], [-.8, .9, .6], [.7, -1.2, .3]])
TARGETS = np.array([[.3], [-.6], [.8], [-.2]])
MASK = np.array([1., 0., 1.])


def finite_differences(function, point, step=1e-4):
    """Differentiate only NumPy function values, without using autograd."""
    directions = np.eye(len(point)) * step
    center = function(point)
    gradient = np.empty_like(point)
    hessian = np.empty((len(point), len(point)))
    for row, delta in enumerate(directions):
        above, below = function(point + delta), function(point - delta)
        gradient[row] = (above - below) / (2 * step)
        hessian[row, row] = (above - 2 * center + below) / step**2
        for column in range(row):
            other = directions[column]
            mixed = (function(point + delta + other) - function(point + delta - other)
                     - function(point - delta + other) + function(point - delta - other))
            hessian[row, column] = hessian[column, row] = mixed / (4 * step**2)
    return gradient, hessian


def loader():
    """Use two batches to exercise the complete derivative input path."""
    return [(torch.from_numpy(INPUTS[:2]), torch.from_numpy(TARGETS[:2])),
            (torch.from_numpy(INPUTS[2:]), torch.from_numpy(TARGETS[2:]))]


@pytest.mark.parametrize("mode", ["full", "diagonal"])
def test_nonlinear_mask_derivatives_match_finite_differences(mode):
    """Check both nonlinear curvature and mixed partials at a nontrivial mask."""
    weights = np.array([[.7, -.4, 1.2], [-.9, .6, .3]])
    bias = np.array([.2, -.1])
    output_weights = np.array([[.8, -1.1]])
    with torch.random.fork_rng(devices=[]):
        model = nn.Sequential(nn.Linear(3, 2), nn.Tanh(), nn.Linear(2, 1)).double()
    with torch.no_grad():
        model[0].weight.copy_(torch.from_numpy(weights))
        model[0].bias.copy_(torch.from_numpy(bias))
        model[2].weight.copy_(torch.from_numpy(output_weights))
        model[2].bias.fill_(.15)
    selector = FeatureSelectionWrapper(model, 3).double()
    selector.mask.copy_(torch.from_numpy(MASK))

    def numpy_loss(mask):
        prediction = np.tanh((INPUTS * mask) @ weights.T + bias) @ output_weights.T + .15
        return np.mean((prediction - TARGETS)**2)

    expected_gradient, expected_hessian = finite_differences(numpy_loss, MASK)
    off_diagonal = expected_hessian - np.diag(np.diag(expected_hessian))
    assert np.max(np.abs(off_diagonal)) > .1
    if mode == "diagonal":
        expected_hessian = np.diag(np.diag(expected_hessian))

    gradient, hessian = selector.compute_mask_derivatives(loader(), nn.MSELoss(), mode)

    np.testing.assert_allclose(gradient, expected_gradient, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(hessian, expected_hessian, rtol=1e-5, atol=1e-6)
    np.testing.assert_array_equal(selector.mask.numpy(), MASK)


@pytest.mark.parametrize("cardinality", [None, 1])
def test_qubo_matches_quadratic_loss_differences_for_every_binary_mask(monkeypatch, cardinality):
    """A quadratic loss has an exact Taylor surrogate up to one constant offset."""
    weights = np.array([[.7, -.4, 1.2]])
    with torch.random.fork_rng(devices=[]):
        model = nn.Linear(3, 1).double()
    with torch.no_grad():
        model.weight.copy_(torch.from_numpy(weights))
        model.bias.fill_(.25)
    selector = FeatureSelectionWrapper(model, 3, lambda_reg=.17, cardinality_k=cardinality,
                                       min_selected_features=0).double()
    selector.mask.copy_(torch.from_numpy(MASK))
    captured = []

    def inspect_solver(quadratic_matrix, linear_vector, initial_state, solver, **_kwargs):
        assert solver == "local_search"
        captured.append((quadratic_matrix.copy(), linear_vector.copy()))
        np.testing.assert_array_equal(initial_state, MASK)
        return initial_state.copy()

    monkeypatch.setattr(plugin, "solve_qubo", inspect_solver)
    selector.update_mask(loader(), nn.MSELoss(), hessian_mode="full")
    assert len(captured) == 1
    quadratic, linear = captured[0]

    def actual_objective(mask):
        prediction = (INPUTS * mask) @ weights.T + .25
        loss = np.mean((prediction - TARGETS)**2) + .17 * mask.sum()
        if cardinality is not None:
            loss += 10. * (mask.sum() - cardinality)**2
        return loss

    def qubo_objective(mask):
        return .5 * mask @ quadratic @ mask + linear @ mask

    reference_loss = actual_objective(MASK)
    reference_qubo = qubo_objective(MASK)
    for bits in product((0., 1.), repeat=3):
        candidate = np.array(bits)
        np.testing.assert_allclose(qubo_objective(candidate) - reference_qubo,
                                   actual_objective(candidate) - reference_loss,
                                   rtol=1e-12, atol=1e-12)
