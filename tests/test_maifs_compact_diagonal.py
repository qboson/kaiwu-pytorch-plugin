"""Diagonal derivatives keep scalar entries until the QUBO matrix boundary."""

import os
import sys
from unittest.mock import patch

import numpy as np
import pytest
import torch
from torch import nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from kaiwu.torch_plugin import FeatureSelectionWrapper


def problem():
    model = nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[2., 3.]]))
    selector = FeatureSelectionWrapper(model, 2, min_selected_features=0, lambda_reg=.5)
    return selector, [(torch.eye(2), torch.zeros(2, 1))]


def test_diagonal_matrix_api_stacks_only_scalar_second_derivatives():
    selector, loader = problem()
    stack = torch.stack
    shapes = []

    def record(tensors, *args, **kwargs):
        shapes.append([tuple(t.shape) for t in tensors])
        assert all(t.untyped_storage().nbytes() == t.element_size() for t in tensors)
        return stack(tensors, *args, **kwargs)

    with patch("torch.stack", side_effect=record):
        gradient, hessian = selector.compute_mask_derivatives(loader, nn.MSELoss(), "diagonal")
    np.testing.assert_allclose(gradient, [4., 9.])
    np.testing.assert_allclose(hessian, [[4., 0.], [0., 9.]])
    assert shapes == [[(), ()]]


def test_compact_diagonal_and_qubo_preserve_analytic_values():
    selector, loader = problem()
    gradient, diagonal = selector.compute_mask_derivatives(
        loader, nn.MSELoss(), "diagonal", compact_diagonal=True,
    )
    assert diagonal.shape == (2,)
    np.testing.assert_allclose(diagonal, [4., 9.])
    quadratic, linear = selector._build_qubo(gradient, diagonal)
    np.testing.assert_allclose(quadratic, [[4., 0.], [0., 9.]])
    np.testing.assert_allclose(linear, [.5, .5])


def test_update_mask_uses_compact_path_and_keeps_solver_behavior():
    selector, loader = problem()
    with patch.object(selector, "compute_mask_derivatives", wraps=selector.compute_mask_derivatives) as compute:
        selected = selector.update_mask(loader, nn.MSELoss(), "diagonal")
    assert compute.call_args.kwargs.get("compact_diagonal") is True
    np.testing.assert_array_equal(selected, [0, 0])
    np.testing.assert_array_equal(selector.get_support(), [False, False])


def test_compact_diagonal_rejects_full_mode():
    selector, loader = problem()
    with pytest.raises(ValueError, match="diagonal"):
        selector.compute_mask_derivatives(loader, nn.MSELoss(), "full", compact_diagonal=True)


def test_compact_qubo_keeps_current_mask_and_cardinality_penalty():
    selector, _ = problem()
    selector.cardinality_k = 1
    with torch.no_grad():
        selector.mask.copy_(torch.tensor([0., 1.]))
    quadratic, linear = selector._build_qubo(np.array([4., 9.]), np.array([4., 9.]))
    np.testing.assert_allclose(quadratic, [[24., 20.], [20., 29.]])
    np.testing.assert_allclose(linear, [-15.5, -19.5])
