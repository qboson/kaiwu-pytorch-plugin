"""QUBO-assembly contract for the MAIFS feature-selection wrapper."""

import os
import sys

import numpy as np
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from kaiwu.torch_plugin.maifs.plugin import FeatureSelectionWrapper  # noqa: E402


def build_wrapper(cardinality_k=None):
    """Builds only the attributes ``_build_qubo`` reads."""
    wrapper = FeatureSelectionWrapper.__new__(FeatureSelectionWrapper)
    wrapper.mask = torch.tensor([1.0, 0.0])
    wrapper.lambda_reg = np.array([0.5, 0.5])
    wrapper.cardinality_k = cardinality_k
    return wrapper


HESSIAN = np.array([[1.0, 2.0], [2.0, 3.0]])
GRADIENT = np.array([10.0, 20.0])


def test_linear_term_centres_the_gradient_at_the_current_mask():
    """Q = H and l = g - H·mask + lambda."""
    wrapper = build_wrapper()
    quadratic, linear = wrapper._build_qubo(GRADIENT, HESSIAN)
    assert np.array_equal(quadratic, HESSIAN)
    assert linear.tolist() == [9.5, 18.5]


def test_cardinality_penalty_is_added_to_every_pair_and_the_constant():
    """The penalty adds ``2P`` to all pairs and subtracts ``2P·k`` from l."""
    wrapper = build_wrapper(cardinality_k=1)
    quadratic, linear = wrapper._build_qubo(GRADIENT, HESSIAN)
    assert quadratic.tolist() == [[21.0, 22.0], [22.0, 23.0]]
    assert linear.tolist() == [-10.5, -1.5]
