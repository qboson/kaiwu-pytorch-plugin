"""Symmetrized quadratic coefficient contract for the full Boltzmann machine."""

import os
import sys

import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from kaiwu.torch_plugin.full_boltzmann_machine import BoltzmannMachine  # noqa: E402


def build_machine():
    """Builds a three-node machine with distinct upper/lower coefficients."""
    machine = BoltzmannMachine(num_nodes=3)
    with torch.no_grad():
        machine.quadratic_coef.copy_(
            torch.tensor(
                [[0.0, 1.0, 2.0], [3.0, 0.0, 5.0], [7.0, 8.0, 0.0]],
                dtype=machine.quadratic_coef.dtype,
            )
        )
    return machine


def test_only_the_upper_triangle_is_used_and_mirrored():
    """The lower triangle is ignored; the result is symmetric with no diagonal."""
    symmetric = build_machine().symmetrized_quadratic_coef()
    assert symmetric.tolist() == [
        [0.0, 1.0, 2.0],
        [1.0, 0.0, 5.0],
        [2.0, 5.0, 0.0],
    ]


def test_result_is_symmetric_and_keeps_the_dtype():
    """The returned matrix is symmetric and uses the parameter dtype."""
    machine = build_machine()
    symmetric = machine.symmetrized_quadratic_coef()
    assert torch.equal(symmetric, symmetric.T)
    assert symmetric.dtype == machine.quadratic_coef.dtype
