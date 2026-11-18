# -*- coding: utf-8 -*-
"""Regression tests for BoltzmannMachine coupling-matrix construction.

``symmetrized_quadratic_coef`` reads only the strict upper triangle, so a
user-supplied ``quadratic_coef`` with a nonzero diagonal used to be accepted
and then silently dropped. For the binary states the machine samples
(``x_i * x_i == x_i``), every diagonal entry is equivalent to a linear bias,
so the dropped diagonal meant the machine modeled a different energy than
the caller supplied. The constructor now rejects such matrices instead.
"""

import pytest
import torch

from kaiwu.torch_plugin.full_boltzmann_machine import BoltzmannMachine


def test_rejects_nonzero_diagonal_test():
    coupling = torch.zeros(3, 3)
    coupling[0, 0] = 0.5
    coupling[0, 1] = 1.0
    with pytest.raises(ValueError, match="zero diagonal"):
        BoltzmannMachine(
            3,
            quadratic_coef=coupling,
            linear_bias=torch.zeros(3),
        )


def test_rejects_non_square_coupling_test():
    with pytest.raises(ValueError, match="square matrix"):
        BoltzmannMachine(
            2,
            quadratic_coef=torch.zeros(2, 3),
            linear_bias=torch.zeros(2),
        )


def test_accepts_zero_diagonal_coupling_test():
    coupling = torch.tensor([[0.0, 1.0, -2.0], [5.0, 0.0, 0.5], [1.0, 1.0, 0.0]])
    machine = BoltzmannMachine(
        3,
        quadratic_coef=coupling,
        linear_bias=torch.tensor([0.1, 0.2, 0.3]),
    )
    # Only the strict upper triangle enters the symmetrized coupling: the
    # lower-triangle entries below were already dropped by design.
    symmetrized = machine.symmetrized_quadratic_coef()
    assert torch.equal(
        symmetrized,
        torch.tensor([[0.0, 1.0, -2.0], [1.0, 0.0, 0.5], [-2.0, 0.5, 0.0]]),
    )


def test_zero_diagonal_energy_matches_hand_computation_test():
    coupling = torch.tensor([[0.0, 1.0], [0.0, 0.0]])
    machine = BoltzmannMachine(
        2,
        quadratic_coef=coupling,
        linear_bias=torch.tensor([0.5, -0.25]),
    )
    states = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    energies = machine.forward(states)
    # E(x) = -b . x - J_01 * x_0 * x_1
    assert torch.allclose(
        energies,
        torch.tensor([0.0, -0.5, 0.25, -0.5 - 1.0 + 0.25]),
    )


def test_default_initialization_is_unaffected_test():
    torch.manual_seed(0)
    machine = BoltzmannMachine(4)
    assert machine.quadratic_coef.shape == (4, 4)
    assert machine.linear_bias.shape == (4,)
