# -*- coding: utf-8 -*-
"""Ground-state correspondence tests for the Ising conversions.

Kaiwu solvers minimize the plain quadratic form ``s^T M s``: the official
``kw.conversion.qubo_matrix_to_ising_matrix`` emits matrices whose
``s^T M s`` equals the QUBO objective exactly, so a solver minimizing the
matrix returns the QUBO's optimal binary state. The machines' binary energy
(low energy = high probability) must therefore enter the matrix **negated**;
otherwise the sampler returns the machine's *highest*-energy states, which
are the least probable ones.

These tests pin that correspondence by brute force, without the Kaiwu SDK:
enumerate every spin configuration of the emitted matrix (the auxiliary spin
included), take the matrix minimum, decode it back to binary, and require it
to be a global minimum of the machine's own energy over all binary states.
"""

import itertools

import numpy as np
import torch

from kaiwu.torch_plugin.full_boltzmann_machine import BoltzmannMachine
from kaiwu.torch_plugin.gbrbm import (
    GaussianBernoulliRestrictedBoltzmannMachine,
)
from kaiwu.torch_plugin.restricted_boltzmann_machine import (
    RestrictedBoltzmannMachine,
)


def _matrix_min_binary(ising_mat: np.ndarray) -> np.ndarray:
    """Decode the argmin of s^T M s over all spins, auxiliary included."""
    size = ising_mat.shape[0]
    best_energy = None
    best_binary = None
    for spins in itertools.product((-1.0, 1.0), repeat=size):
        s = np.array(spins)
        energy = float(s @ ising_mat @ s)
        if best_energy is None or energy < best_energy:
            best_energy = energy
            best_binary = ((s[:-1] * s[-1]) + 1.0) / 2.0
    return best_binary


def _all_binary_states(num_nodes: int) -> torch.Tensor:
    return torch.tensor(
        list(itertools.product((0.0, 1.0), repeat=num_nodes)), dtype=torch.float32
    )


def _assert_ground_state(machine, states: torch.Tensor, energies: torch.Tensor):
    ising_mat = np.asarray(machine.get_ising_matrix())
    decoded = _matrix_min_binary(ising_mat)
    minimum = energies.min()
    argmins = {
        tuple(states[idx].tolist()) for idx in range(energies.shape[0])
        if energies[idx].item() == minimum
    }
    assert tuple(decoded.tolist()) in argmins, (
        f"matrix minimum decoded to {decoded.tolist()}, which is not a machine "
        f"ground state among {sorted(argmins)}"
    )


def test_boltzmann_machine_ground_state_test():
    torch.manual_seed(3)
    machine = BoltzmannMachine(4)
    machine.linear_bias.data = torch.tensor([0.5, -0.8, 0.2, 1.1])
    machine.quadratic_coef.data = torch.tensor(
        [
            [0.0, 0.9, -0.4, 0.2],
            [0.0, 0.0, 1.3, -0.7],
            [0.0, 0.0, 0.0, 0.6],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )
    states = _all_binary_states(4)
    energies = machine.forward(states)
    _assert_ground_state(machine, states, energies)


def test_restricted_boltzmann_machine_ground_state_test():
    torch.manual_seed(5)
    machine = RestrictedBoltzmannMachine(num_visible=3, num_hidden=2)
    machine.quadratic_coef.data = torch.tensor(
        [[0.8, -0.5], [1.2, 0.3], [-0.9, 0.7]]
    )
    machine.linear_bias.data = torch.tensor([0.4, -0.6, 0.9, -0.3, 0.2])
    states = _all_binary_states(5)
    energies = machine.forward(states)
    _assert_ground_state(machine, states, energies)


def test_gbrbm_bernoulli_ground_state_test():
    torch.manual_seed(7)
    machine = GaussianBernoulliRestrictedBoltzmannMachine(
        num_visible=3, num_hidden=2, is_visible_gaussian=True
    )
    machine.mu.data = torch.tensor([0.3, -0.6, 0.1])
    machine.log_var.data = torch.tensor([0.0, 0.0, 0.0])
    machine.quadratic_coef.data = torch.tensor(
        [[0.9, -0.4], [0.2, 0.8], [-0.6, 0.5]]
    )
    machine.linear_bias.data = torch.tensor([0.7, -0.2])
    # The matrix covers the Bernoulli units; minimize the machine energy over
    # the Bernoulli states with the Gaussian units fixed at their means.
    states = _all_binary_states(2)
    gaussian = machine.mu.data.unsqueeze(0).expand(states.shape[0], -1)
    s_all = torch.cat([gaussian, states], dim=1)
    energies = machine.energy(s_all)
    _assert_ground_state(machine, states, energies)


def test_hidden_ising_matrix_ground_state_test():
    torch.manual_seed(11)
    machine = BoltzmannMachine(4)
    machine.linear_bias.data = torch.tensor([0.3, -0.5, 0.8, 0.1])
    machine.quadratic_coef.data = torch.tensor(
        [
            [0.0, 0.6, -0.3, 0.4],
            [0.0, 0.0, 0.9, -0.2],
            [0.0, 0.0, 0.0, 0.5],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )
    s_visible = torch.tensor([1.0, 0.0])
    ising_mat = np.asarray(machine._hidden_to_ising_matrix(s_visible))
    decoded = _matrix_min_binary(ising_mat)
    states = _all_binary_states(2)
    s_all = torch.cat(
        [s_visible.unsqueeze(0).expand(states.shape[0], -1), states], dim=1
    )
    energies = machine.forward(s_all)
    minimum = energies.min()
    argmins = {
        tuple(states[idx].tolist()) for idx in range(energies.shape[0])
        if energies[idx].item() == minimum
    }
    assert tuple(decoded.tolist()) in argmins
