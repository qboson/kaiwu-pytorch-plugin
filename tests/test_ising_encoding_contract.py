"""Pin the Ising encoding used by the samplers.

For 0/1 states ``s`` and Ising spins ``sigma = 2s - 1`` with a +1 auxiliary spin,
the model energy must equal ``-[sigma, 1] @ J @ [sigma, 1]`` up to an additive
constant, so the argmin/argmax over all states are preserved.
"""

import itertools

import numpy as np
import torch

from kaiwu.torch_plugin import BoltzmannMachine, RestrictedBoltzmannMachine


def _check(model) -> None:
    num_nodes = model.num_nodes
    ising = model.get_ising_matrix()
    assert ising.shape == (num_nodes + 1, num_nodes + 1)
    assert np.allclose(ising, ising.T)

    states = [
        torch.tensor(state, dtype=torch.float32)
        for state in itertools.product([0, 1], repeat=num_nodes)
    ]
    energies = np.array(
        [float(model(state.unsqueeze(0))[0]) for state in states]
    )
    ising_energies = np.array(
        [
            -float(
                np.r_[2 * state.numpy() - 1, 1.0]
                @ ising
                @ np.r_[2 * state.numpy() - 1, 1.0]
            )
            for state in states
        ]
    )

    offset = energies - ising_energies
    assert np.ptp(offset) < 1e-5
    assert energies.argmin() == ising_energies.argmin()
    assert energies.argmax() == ising_energies.argmax()


def test_restricted_boltzmann_machine_encoding() -> None:
    torch.manual_seed(0)
    _check(RestrictedBoltzmannMachine(num_visible=2, num_hidden=2))


def test_boltzmann_machine_encoding() -> None:
    torch.manual_seed(0)
    _check(BoltzmannMachine(num_nodes=4))
