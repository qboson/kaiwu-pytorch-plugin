"""Brute-force equivalence tests between model energies and Ising matrices.

``_to_ising_matrix`` and ``_hidden_to_ising_matrix`` are the bridge between
the PyTorch energy models and the Kaiwu samplers: the sampler maximises
``s @ J @ s`` over spins, so a conversion bug silently turns sampling and
training into optimisation of the wrong objective. The existing tests check
shapes, symmetry and one hand-computed energy difference; here the full
state space is small enough to enumerate, so each conversion is verified
end to end: the spin configuration decoded from the Ising objective optimum
must attain the brute-force minimum of the model energy (ties may differ in
state, never in value).

The aux spin decode follows ``AbstractBoltzmannMachine.sample``:
``bits = (spins[:, :-1] * spins[:, [-1]] + 1) / 2``.
"""

import itertools
import unittest

import numpy as np
import torch

from kaiwu.torch_plugin import BoltzmannMachine as BM
from kaiwu.torch_plugin.gbrbm import GaussianBernoulliRestrictedBoltzmannMachine


def enumerate_bits(num_bits: int) -> torch.Tensor:
    """All 0/1 states, shape ``(2**num_bits, num_bits)``."""
    return torch.tensor(
        list(itertools.product([0.0, 1.0], repeat=num_bits)), dtype=torch.float32
    )


def best_ising_bits(ising_mat: np.ndarray) -> torch.Tensor:
    """Decoded bits maximising the Kaiwu objective ``spins @ J @ spins``.

    Enumerates every spin row including both aux values (flipping all spins
    leaves the quadratic objective invariant, so both decodings occur).
    """
    mat = torch.tensor(ising_mat, dtype=torch.float64)
    num_spins = mat.shape[0]
    bits = enumerate_bits(num_spins - 1).double()
    spins = 2.0 * bits - 1.0
    spins = spins.repeat_interleave(2, dim=0)
    rows = spins.shape[0]
    aux = torch.where(torch.arange(rows) % 2 == 0, -1.0, 1.0).unsqueeze(1)
    spins = torch.cat([spins, aux], dim=-1)
    objective = torch.einsum("bi,ij,bj->b", spins, mat, spins)
    winner = objective.argmax()
    # Recover the bits with the aux-spin convention of AbstractBoltzmannMachine.sample.
    row = spins[winner]
    return ((row[:-1] * row[[-1]] + 1.0) / 2.0).float()


class TestBoltzmannMachineIsingConversion(unittest.TestCase):
    """The Ising matrix optimum must attain the brute-force energy minimum."""

    def test_conversion_attains_energy_minimum(self):
        for seed in range(10):
            with self.subTest(seed=seed):
                torch.manual_seed(seed)
                model = BM(6)
                with torch.no_grad():
                    model.quadratic_coef.copy_(torch.randn(6, 6))
                    model.linear_bias.copy_(torch.randn(6) * 0.5)
                states = enumerate_bits(model.num_nodes)
                with torch.no_grad():
                    energies = model(states)
                solved = best_ising_bits(model.get_ising_matrix())
                solved_energy = model(solved.unsqueeze(0)).item()
                self.assertAlmostEqual(
                    solved_energy,
                    energies.min().item(),
                    places=4,
                    msg=f"seed {seed}: Ising optimum energy {solved_energy} != "
                    f"brute-force minimum {energies.min().item()}",
                )

    def test_ising_matrix_is_symmetric_with_zero_diagonal(self):
        torch.manual_seed(101)
        model = BM(5)
        mat = model.get_ising_matrix()
        np.testing.assert_allclose(mat, mat.T, atol=1e-7)
        self.assertTrue(np.allclose(np.diag(mat), 0.0, atol=1e-7))

    def test_hidden_conditioning_attains_conditional_minimum(self):
        for seed in range(10):
            with self.subTest(seed=seed):
                torch.manual_seed(200 + seed)
                model = BM(7)
                with torch.no_grad():
                    model.quadratic_coef.copy_(torch.randn(7, 7))
                    model.linear_bias.copy_(torch.randn(7) * 0.5)
                s_visible = torch.bernoulli(torch.full((1, 3), 0.5))
                hidden = enumerate_bits(model.num_nodes - 3)
                states = torch.cat([s_visible.expand(hidden.shape[0], -1), hidden], dim=-1)
                with torch.no_grad():
                    energies = model(states)
                sub_mat = model._hidden_to_ising_matrix(s_visible[0])
                solved_hidden = best_ising_bits(sub_mat)
                solved_state = torch.cat([s_visible, solved_hidden.unsqueeze(0)], dim=-1)
                solved_energy = model(solved_state).item()
                self.assertAlmostEqual(
                    solved_energy,
                    energies.min().item(),
                    places=4,
                    msg=f"seed {200 + seed}: conditional Ising optimum {solved_energy} != "
                    f"brute-force minimum {energies.min().item()}",
                )


class TestGaussianBernoulliRBMIsingConversion(unittest.TestCase):
    """The Bernoulli-partition Ising matrix must encode the marginal free energy.

    Completing the square in the Gaussian partition of
    ``GaussianBernoulliRestrictedBoltzmannMachine.energy`` gives the marginal
    Hamiltonian ``E(s) = -0.5 s' W' diag(1/var) W s - s'(W'(mu/var) + b) s + const``
    up to an s-independent constant, which is the objective the conversion
    must encode.
    """

    def test_conversion_attains_marginal_minimum(self):
        for seed in range(10):
            with self.subTest(seed=seed):
                torch.manual_seed(300 + seed)
                model = GaussianBernoulliRestrictedBoltzmannMachine(4, 3)
                with torch.no_grad():
                    model.quadratic_coef.copy_(torch.randn(4, 3))
                    model.linear_bias.copy_(torch.randn(3) * 0.5)
                    model.mu.copy_(torch.randn(4) * 0.5)
                    model.log_var.copy_(torch.randn(4))
                mu, var, weight, bias = (
                    model.mu,
                    model.var,
                    model.quadratic_coef,
                    model.linear_bias,
                )
                states = enumerate_bits(model.num_bernoulli)
                with torch.no_grad():
                    linear = states @ (weight.t() @ (mu / var) + bias)
                    quadratic = torch.einsum(
                        "bi,ij,bj->b", states, weight.t() / var @ weight, states
                    )
                    marginals = -linear - 0.5 * quadratic
                solved = best_ising_bits(model.get_ising_matrix())
                solved_linear = solved @ (weight.t() @ (mu / var) + bias)
                solved_quadratic = torch.einsum(
                    "i,ij,j->", solved, weight.t() / var @ weight, solved
                )
                solved_energy = (-solved_linear - 0.5 * solved_quadratic).item()
                self.assertAlmostEqual(
                    solved_energy,
                    marginals.min().item(),
                    places=4,
                    msg=f"seed {300 + seed}: GBRBM Ising optimum {solved_energy} != "
                    f"brute-force minimum {marginals.min().item()}",
                )


if __name__ == "__main__":
    unittest.main()
