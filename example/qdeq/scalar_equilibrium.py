"""Check a scalar QDEQ equilibrium and gradient against closed-form values."""

from itertools import product
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from kaiwu.torch_plugin import QDEQ, QDEQConfig


def solve_ising(matrix):
    """Return a maximizing spin row with the final auxiliary spin fixed."""
    num_free = matrix.shape[0] - 1
    if num_free > 12:
        raise ValueError("Exhaustive enumeration is only for tiny Ising problems.")
    # A global sign flip leaves s.T @ matrix @ s unchanged.
    spins = np.array([(*row, 1) for row in product((-1, 1), repeat=num_free)])
    energies = np.einsum("bi,ij,bj->b", spins, matrix, spins)
    return spins[np.argmax(energies)][None, :]


class ScalarLayer(nn.Module):
    """The scalar map f(z, x) = a*z + x, with a trainable coefficient."""

    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.tensor(0.5, dtype=torch.float64))

    def forward(self, z, x):
        """Evaluate the shared map at z."""
        return self.a * z + x


def main():
    """Check the equilibrium and one regularized optimizer step on CPU."""
    layer = ScalarLayer()
    config = QDEQConfig(
        f_thres=4,
        f_eps=1e-10,
        b_thres=40,
        b_eps=1e-9,
        anderson_lam=1e-10,
        num_basis=1,
        coeff_bits=2,
        jac_loss_vecs=1,
    )
    model = QDEQ(layer, SimpleNamespace(solve=solve_ising), config)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    x = torch.tensor([[1.5]], dtype=torch.float64)
    target = torch.ones_like(x)

    optimizer.zero_grad()
    prediction = model(x)
    expected = x / (1 - layer.a.detach())
    torch.testing.assert_close(prediction, expected)

    # The Hutchinson probe is random. In one dimension its square is the
    # factor multiplying a**2, so the sampled penalty also gives its slope.
    penalty = model.last_jac_loss
    task_loss = 0.5 * (prediction - target).square().mean()
    penalty_weight = 0.1
    loss = task_loss + penalty_weight * penalty
    loss.backward()

    coefficient = layer.a.detach().clone()
    task_gradient = ((expected - target) * x / (1 - coefficient).square()).mean()
    expected_gradient = task_gradient + penalty_weight * 2 * penalty.detach() / coefficient
    torch.testing.assert_close(layer.a.grad, expected_gradient, rtol=1e-6, atol=1e-8)
    optimizer.step()
    torch.testing.assert_close(layer.a.detach(), coefficient - 0.01 * expected_gradient)

    stats = model.get_last_stats()
    print(f"Equilibrium: {prediction.item():.6f}")
    print(f"Forward residual: {stats['forward_residual']:.3g}")
    print(f"Backward residual: {stats['backward_residual']:.3g}")
    print("Analytic equilibrium, regularized gradient, and SGD update agree.")


if __name__ == "__main__":
    main()
