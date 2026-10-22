"""QDEQ must support tensors with more than two dimensions."""

import numpy as np
import torch

from kaiwu.torch_plugin import QDEQ, QDEQConfig


class LocalSearchSampler:
    """Deterministic spin sampler that needs no license."""

    def __init__(self, restarts=4, seed=0):
        self.restarts = restarts
        self.seed = seed

    def solve(self, ising_mat):
        """Returns one locally optimal spin row per restart."""
        rng = np.random.default_rng(self.seed)
        coupling = (ising_mat + ising_mat.T) / 2
        np.fill_diagonal(coupling, 0.0)
        rows = []
        for _ in range(self.restarts):
            spins = rng.choice([-1.0, 1.0], size=coupling.shape[0])
            field = coupling @ spins
            while True:
                gain = -4.0 * spins * field
                index = int(np.argmax(gain))
                if gain[index] <= 1e-12:
                    break
                spins[index] = -spins[index]
                field += 2.0 * spins[index] * coupling[:, index]
            rows.append(spins.copy())
        return np.stack(rows)


class ElementwiseContraction(torch.nn.Module):
    """``layer(z, x) = z / 2 + x`` for tensors of any shape."""

    def forward(self, z, x):
        """Returns ``z / 2 + x``."""
        return 0.5 * z + x


def _model():
    config = QDEQConfig(f_thres=40, f_eps=1e-6, num_basis=4, coeff_bits=4)
    return QDEQ(ElementwiseContraction(), LocalSearchSampler(), config)


def test_three_dimensional_inputs_reach_the_fixed_point():
    x = torch.randn(2, 3, 4, dtype=torch.float64)
    deq = _model().eval()

    out = deq(x)

    assert out.shape == x.shape
    assert deq.get_last_stats()["forward_residual"] < 1e-2
    torch.testing.assert_close(out, 2.0 * x, rtol=1e-2, atol=1e-2)


def test_three_dimensional_inputs_get_implicit_gradients():
    x = torch.randn(2, 3, 4, dtype=torch.float64, requires_grad=True)
    deq = _model()

    deq(x).sum().backward()

    torch.testing.assert_close(x.grad, torch.full_like(x, 2.0), rtol=1e-2, atol=1e-2)
