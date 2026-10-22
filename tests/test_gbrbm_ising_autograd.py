"""GBRBM solver conversion must not retain unused backward tensors."""
import itertools

import numpy as np
import pytest
import torch

from kaiwu.torch_plugin.gbrbm import GaussianBernoulliRestrictedBoltzmannMachine


def model(gaussian_visible):
    torch.manual_seed(17)
    result = GaussianBernoulliRestrictedBoltzmannMachine(
        5 if gaussian_visible else 3, 3 if gaussian_visible else 5,
        is_visible_gaussian=gaussian_visible, device='cpu')
    with torch.no_grad():
        result.mu.copy_(torch.linspace(-.4, .4, 5))
        result.log_var.copy_(torch.linspace(-.3, .3, 5))
        result.quadratic_coef.copy_(torch.arange(15).reshape(5, 3) / 30 - .2)
        result.linear_bias.copy_(torch.tensor([-.1, .2, .3]))
    return result


@pytest.mark.parametrize('gaussian_visible', [True, False])
@pytest.mark.parametrize('entry', ['get_ising_matrix', 'sample'])
def test_conversion_without_graph_preserves_energies_and_existing_gradients(gaussian_visible, entry):
    machine = model(gaussian_visible)
    before = {key: value.clone() for key, value in machine.state_dict().items()}
    for parameter in machine.parameters():
        parameter.grad = torch.ones_like(parameter)
    gradients = [p.grad.clone() for p in machine.parameters()]
    saved = []

    class Solver:
        def solve(self, matrix):
            assert torch.is_grad_enabled()  # Conversion must restore caller mode.
            self.matrix = matrix
            return np.ones((2, len(matrix)), dtype=np.float32)

    solver = Solver()
    with torch.enable_grad(), torch.autograd.graph.saved_tensors_hooks(
        lambda tensor: (saved.append(tuple(tensor.shape)) or tensor), lambda tensor: tensor
    ):
        if entry == 'sample':
            samples = machine.sample(solver)
            assert samples.shape == (2, 8)
            assert not samples.requires_grad
            matrix = solver.matrix
        else:
            matrix = machine.get_ising_matrix()
        assert torch.is_grad_enabled()
    assert saved == []
    for key, value in machine.state_dict().items():
        torch.testing.assert_close(value, before[key])
    for parameter, gradient in zip(machine.parameters(), gradients):
        torch.testing.assert_close(parameter.grad, gradient)

    # Independent Gaussian-integrated marginal over Bernoulli assignments.
    mu = machine.mu.detach().numpy()
    variance = np.exp(machine.log_var.detach().numpy())
    weights = machine.quadratic_coef.detach().numpy()
    bias = machine.linear_bias.detach().numpy()
    marginal = []
    ising = []
    for bits in itertools.product([0., 1.], repeat=3):
        z = np.asarray(bits)
        displacement = weights @ z
        marginal.append(-bias @ z - np.sum((mu * displacement + .5 * displacement**2) / variance))
        spins = np.r_[2 * z - 1, 1]
        ising.append(-spins @ matrix @ spins)
    np.testing.assert_allclose(np.array(ising) - ising[0], np.array(marginal) - marginal[0],
                               rtol=1e-5, atol=1e-7)


def test_energy_training_still_backpropagates_after_conversion():
    machine = model(True)
    states = torch.randn(2, 8, requires_grad=True)
    machine.get_ising_matrix()
    loss = machine(states).sum()
    loss.backward()
    assert states.grad is not None and torch.isfinite(states.grad).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in machine.parameters())
