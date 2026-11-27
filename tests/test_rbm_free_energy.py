"""Check RBM free energy against the existing Hamiltonian."""

import itertools

import pytest
import torch

from kaiwu.torch_plugin import RestrictedBoltzmannMachine


def test_free_energy_matches_exhaustive_hidden_marginal_and_its_gradient():
    model = RestrictedBoltzmannMachine(
        2,
        3,
        quadratic_coef=torch.tensor(
            [[0.2, -0.3, 1.1], [1.3, 0.7, -0.2]], dtype=torch.float64
        ),
        linear_bias=torch.tensor([0.1, -0.5, -0.9, 0.2, 0.4], dtype=torch.float64),
        device="cpu",
    )
    visible = torch.tensor([[0, 0], [1, 0], [0, 1], [1, 1]])
    hidden = torch.tensor(list(itertools.product((0, 1), repeat=3)))
    states = torch.cat(
        (visible.repeat_interleave(len(hidden), dim=0), hidden.repeat(len(visible), 1)), dim=1
    ).to(torch.float64)
    expected = -torch.logsumexp(-model(states).reshape(len(visible), -1), dim=1)

    actual = model.free_energy(visible)

    torch.testing.assert_close(actual, expected)
    actual_grads = torch.autograd.grad(actual.sum(), tuple(model.parameters()))
    expected_grads = torch.autograd.grad(expected.sum(), tuple(model.parameters()))
    for grad, reference in zip(actual_grads, expected_grads):
        torch.testing.assert_close(grad, reference)


def test_free_energy_rejects_state_width_mismatch_and_handles_large_logits():
    model = RestrictedBoltzmannMachine(
        2, 2, quadratic_coef=torch.full((2, 2), 1000.0), device="cpu"
    )
    torch.testing.assert_close(model.free_energy(torch.ones(2)), torch.tensor(-4000.0))
    with pytest.raises(ValueError, match="num_visible"):
        model.free_energy(torch.ones(2, 3))
