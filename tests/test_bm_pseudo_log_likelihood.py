"""Compare the BM's conditional score against explicit one-bit flips."""

import pytest
import torch

from kaiwu.torch_plugin import BoltzmannMachine


def test_pseudo_log_likelihood_uses_the_models_energy_differences():
    model = BoltzmannMachine(
        3,
        quadratic_coef=torch.tensor(
            [[5.0, -0.3, 0.1], [20.0, 7.0, 1.1], [30.0, 40.0, 9.0]],
            dtype=torch.float64,
        ),
        linear_bias=torch.tensor([0.5, -0.7, 0.2], dtype=torch.float64),
        device="cpu",
    )
    states = torch.tensor([[0, 0, 1], [1, 1, 1], [0, 1, 0]])
    scores = -model(states.double())
    conditionals = []
    for index in range(states.shape[1]):
        flipped = states.clone()
        flipped[:, index] = 1 - flipped[:, index]
        conditionals.append(scores - torch.logaddexp(scores, -model(flipped.double())))
    expected = torch.stack(conditionals).sum(dim=0)

    actual = model.pseudo_log_likelihood(states)

    torch.testing.assert_close(actual, expected)
    actual.sum().backward()
    assert torch.isfinite(model.quadratic_coef.grad).all()
    assert torch.isfinite(model.linear_bias.grad).all()


def test_pseudo_log_likelihood_is_finite_for_unlikely_states():
    model = BoltzmannMachine(
        2, quadratic_coef=torch.zeros(2, 2), linear_bias=torch.tensor([1000.0, -1000.0]),
        device="cpu",
    )
    torch.testing.assert_close(
        model.pseudo_log_likelihood(torch.tensor([False, True])), torch.tensor(-2000.0)
    )
    with pytest.raises(ValueError, match="binary 0/1"):
        model.pseudo_log_likelihood(torch.tensor([0.5, 1.0]))
