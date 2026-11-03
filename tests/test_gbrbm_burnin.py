"""GBRBM burn-in contracts checked against independently generated transitions."""
from itertools import product
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

SOURCE = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SOURCE))
import kaiwu

kaiwu.__path__.insert(0, str(SOURCE / "kaiwu"))
from kaiwu.torch_plugin import gbrbm

assert Path(gbrbm.__file__).resolve() == (
    SOURCE / "kaiwu" / "torch_plugin" / "gbrbm.py"
).resolve()


class FixedSampler:
    """Return both auxiliary-spin gauges without changing the torch RNG."""

    def __init__(self):
        self.calls = 0

    def solve(self, ising_matrix):
        """Provide Bernoulli states [1, 0] and [0, 1] through the SDK interface."""
        assert ising_matrix.shape == (3, 3)
        self.calls += 1
        return np.array([[1, -1, 1], [1, -1, -1]], dtype=np.float32)


def make_model(gaussian_visible):
    """Use one Gaussian and two Bernoulli nodes in either visible layout."""
    model = gbrbm.GaussianBernoulliRestrictedBoltzmannMachine(
        1 if gaussian_visible else 2,
        2 if gaussian_visible else 1,
        is_visible_gaussian=gaussian_visible,
        device="cpu",
    )
    with torch.no_grad():
        model.mu.fill_(0.25)
        model.log_var.copy_(torch.log(torch.tensor([0.75])))
        model.quadratic_coef.copy_(torch.tensor([[0.8, -0.55]]))
        model.linear_bias.copy_(torch.tensor([-0.3, 0.4]))
    return model


def initial_arguments(start):
    """Construct every supported initialization without consuming torch randomness."""
    if start == "gaussian":
        return {"s_gaussian": torch.tensor([[-3.0], [2.0]])}
    if start == "bernoulli":
        return {"s_bernoulli": torch.tensor([[1.0, 0.0], [0.0, 1.0]])}
    if start == "sampler":
        return {"sampler": FixedSampler()}
    return {"n_sample": 2}


def independent_chain(model, n_step, start, arguments):
    """Enumerate Bernoulli energies and complete the Gaussian energy square.

    No infer_* or gibbs_sample calls are used. Enumerating all four hidden
    states gives the Bernoulli conditional marginals. Completing the energy
    square gives Gaussian mean mu + W*h and variance exp(log_var).
    """
    hidden_states = torch.tensor(list(product((0.0, 1.0), repeat=2)))
    if start == "gaussian":
        gaussian = arguments["s_gaussian"].clone()
    elif start == "random":
        gaussian = torch.randn(2, 1)
    else:
        hidden = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

    def draw_hidden(gaussian):
        observed = gaussian[:, None, :].expand(-1, len(hidden_states), -1)
        candidates = hidden_states[None, :, :].expand(len(gaussian), -1, -1)
        all_states = torch.cat((observed, candidates), dim=-1)
        energies = model.energy(all_states.reshape(-1, model.num_nodes))
        probabilities = torch.softmax(-energies.reshape(len(gaussian), -1), dim=-1)
        conditional_means = probabilities @ hidden_states
        return (conditional_means > torch.rand_like(conditional_means)).float()

    def draw_gaussian(hidden):
        conditional_means = model.mu + hidden @ model.quadratic_coef.t()
        return conditional_means + torch.randn_like(conditional_means) * model.var.sqrt()

    chain = []
    with torch.no_grad():
        for _ in range(n_step):
            if start in ("gaussian", "random"):
                hidden = draw_hidden(gaussian)
                gaussian = draw_gaussian(hidden)
            else:
                gaussian = draw_gaussian(hidden)
                hidden = draw_hidden(gaussian)
            chain.append(torch.cat((gaussian, hidden), dim=-1))
    return chain


@pytest.mark.parametrize("gaussian_visible", [True, False])
@pytest.mark.parametrize("start", ["gaussian", "bernoulli", "sampler", "random"])
@pytest.mark.parametrize("burn_in", [0, 1, 2, 4])
def test_burn_in_discards_exactly_the_requested_transitions(
    gaussian_visible, start, burn_in
):
    """A burn-in of N drops complete transitions 1 through N in every chain."""
    model = make_model(gaussian_visible)
    arguments = initial_arguments(start)
    torch.manual_seed(47)
    actual = model.gibbs_sample(n_step=5, n_burnin=burn_in, **arguments)
    actual_rng_state = torch.get_rng_state()
    torch.manual_seed(47)
    chain = independent_chain(model, 5, start, arguments)
    expected = torch.cat(chain[burn_in:])

    assert actual.shape == (2 * (5 - burn_in), model.num_nodes)
    assert not actual.requires_grad
    torch.testing.assert_close(actual, expected)
    assert torch.equal(actual_rng_state, torch.get_rng_state())
    if start == "sampler":
        assert arguments["sampler"].calls == 1


@pytest.mark.parametrize("gaussian_visible", [True, False])
@pytest.mark.parametrize("start", ["gaussian", "bernoulli", "sampler", "random"])
@pytest.mark.parametrize("n_step,burn_in", [(5, 5), (5, 6), (0, 0)])
def test_no_retained_transitions_returns_an_empty_state_batch(
    gaussian_visible, start, n_step, burn_in
):
    """Discarding all transitions yields no rows without altering chain execution."""
    model = make_model(gaussian_visible)
    arguments = initial_arguments(start)
    torch.manual_seed(47)
    actual = model.gibbs_sample(n_step=n_step, n_burnin=burn_in, **arguments)
    actual_rng_state = torch.get_rng_state()
    torch.manual_seed(47)
    independent_chain(model, n_step, start, arguments)

    assert actual.shape == (0, model.num_nodes)
    assert actual.dtype == model.mu.dtype
    assert actual.device == model.mu.device
    assert not actual.requires_grad
    assert torch.equal(actual_rng_state, torch.get_rng_state())
    if start == "sampler":
        assert arguments["sampler"].calls == 1
