"""Conditional solver samples preserve observations in arbitrary node positions."""

from itertools import product

import numpy as np
import pytest
import torch

from kaiwu.torch_plugin import BoltzmannMachine


class ExactSampler:
    """Maximize the small SDK Ising objective with the reference spin fixed to 1."""

    def solve(self, matrix):
        states = np.array([(*state, 1) for state in product((-1, 1), repeat=len(matrix) - 1)])
        scores = np.einsum("ai,ij,aj->a", states, matrix, states)
        return states[np.argsort(-scores, kind="stable")]


def make_model():
    return BoltzmannMachine(
        4,
        linear_bias=torch.tensor([0.1, -0.4, 0.2, -0.5], dtype=torch.float64),
        quadratic_coef=torch.tensor(
            [
                [0.0, 1.2, -0.7, 0.4],
                [0.0, 0.0, 0.9, -1.5],
                [0.0, 0.0, 0.0, 0.6],
                [0.0, 0.0, 0.0, 0.0],
            ],
            dtype=torch.float64,
        ),
    )


def test_nonprefix_observations_match_exhaustive_model_energies():
    model = make_model()
    observations = torch.tensor([[1., 0.], [0., 1.]], dtype=torch.float64)
    samples = model.condition_sample(ExactSampler(), observations, visible_indices=[3, 0])
    full_states = torch.tensor(list(product((0., 1.), repeat=4)), dtype=torch.float64)

    for index, observed in enumerate(observations):
        matching = full_states[(full_states[:, [3, 0]] == observed).all(dim=1)]
        expected = matching[model(matching).argsort()]
        torch.testing.assert_close(samples[index * 4 : (index + 1) * 4], expected)


def test_explicit_prefix_matches_default_and_can_condition_on_no_columns():
    model = make_model()
    sampler = ExactSampler()
    observations = torch.ones(1, 2, dtype=torch.float64)
    torch.testing.assert_close(
        model.condition_sample(sampler, observations, visible_indices=(0, 1)),
        model.condition_sample(sampler, observations),
    )
    none_observed = torch.empty(1, 0, dtype=torch.float64)
    samples = model.condition_sample(sampler, none_observed, visible_indices=[])
    assert samples.shape == (16, 4)
    torch.testing.assert_close(model(samples)[0], model(samples).min())


def test_all_observed_values_can_be_supplied_in_any_order():
    model = make_model()
    result = model.condition_sample(
        ExactSampler(), torch.tensor([[0, 1, 1, 0]]), visible_indices=[3, 2, 0, 1]
    )
    torch.testing.assert_close(result, torch.tensor([[1., 0., 1., 0.]], dtype=torch.float64))


@pytest.mark.parametrize(
    "indices, error", [([0], "one index"), ([0, 0], "distinct"),
                        ([0, 4], "valid node"), ([-1, 0], "valid node"),
                        ([0., 1.], "integers"), ([True, False], "integers")]
)
def test_bad_indices_fail_before_solver(indices, error):
    with pytest.raises(ValueError, match=error):
        make_model().condition_sample(None, torch.zeros(1, 2), visible_indices=indices)
