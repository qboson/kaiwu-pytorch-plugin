"""Boltzmann machines must validate their node counts at construction."""

import pytest
import torch

from kaiwu.torch_plugin import BoltzmannMachine, RestrictedBoltzmannMachine
from kaiwu.torch_plugin.gbrbm import GaussianBernoulliRestrictedBoltzmannMachine

FACTORIES = [
    lambda value: BoltzmannMachine(num_nodes=value),
    lambda value: RestrictedBoltzmannMachine(num_visible=value, num_hidden=3),
    lambda value: RestrictedBoltzmannMachine(num_visible=3, num_hidden=value),
    lambda value: GaussianBernoulliRestrictedBoltzmannMachine(
        num_visible=value, num_hidden=3
    ),
    lambda value: GaussianBernoulliRestrictedBoltzmannMachine(
        num_visible=3, num_hidden=value
    ),
]


@pytest.mark.parametrize("factory", FACTORIES)
@pytest.mark.parametrize("value", [0, -2])
def test_non_positive_node_count_is_rejected(factory, value) -> None:
    with pytest.raises(ValueError, match="must be a positive integer"):
        factory(value)


@pytest.mark.parametrize("factory", FACTORIES)
@pytest.mark.parametrize("value", [2.5, True])
def test_non_integer_node_count_is_rejected(factory, value) -> None:
    with pytest.raises(TypeError, match="must be a positive integer"):
        factory(value)


def test_valid_node_counts_build_parameters() -> None:
    bm = BoltzmannMachine(num_nodes=4)
    assert bm.quadratic_coef.shape == (4, 4)

    rbm = RestrictedBoltzmannMachine(num_visible=3, num_hidden=2)
    assert rbm.quadratic_coef.shape == (3, 2)
    assert rbm.linear_bias.shape == (5,)

    gb = GaussianBernoulliRestrictedBoltzmannMachine(num_visible=3, num_hidden=2)
    assert gb.num_nodes == 5
