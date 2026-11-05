"""Boltzmann machines must validate the sampler's spin-matrix width."""

import numpy as np
import pytest
import torch

from kaiwu.torch_plugin import BoltzmannMachine
from kaiwu.torch_plugin.gbrbm import GaussianBernoulliRestrictedBoltzmannMachine


class _GoodSampler:  # pylint: disable=too-few-public-methods
    def solve(self, ising_matrix):
        return np.ones((1, ising_matrix.shape[0]), dtype=np.float32)


class _NarrowSampler:  # pylint: disable=too-few-public-methods
    def solve(self, ising_matrix):
        return np.ones((1, ising_matrix.shape[0] - 2), dtype=np.float32)


class _OneDimensionalSampler:  # pylint: disable=too-few-public-methods
    def solve(self, ising_matrix):
        return np.ones(ising_matrix.shape[0], dtype=np.float32)


@pytest.mark.parametrize("sampler", [_NarrowSampler(), _OneDimensionalSampler()])
def test_sample_rejects_sampler_with_wrong_width(sampler) -> None:
    with pytest.raises(ValueError, match="sampler must return"):
        BoltzmannMachine(num_nodes=4).sample(sampler)


def test_condition_sample_rejects_sampler_with_wrong_width() -> None:
    with pytest.raises(ValueError, match="sampler must return"):
        BoltzmannMachine(num_nodes=4).condition_sample(
            _NarrowSampler(), torch.zeros(1, 3)
        )


def test_gbrbm_sample_rejects_sampler_with_wrong_width() -> None:
    model = GaussianBernoulliRestrictedBoltzmannMachine(num_visible=3, num_hidden=2)
    with pytest.raises(ValueError, match="sampler must return"):
        model.sample(_NarrowSampler())


def test_valid_sampler_output_still_works() -> None:
    bm = BoltzmannMachine(num_nodes=4)
    assert bm.sample(_GoodSampler()).shape == (1, 4)
    assert bm.condition_sample(_GoodSampler(), torch.zeros(1, 3)).shape[0] == 1
