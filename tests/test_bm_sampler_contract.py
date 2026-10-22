"""BoltzmannMachine.sample must validate the sampler interface."""

import numpy as np
import pytest
import torch

from kaiwu.torch_plugin import BoltzmannMachine


class _Sampler:  # pylint: disable=too-few-public-methods
    def solve(self, ising_matrix):
        return np.ones((1, ising_matrix.shape[0]), dtype=np.float32)


@pytest.mark.parametrize("sampler", [None, object()])
def test_sampler_without_solve_is_rejected(sampler) -> None:
    with pytest.raises(TypeError, match="sampler must provide a solve"):
        BoltzmannMachine(num_nodes=4).sample(sampler)


def test_valid_sampler_still_samples() -> None:
    assert BoltzmannMachine(num_nodes=4).sample(_Sampler()).shape == (1, 4)
