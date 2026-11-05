"""BoltzmannMachine.condition_sample must accept an empty visible batch."""

import numpy as np
import torch

from kaiwu.torch_plugin import BoltzmannMachine


class _Sampler:  # pylint: disable=too-few-public-methods
    def solve(self, ising_matrix):
        return np.ones((1, ising_matrix.shape[0]), dtype=np.float32)


def test_empty_batch_returns_empty_states() -> None:
    sampled = BoltzmannMachine(num_nodes=4).condition_sample(
        _Sampler(), torch.zeros(0, 3)
    )
    assert sampled.shape == (0, 4)


def test_non_empty_batch_still_samples() -> None:
    sampled = BoltzmannMachine(num_nodes=4).condition_sample(
        _Sampler(), torch.zeros(1, 3)
    )
    assert sampled.shape == (1, 4)
