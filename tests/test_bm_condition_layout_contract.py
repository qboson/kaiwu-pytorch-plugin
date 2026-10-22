"""Pin the concatenation order of BoltzmannMachine.condition_sample."""

import numpy as np
import torch

from kaiwu.torch_plugin import BoltzmannMachine


class _OneSampler:  # pylint: disable=too-few-public-methods
    def solve(self, ising_matrix):
        return np.ones((1, ising_matrix.shape[0]), dtype=np.float32)


def test_sampled_units_are_appended_after_the_clamped_state() -> None:
    bm = BoltzmannMachine(num_nodes=4)
    s_visible = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])

    sampled = bm.condition_sample(_OneSampler(), s_visible)

    assert sampled.shape == (2, 4)
    torch.testing.assert_close(sampled[:, :3], s_visible)
    # All-ones spins decode to binary ones in the sampled tail.
    torch.testing.assert_close(sampled[:, 3:], torch.ones(2, 1))
