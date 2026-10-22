"""Pin the split sizes produced by EnergyModel.sample_hidden_state."""

import numpy as np
import torch

from kaiwu.torch_plugin import EnergyModel


class _Sampler:  # pylint: disable=too-few-public-methods
    def solve(self, ising_matrix):
        return np.ones((2, ising_matrix.shape[0]), dtype=np.float32)


def test_split_sizes_match_the_sampler_output() -> None:
    model = EnergyModel(bm_num_visible=2, bm_num_hidden=2, sampler=_Sampler())
    visible = torch.tensor([[0.1, 0.9], [0.5, 0.6], [0.2, 0.3]])

    full_states, split_sizes = model.sample_hidden_state(visible)

    assert full_states.shape == (6, 4)
    assert split_sizes.tolist() == [2, 2, 2]
    torch.testing.assert_close(full_states[:, :2], visible.repeat_interleave(2, dim=0))
