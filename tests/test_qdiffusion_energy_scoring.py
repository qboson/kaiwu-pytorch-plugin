"""EnergyModel must score sigmoid-conditioned, per-sample mean BM energies."""

import numpy as np
import torch

from kaiwu.torch_plugin.qdiffusion import EnergyModel


class TwoRowSampler:
    """Returns two all-plus spin rows for any Ising matrix."""

    def solve(self, ising_mat):
        """Returns two solution rows."""
        return np.ones((2, ising_mat.shape[0]))


def _model():
    return EnergyModel(bm_num_visible=2, bm_num_hidden=3, sampler=TwoRowSampler())


def test_visible_logits_are_sigmoid_conditions():
    model = _model()
    logits = torch.tensor([[0.0, 100.0, -100.0]])

    assert torch.allclose(model.discretize_visible_state(logits), torch.sigmoid(logits))


def test_scoring_is_the_mean_energy_of_the_conditioned_completions():
    model = _model()
    logits = torch.zeros(2, 2)

    full_states, split_sizes = model.sample_hidden_state(
        model.discretize_visible_state(logits)
    )
    energies = model.energy_bm(full_states)
    expected = torch.stack(
        [group.mean() for group in energies.split(split_sizes.tolist())]
    )

    scores = model.score_visible_logits(logits)

    assert scores.shape == (2, 1)
    torch.testing.assert_close(scores[:, 0], expected)
