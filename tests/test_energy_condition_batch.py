"""Batch conditioned sampling without losing per-condition solution groups."""

import os
import sys
from unittest.mock import patch

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from kaiwu.torch_plugin import BoltzmannMachine, EnergyModel


class VariableSampler:
    """Return legal Ising solutions with a different count per condition."""

    def __init__(self):
        self.responses = iter((
            [[-1, 1, 1]],
            [[1, 1, 1], [1, -1, -1], [-1, -1, 1]],
            [[1, -1, 1], [-1, 1, -1]],
        ))

    def solve(self, matrix):
        assert matrix.shape == (3, 3)
        return np.asarray(next(self.responses), dtype=np.float32)


@pytest.mark.parametrize("return_counts", [False, True])
def test_condition_sample_preserves_variable_solution_groups(return_counts):
    model = BoltzmannMachine(4, device="cpu")
    visible = torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
    kwargs = {"return_counts": True} if return_counts else {}

    result = model.condition_sample(VariableSampler(), visible, **kwargs)

    if return_counts:
        states, counts = result
        assert counts == [1, 3, 2]
    else:
        states = result
        assert isinstance(states, torch.Tensor)
    torch.testing.assert_close(states, torch.tensor([
        [0.1, 0.2, 0., 1.],
        [0.3, 0.4, 1., 1.],
        [0.3, 0.4, 0., 1.],
        [0.3, 0.4, 0., 0.],
        [0.5, 0.6, 1., 0.],
        [0.5, 0.6, 1., 0.],
    ]))


def test_energy_batches_conditions_and_preserves_group_means_and_gradients():
    model = EnergyModel(bm_num_visible=2, bm_num_hidden=2, sampler=VariableSampler())
    with torch.no_grad():
        model.energy_bm.quadratic_coef.zero_()
        model.energy_bm.linear_bias.copy_(torch.tensor([0.7, -0.2, 0.4, -0.3]))
    logits = torch.zeros(3, 2, requires_grad=True)

    with patch.object(
        model.energy_bm, "condition_sample", wraps=model.energy_bm.condition_sample
    ) as sample:
        energy = model.score_visible_logits(logits)

    assert sample.call_count == 1
    torch.testing.assert_close(energy, torch.tensor([[0.05], [-0.55 / 3], [-0.65]]))
    energy.sum().backward()
    torch.testing.assert_close(logits.grad, torch.tensor([[-0.175, 0.05]]).expand(3, -1))
    torch.testing.assert_close(
        model.energy_bm.linear_bias.grad, torch.tensor([-1.5, -1.5, -4 / 3, -5 / 3])
    )
