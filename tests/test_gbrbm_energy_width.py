"""GBRBM energy paths must reject state tensors of the wrong width."""

from __future__ import annotations

import pytest
import torch

from kaiwu.torch_plugin.gbrbm import GaussianBernoulliRestrictedBoltzmannMachine

NUM_GAUSSIAN = 3
NUM_BERNOULLI = 2
NUM_NODES = NUM_GAUSSIAN + NUM_BERNOULLI


def _model() -> GaussianBernoulliRestrictedBoltzmannMachine:
    return GaussianBernoulliRestrictedBoltzmannMachine(
        num_visible=NUM_GAUSSIAN, num_hidden=NUM_BERNOULLI
    )


@pytest.mark.parametrize("width", [1, NUM_NODES - 1, NUM_NODES + 1])
def test_energy_rejects_wrong_width(width: int) -> None:
    with pytest.raises(ValueError, match="s_all must have shape"):
        _model().energy(torch.randn(2, width))


@pytest.mark.parametrize("width", [1, NUM_GAUSSIAN - 1, NUM_GAUSSIAN + 1])
def test_marginal_energy_rejects_wrong_width(width: int) -> None:
    with pytest.raises(ValueError, match="s_gaussian must have shape"):
        _model().marginal_energy(torch.randn(2, width))


def test_correct_widths_still_work() -> None:
    model = _model()
    assert model.energy(torch.rand(2, NUM_NODES)).shape == (2,)
    assert model.marginal_energy(torch.rand(2, NUM_GAUSSIAN)).shape == (2,)
