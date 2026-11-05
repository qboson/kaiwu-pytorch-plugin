"""GBRBM state inputs must match the Gaussian/Bernoulli unit counts."""

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


@pytest.mark.parametrize("width", [1, NUM_GAUSSIAN - 1, NUM_GAUSSIAN + 1])
def test_gibbs_sample_rejects_wrong_gaussian_width(width: int) -> None:
    model = _model()
    with pytest.raises(ValueError, match="s_gaussian must have shape"):
        model.gibbs_sample(n_step=1, s_gaussian=torch.randn(2, width))


@pytest.mark.parametrize("width", [1, NUM_BERNOULLI - 1, NUM_BERNOULLI + 1])
def test_gibbs_sample_rejects_wrong_bernoulli_width(width: int) -> None:
    model = _model()
    with pytest.raises(ValueError, match="s_bernoulli must have shape"):
        model.gibbs_sample(n_step=1, s_bernoulli=torch.randn(2, width))


def test_infer_from_gaussian_rejects_wrong_width() -> None:
    model = _model()
    with pytest.raises(ValueError, match="s_gaussian must have shape"):
        model.infer_from_gaussian(torch.randn(2, 1), binarize=False)


def test_infer_from_bernoulli_rejects_wrong_width() -> None:
    model = _model()
    with pytest.raises(ValueError, match="s_bernoulli must have shape"):
        model.infer_from_bernoulli(torch.randn(2, 1), no_random=True)


def test_one_dimensional_input_is_rejected() -> None:
    model = _model()
    with pytest.raises(ValueError, match="must have shape"):
        model.infer_from_gaussian(torch.randn(NUM_GAUSSIAN), binarize=False)


def test_correct_widths_still_work() -> None:
    model = _model()
    visible = torch.randn(2, NUM_GAUSSIAN)
    bernoulli = torch.rand(2, NUM_BERNOULLI)
    assert model.gibbs_sample(n_step=1, s_gaussian=visible).shape == (2, NUM_NODES)
    assert model.gibbs_sample(n_step=1, s_bernoulli=bernoulli).shape == (2, NUM_NODES)
    assert model.infer_from_gaussian(visible, binarize=False).shape == (2, NUM_NODES)
    assert model.infer_from_bernoulli(bernoulli, no_random=True).shape == (2, NUM_NODES)
