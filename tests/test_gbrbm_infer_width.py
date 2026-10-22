"""GBRBM conditional inference must validate the conditioning state width."""

import pytest
import torch

from kaiwu.torch_plugin.gbrbm import GaussianBernoulliRestrictedBoltzmannMachine


def _model() -> GaussianBernoulliRestrictedBoltzmannMachine:
    return GaussianBernoulliRestrictedBoltzmannMachine(num_visible=3, num_hidden=2)


def test_infer_from_gaussian_rejects_wrong_width() -> None:
    with pytest.raises(ValueError, match=r"s_gaussian must have shape \(batch, 3\)"):
        _model().infer_from_gaussian(torch.zeros(2, 4))


def test_infer_from_bernoulli_rejects_wrong_width() -> None:
    with pytest.raises(ValueError, match=r"s_bernoulli must have shape \(batch, 2\)"):
        _model().infer_from_bernoulli(torch.zeros(2, 3))


@pytest.mark.parametrize("call", ["gaussian", "bernoulli"])
def test_conditional_inference_rejects_one_dimensional_state(call) -> None:
    model = _model()
    if call == "gaussian":
        with pytest.raises(ValueError, match="s_gaussian must have shape"):
            model.infer_from_gaussian(torch.zeros(3))
    else:
        with pytest.raises(ValueError, match="s_bernoulli must have shape"):
            model.infer_from_bernoulli(torch.zeros(2))


def test_conditional_inference_still_accepts_matching_widths() -> None:
    model = _model()
    assert model.infer_from_gaussian(torch.zeros(2, 3)).shape == (2, 5)
    assert model.infer_from_bernoulli(torch.zeros(2, 2)).shape == (2, 5)
    assert model.infer_from_gaussian(torch.zeros(0, 3)).shape == (0, 5)
    assert model.infer_from_bernoulli(torch.zeros(0, 2)).shape == (0, 5)
