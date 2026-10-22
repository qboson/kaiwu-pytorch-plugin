"""GBRBM variance, deviation and precision accessors must agree."""

import torch

from kaiwu.torch_plugin.gbrbm import GaussianBernoulliRestrictedBoltzmannMachine


def test_variance_properties_match_log_var():
    torch.manual_seed(0)
    model = GaussianBernoulliRestrictedBoltzmannMachine(
        num_visible=3, num_hidden=2, dtype=torch.float64
    )
    with torch.no_grad():
        model.log_var.copy_(torch.log(torch.tensor([0.25, 1.0, 4.0], dtype=torch.float64)))

    torch.testing.assert_close(model.var, torch.tensor([0.25, 1.0, 4.0], dtype=torch.float64))
    torch.testing.assert_close(model.std, torch.tensor([0.5, 1.0, 2.0], dtype=torch.float64))
    torch.testing.assert_close(
        model.diag_precision,
        torch.diag(torch.tensor([4.0, 1.0, 0.25], dtype=torch.float64)),
    )


def test_variance_is_clipped_at_eps():
    torch.manual_seed(0)
    model = GaussianBernoulliRestrictedBoltzmannMachine(
        num_visible=3, num_hidden=2, dtype=torch.float64
    )
    with torch.no_grad():
        model.log_var.fill_(-100.0)

    assert torch.all(model.var == model.eps)
