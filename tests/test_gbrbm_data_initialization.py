"""Initialize only the Gaussian moments using population statistics."""

import pytest
import torch

from kaiwu.torch_plugin.gbrbm import GaussianBernoulliRestrictedBoltzmannMachine


@pytest.mark.parametrize("gaussian_visible", [True, False])
def test_fit_gaussian_moments_updates_existing_parameters(gaussian_visible):
    dimensions = (3, 2) if gaussian_visible else (2, 3)
    model = GaussianBernoulliRestrictedBoltzmannMachine(
        *dimensions, is_visible_gaussian=gaussian_visible, device="cpu",
        dtype=torch.float64, eps=1e-5,
    )
    samples = torch.tensor([[1, -3, 2], [3, -3, -1], [5, -3, 5]], dtype=torch.float32)
    original_mu, original_log_var = model.mu, model.log_var
    couplings = model.quadratic_coef.detach().clone()
    bias = model.linear_bias.detach().clone()

    model.initialize_gaussian_from_data(samples)

    assert model.mu is original_mu and model.log_var is original_log_var
    torch.testing.assert_close(model.mu, torch.tensor([3., -3., 2.], dtype=torch.float64))
    torch.testing.assert_close(model.var, torch.tensor([8 / 3, 1e-5, 6.], dtype=torch.float64))
    torch.testing.assert_close(model.quadratic_coef, couplings)
    torch.testing.assert_close(model.linear_bias, bias)


def test_single_row_uses_the_floor_for_all_features():
    model = GaussianBernoulliRestrictedBoltzmannMachine(2, 1, device="cpu", eps=0.001)
    model.initialize_gaussian_from_data(torch.tensor([[1., -2.]]))
    torch.testing.assert_close(model.mu, torch.tensor([1., -2.]))
    torch.testing.assert_close(model.var, torch.full((2,), 0.001))


@pytest.mark.parametrize(
    "samples, message",
    [(torch.empty(0, 2), "nonempty"), (torch.ones(3), "shape"),
     (torch.ones(2, 3), "shape"), (torch.tensor([[0., float("nan")]]), "finite"),
     (torch.tensor([[-1e35, 0.], [1e35, 0.]]), "representable")],
)
def test_invalid_data_leaves_parameters_intact(samples, message):
    model = GaussianBernoulliRestrictedBoltzmannMachine(2, 1, device="cpu")
    previous = (model.mu.detach().clone(), model.log_var.detach().clone())
    with pytest.raises(ValueError, match=message):
        model.initialize_gaussian_from_data(samples)
    torch.testing.assert_close(model.mu, previous[0])
    torch.testing.assert_close(model.log_var, previous[1])
