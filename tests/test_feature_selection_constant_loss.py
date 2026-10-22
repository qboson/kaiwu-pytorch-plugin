"""Feature-independent predictors have well-defined zero mask derivatives."""
import numpy as np
import pytest
import torch
from torch import nn

from kaiwu.torch_plugin.maifs.plugin import FeatureSelectionWrapper


class BiasOnlyPredictor(nn.Module):
    """A constant baseline model whose loss cannot depend on feature selection."""
    def __init__(self, dtype):
        super().__init__()
        self.bias = nn.Parameter(torch.tensor([0.5], dtype=dtype))

    def forward(self, inputs):
        return self.bias.expand(len(inputs), 1)


def data(dtype):
    inputs = torch.tensor([[1.0, 2.0, 3.0], [4.0, -2.0, 1.0]], dtype=dtype)
    targets = torch.tensor([[0.0], [1.0]], dtype=dtype)
    return [(inputs, targets)]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("mode", ["full", "diagonal"])
@pytest.mark.parametrize("criterion", ["squared", "linear"])
def test_feature_independent_losses_have_zero_gradient_and_hessian(dtype, mode, criterion):
    model = BiasOnlyPredictor(dtype)
    wrapper = FeatureSelectionWrapper(model, feature_dim=3).to(dtype=dtype)
    model.bias.grad = torch.full_like(model.bias, 7.0)
    if criterion == "squared":
        loss_fn = nn.MSELoss()
    else:
        loss_fn = lambda prediction, target: (prediction - target).mean()

    gradient, hessian = wrapper.compute_mask_derivatives(data(dtype), loss_fn, mode)

    np.testing.assert_array_equal(gradient, np.zeros(3))
    np.testing.assert_array_equal(hessian, np.zeros((3, 3)))
    assert model.bias.requires_grad
    assert model.training and wrapper.training
    torch.testing.assert_close(model.bias.grad, torch.full_like(model.bias, 7.0))


@pytest.mark.parametrize("mode", ["full", "diagonal"])
def test_unrelated_differentiable_loss_term_does_not_create_mask_derivatives(mode):
    wrapper = FeatureSelectionWrapper(BiasOnlyPredictor(torch.float64), feature_dim=3).double()
    regularizer = nn.Parameter(torch.tensor(2.0, dtype=torch.float64))

    def loss_fn(prediction, target):
        return (prediction - target).square().mean() + regularizer.square()

    gradient, hessian = wrapper.compute_mask_derivatives(data(torch.float64), loss_fn, mode)

    np.testing.assert_array_equal(gradient, np.zeros(3))
    np.testing.assert_array_equal(hessian, np.zeros((3, 3)))
    assert regularizer.grad is None


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_update_mask_can_select_no_features_for_a_constant_predictor(dtype):
    wrapper = FeatureSelectionWrapper(
        BiasOnlyPredictor(dtype), feature_dim=3, lambda_reg=1.0,
        min_selected_features=0, solver="local_search",
    ).to(dtype=dtype)

    selected = wrapper.update_mask(data(dtype), nn.MSELoss())

    np.testing.assert_array_equal(selected, np.zeros(3, dtype=int))
    assert wrapper.num_selected() == 0
    for inputs, _ in data(dtype):
        torch.testing.assert_close(wrapper(inputs), torch.full((len(inputs), 1), 0.5, dtype=dtype))


@pytest.mark.parametrize("mode", ["full", "diagonal"])
def test_nonconstant_mask_derivatives_still_match_squared_loss_algebra(mode):
    model = nn.Linear(3, 1, bias=False).double()
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[0.5, -1.0, 2.0]], dtype=torch.float64))
    wrapper = FeatureSelectionWrapper(model, feature_dim=3).double()
    inputs, targets = data(torch.float64)[0]
    masked_columns = inputs.numpy() * np.array([0.5, -1.0, 2.0])
    residual = masked_columns.sum(axis=1) - targets.numpy().ravel()
    expected_gradient = 2.0 * (masked_columns * residual[:, None]).mean(axis=0)
    expected_hessian = 2.0 * masked_columns.T @ masked_columns / len(inputs)
    if mode == "diagonal":
        expected_hessian = np.diag(np.diag(expected_hessian))

    gradient, hessian = wrapper.compute_mask_derivatives([(inputs, targets)], nn.MSELoss(), mode)

    np.testing.assert_allclose(gradient, expected_gradient, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(hessian, expected_hessian, rtol=1e-12, atol=1e-12)


def test_disabled_autograd_does_not_silently_report_zero_for_a_nonconstant_loss():
    wrapper = FeatureSelectionWrapper(nn.Linear(3, 1), feature_dim=3)

    with torch.no_grad():
        with pytest.raises(RuntimeError):
            wrapper.compute_mask_derivatives(data(torch.float32), nn.MSELoss())


@pytest.mark.parametrize("enable_grad_inside_inference", [False, True])
def test_inference_mode_does_not_silently_report_zero_for_a_nonconstant_loss(
    enable_grad_inside_inference,
):
    wrapper = FeatureSelectionWrapper(nn.Linear(3, 1), feature_dim=3)

    with torch.inference_mode(), torch.set_grad_enabled(enable_grad_inside_inference):
        with pytest.raises(RuntimeError):
            wrapper.compute_mask_derivatives(data(torch.float32), nn.MSELoss())
