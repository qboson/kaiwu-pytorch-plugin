"""Mask derivatives use a local autograd context and reject invalid results."""

from contextlib import nullcontext

import numpy as np
import pytest
import torch

from kaiwu.torch_plugin.maifs import FeatureSelectionWrapper


def make_selector():
    """Build a linear regression with analytically known mask derivatives."""
    model = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[1.0, -2.0]]))
    model.weight.grad = torch.tensor([[7.0, 8.0]])
    selector = FeatureSelectionWrapper(model, feature_dim=2)
    loader = [(torch.tensor([[1.0, 2.0], [3.0, -1.0]]), torch.tensor([[0.0], [1.0]]))]
    return selector, loader


@pytest.mark.parametrize("no_grad", [False, True])
@pytest.mark.parametrize("mode", ["full", "diagonal"])
def test_mask_derivatives_match_analytic_result(no_grad, mode):
    """Evaluation callers get the same derivatives without changing their state."""
    selector, loader = make_selector()
    original_grad = selector.model.weight.grad.clone()
    with torch.no_grad() if no_grad else nullcontext():
        gradient, hessian = selector.compute_mask_derivatives(
            loader, torch.nn.MSELoss(), hessian_mode=mode
        )
        assert torch.is_grad_enabled() is not no_grad
    np.testing.assert_allclose(gradient, [9.0, 20.0])
    expected = np.array([[10.0, 2.0], [2.0, 20.0]])
    np.testing.assert_allclose(hessian, np.diag(np.diag(expected)) if mode == "diagonal" else expected)
    assert selector.training
    assert selector.model.weight.requires_grad
    torch.testing.assert_close(selector.model.weight.grad, original_grad)
    torch.testing.assert_close(selector.mask, torch.ones(2))


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_nonfinite_loss_is_rejected_and_state_restored(value):
    """Invalid training data must not silently produce a new feature mask."""
    selector, loader = make_selector()
    selector.eval()
    selector.model.weight.requires_grad_(False)

    def loss_fn(prediction, target):
        return (prediction - target).square().mean() * value

    with pytest.raises(ValueError, match="finite"):
        selector.compute_mask_derivatives(loader, loss_fn)
    assert not selector.training
    assert not selector.model.weight.requires_grad
    torch.testing.assert_close(selector.model.weight.grad, torch.tensor([[7.0, 8.0]]))


def test_nonfinite_derivatives_of_finite_loss_are_rejected():
    """A finite value does not imply that its derivatives are finite."""
    selector, _ = make_selector()
    loader = [(torch.ones(1, 2), torch.tensor([[-1.0]]))]

    def loss_fn(prediction, target):
        return torch.sqrt((prediction - target).square()).mean()

    with pytest.raises(ValueError, match="finite"):
        selector.compute_mask_derivatives(loader, loss_fn)
    assert selector.training
    assert selector.model.weight.requires_grad
