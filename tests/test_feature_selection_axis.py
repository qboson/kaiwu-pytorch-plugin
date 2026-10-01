"""Feature-mask axis validation and broadcasting behavior tests."""

from math import prod

import pytest
import torch
from torch import nn

from kaiwu.torch_plugin import FeatureSelectionWrapper


class RecordingModel(nn.Module):
    """Record whether an invalid input reached the wrapped model."""

    def __init__(self):
        super().__init__()
        self.inputs = []

    def forward(self, data):
        self.inputs.append(data.detach().clone())
        return data


@pytest.mark.parametrize(
    "shape,axis",
    [
        ((3,), 1),
        ((3,), -2),
        ((3, 3), 2),
        ((3, 3), -3),
        ((3, 3), 200),
        ((3, 3), -201),
        ((3, 3, 3), 3),
        ((3, 3, 3), -4),
    ],
)
def test_out_of_range_axis_is_rejected_before_model_execution(shape, axis):
    """Equal dimension sizes must not hide an axis that wraps onto other data."""
    data = torch.ones(shape)
    model = RecordingModel()
    selector = FeatureSelectionWrapper(model, feature_dim=3, input_feature_axis=axis)
    with torch.no_grad():
        selector.mask.copy_(torch.tensor([1.0, 0.0, 1.0]))

    # PyTorch treats these as existing-dimension indices, not insertion axes.
    with pytest.raises(IndexError):
        data.select(axis, 0)
    with pytest.raises(ValueError, match="input_feature_axis.*out of range"):
        selector(data)

    assert model.inputs == []


@pytest.mark.parametrize(
    "shape,axis,feature_axis",
    [
        ((3,), 0, 0),
        ((3,), -1, 0),
        ((2, 3), 1, 1),
        ((2, 3), -1, 1),
        ((3, 2), 0, 0),
        ((3, 2), -2, 0),
        ((2, 3, 4), 1, 1),
        ((2, 3, 4), -2, 1),
        ((2, 4, 3), 2, 2),
        ((2, 4, 3), -1, 2),
        ((3, 2, 4), 0, 0),
        ((3, 2, 4), -3, 0),
    ],
)
def test_valid_axis_masks_only_the_selected_feature_slice(shape, axis, feature_axis):
    """Positive and negative axes identify the same feature across other dimensions."""
    data = torch.arange(1, prod(shape) + 1, dtype=torch.float64)
    data = data.reshape(shape)
    selector = FeatureSelectionWrapper(
        nn.Identity(), feature_dim=3, input_feature_axis=axis
    )
    mask = torch.tensor([1.0, 0.0, 1.0])
    expected = data.clone()
    feature_slice = [slice(None)] * len(shape)
    feature_slice[feature_axis] = 1
    expected[tuple(feature_slice)] = 0.0

    actual = selector.apply_mask(data, mask)

    assert actual.dtype == data.dtype
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("axis", [0, -2])
def test_valid_axis_supports_transposed_noncontiguous_input(axis):
    """Broadcasting depends on the feature dimension rather than memory layout."""
    data = torch.arange(1.0, 7.0).reshape(2, 3).transpose(0, 1)
    assert not data.is_contiguous()
    selector = FeatureSelectionWrapper(
        nn.Identity(), feature_dim=3, input_feature_axis=axis
    )
    mask = torch.tensor([1.0, 0.0, 1.0])
    expected = data.clone()
    expected[1, :] = 0.0

    actual = selector.apply_mask(data, mask)

    torch.testing.assert_close(actual, expected)


def test_middle_feature_axis_preserves_soft_mask_derivatives():
    """Sequence inputs retain gradients and Hessians for each shared mask entry."""
    data = torch.arange(1.0, 25.0, dtype=torch.float64).reshape(2, 3, 4)
    data.requires_grad_(True)
    mask = torch.tensor([0.2, 0.5, 0.8], dtype=torch.float64, requires_grad=True)
    selector = FeatureSelectionWrapper(
        nn.Identity(), feature_dim=3, input_feature_axis=-2
    )

    loss = selector.apply_mask(data, mask).square().sum()
    data_gradient, mask_gradient = torch.autograd.grad(loss, (data, mask), create_graph=True)
    mask_hessian = torch.stack(
        [
            torch.autograd.grad(mask_gradient[index], mask, retain_graph=True)[0]
            for index in range(3)
        ]
    )

    expected_data_gradient = torch.empty_like(data)
    for feature_index in range(3):
        expected_data_gradient[:, feature_index, :] = (
            2.0 * data.detach()[:, feature_index, :] * mask.detach()[feature_index] ** 2
        )
    expected_mask_gradient = (
        2.0 * mask.detach() * data.detach().square().sum(dim=(0, 2))
    )
    torch.testing.assert_close(data_gradient, expected_data_gradient)
    torch.testing.assert_close(mask_gradient, expected_mask_gradient)
    expected_mask_hessian = torch.diag(2.0 * data.detach().square().sum(dim=(0, 2)))
    torch.testing.assert_close(mask_hessian, expected_mask_hessian)


def test_scalar_input_keeps_its_existing_dimension_error():
    """A scalar has no feature axis to normalize or validate."""
    selector = FeatureSelectionWrapper(nn.Identity(), feature_dim=3)

    with pytest.raises(ValueError, match="must have at least one dimension"):
        selector.apply_mask(torch.tensor(1.0))


def test_valid_axis_keeps_feature_size_validation():
    """A valid axis still needs the configured number of selectable features."""
    selector = FeatureSelectionWrapper(
        nn.Identity(), feature_dim=3, input_feature_axis=-1
    )

    with pytest.raises(ValueError, match="axis has size 4, expected 3"):
        selector.apply_mask(torch.ones(2, 4))
