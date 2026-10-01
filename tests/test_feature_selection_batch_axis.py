"""Explicit input sample-axis learning with independent MSE/binary-loss oracles."""
from itertools import product
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from kaiwu.torch_plugin import FeatureSelectionWrapper
from kaiwu.torch_plugin.maifs import plugin


class LayoutLinear(nn.Module):
    """Real trainable linear model accepting feature-first or batch-first tensors."""

    def __init__(self, sequence, feature_first):
        super().__init__()
        self.sequence = sequence
        self.feature_first = feature_first
        self.linear = nn.Linear(3, 1, bias=False, dtype=torch.float64)
        with torch.no_grad():
            self.linear.weight.fill_(1)

    def forward(self, data):
        if self.feature_first:
            data = data.permute(1, 2, 0) if self.sequence else data.T
        elif self.sequence:
            data = data.transpose(1, 2)
        return self.linear(data)


def _setup(sequence=False, feature_first=False, negative=False, cadence=None):
    source = Path(__file__).resolve().parents[1] / 'src'
    assert Path(plugin.__file__).resolve().is_relative_to(source)
    rows = torch.tensor([[.1, 0, 0], [.2, 0, 0], [3, 1, 1], [4, 2, -1]],
                        dtype=torch.float64)
    canonical = torch.stack([rows / 4, rows * 3 / 4], dim=1) if sequence else rows
    targets = canonical[..., :1].clone()
    if feature_first:
        data = canonical.permute(2, 0, 1) if sequence else canonical.T
        batch_axis, feature_axis = 1, 0
    else:
        data = canonical.transpose(1, 2) if sequence else canonical
        batch_axis, feature_axis = 0, 1
    options = {}
    if negative:
        batch_axis -= data.ndim
        feature_axis -= data.ndim
        options['input_batch_axis'] = batch_axis
    elif feature_first:
        options['input_batch_axis'] = batch_axis
    selector = FeatureSelectionWrapper(LayoutLinear(sequence, feature_first), 3,
        lambda_reg=.3, min_selected_features=0, input_feature_axis=feature_axis,
        mask_update_epochs=cadence, solver='local_search', **options)
    return selector, data, targets, canonical, batch_axis


def _batches(data, targets, axis, counts=(2, 2)):
    batches, start = [], 0
    for count in counts:
        batches.append((data.narrow(axis, start, count), targets[start:start + count]))
        start += count
    return batches


def _oracle(canonical, targets, count, weights=None):
    design = canonical[:count].numpy().reshape(-1, 3)
    if weights is not None:
        design = design * weights
    residual = design.sum(axis=1) - targets[:count].numpy().reshape(-1)
    return 2 * design.T @ residual / len(design), 2 * design.T @ design / len(design)


@pytest.mark.parametrize('sequence', [False, True])
@pytest.mark.parametrize('feature_first', [False, True])
@pytest.mark.parametrize('maximum', [1, 3, None])
def test_layout_derivatives_match_independent_observation_mean(sequence, feature_first, maximum):
    selector, data, targets, canonical, axis = _setup(sequence, feature_first)
    actual = selector.compute_mask_derivatives(_batches(data, targets, axis), nn.MSELoss(),
                                               max_samples=maximum)
    expected = _oracle(canonical, targets, 4 if maximum is None else maximum)
    for result, oracle in zip(actual, expected):
        np.testing.assert_allclose(result, oracle, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize('sequence', [False, True])
def test_negative_sample_axis_and_unequal_batches_keep_partial_final_batch(sequence):
    selector, data, targets, canonical, axis = _setup(sequence, True, negative=True)
    for maximum in [3, None]:
        actual = selector.compute_mask_derivatives(_batches(data, targets, axis, (1, 3)),
                                                   nn.MSELoss(), max_samples=maximum)
        for result, oracle in zip(actual, _oracle(canonical, targets, maximum or 4)):
            np.testing.assert_allclose(result, oracle, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize('sequence', [False, True])
def test_feature_first_local_search_selects_unique_binary_loss_minimum(sequence):
    selector, data, targets, canonical, axis = _setup(sequence, True)
    choices = list(product((0, 1), repeat=3))
    losses = [np.mean((canonical[:3].numpy() @ mask - targets[:3].numpy()[..., 0])**2)
              + .3 * sum(mask) for mask in choices]
    assert choices[int(np.argmin(losses))] == (1, 0, 0)
    assert sorted(losses)[0] < sorted(losses)[1]

    selected = selector.update_mask(_batches(data, targets, axis), nn.MSELoss(), max_samples=3)

    np.testing.assert_array_equal(selected, choices[int(np.argmin(losses))])
    np.testing.assert_array_equal(selector.get_support(), [True, False, False])


@pytest.mark.parametrize('sequence', [False, True])
def test_actual_weight_training_and_repeated_cadence_use_configured_sample_axis(sequence):
    selector, data, targets, _, axis = _setup(sequence, True, cadence=1)
    initial = selector.model.linear.weight.detach().clone()
    loader = _batches(data, targets, axis, (1, 3))
    optimizer = torch.optim.SGD(selector.model.parameters(), lr=.01)

    first = selector.fit_weights(loader, nn.MSELoss(), optimizer)
    second = selector.fit_weights(loader, nn.MSELoss(), optimizer)

    assert np.isfinite(first) and np.isfinite(second)
    assert not torch.equal(initial, selector.model.linear.weight)
    assert torch.isfinite(selector.model.linear.weight.grad).all()
    assert selector._trained_epochs == 2
    np.testing.assert_array_equal(selector.mask.cpu().numpy(), [1, 0, 0])


def test_target_sample_count_must_match_explicit_input_sample_count():
    selector, data, targets, _, _ = _setup(feature_first=True)
    with pytest.raises(ValueError, match='sample counts'):
        selector.compute_mask_derivatives([(data, targets[:3])], nn.MSELoss())


@pytest.mark.parametrize('batch_axis', [2, -3])
def test_sample_axis_range_is_checked_when_collecting_derivative_batches(batch_axis):
    selector = FeatureSelectionWrapper(nn.Identity(), 3, input_feature_axis=0,
                                       input_batch_axis=batch_axis)
    with pytest.raises(ValueError, match='input_batch_axis.*out of range'):
        selector.compute_mask_derivatives([(torch.ones(3, 2), torch.ones(2, 3))], nn.MSELoss())


def test_sample_axis_must_differ_from_feature_axis_only_for_derivative_batches():
    selector = FeatureSelectionWrapper(nn.Identity(), 3, input_feature_axis=0)
    with torch.no_grad():
        selector.mask.copy_(torch.tensor([1., 0., 1.]))
    vector = torch.tensor([1., 2., 3.])
    torch.testing.assert_close(selector(vector), torch.tensor([1., 0., 3.]))
    with pytest.raises(ValueError, match='different'):
        selector.compute_mask_derivatives([(vector, vector)], nn.MSELoss())


def test_existing_positional_solver_arguments_keep_default_batch_first_behavior():
    model = LayoutLinear(False, False)
    selector = FeatureSelectionWrapper(model, 3, .3, None, 0, 3, 'local_search', None,
                                       -1, {'max_iter': 10})
    _, data, targets, canonical, axis = _setup()
    actual = selector.compute_mask_derivatives(_batches(data, targets, axis), nn.MSELoss(),
                                               max_samples=3)
    for result, oracle in zip(actual, _oracle(canonical, targets, 3)):
        np.testing.assert_allclose(result, oracle, rtol=1e-12, atol=1e-12)
    assert selector.solver_kwargs == {'max_iter': 10}
