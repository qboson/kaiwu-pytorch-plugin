"""Regression tests for the derivative collection sample budget."""
import numpy as np
import pytest
import torch
from torch import nn

from kaiwu.torch_plugin.maifs.plugin import FeatureSelectionWrapper


def selector():
    model = nn.Linear(4, 1, bias=False).double()
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[0.5, -0.25, 1.0, 0.75]]))
    return FeatureSelectionWrapper(model, feature_dim=4).double()


def data(rows):
    x = torch.arange(rows * 4, dtype=torch.float64).reshape(rows, 4) / 100
    y = x[:, :1] - 0.5 * x[:, 1:2]
    return x, y


@pytest.mark.parametrize('limit', [0, -1, -20])
def test_nonpositive_budget_rejected_before_iteration(limit):
    model = selector()
    def loader():
        pytest.fail('invalid budget consumed the loader')
        yield data(4)
    with pytest.raises(ValueError, match='max_samples'):
        model.compute_mask_derivatives(loader(), nn.MSELoss(), max_samples=limit)


@pytest.mark.parametrize('limit', [1.5, '2', True, False, np.bool_(True)])
def test_noninteger_budget_rejected_before_iteration(limit):
    model = selector()
    def loader():
        pytest.fail('invalid budget consumed the loader')
        yield data(4)
    with pytest.raises(TypeError, match='max_samples'):
        model.compute_mask_derivatives(loader(), nn.MSELoss(), max_samples=limit)


@pytest.mark.parametrize('limit', [1, 5, 7, 12, 30, None, np.int64(9)])
@pytest.mark.parametrize('mode', ['full', 'diagonal'])
def test_derivatives_match_independent_prefix(limit, mode):
    x, y = data(20)
    batches = [(x[:5], y[:5]), (x[5:12], y[5:12]), (x[12:], y[12:])]
    expected_rows = len(x) if limit is None else min(int(limit), len(x))
    model = selector()
    actual = model.compute_mask_derivatives(batches, nn.MSELoss(), mode, limit)
    expected = model.compute_mask_derivatives(
        [(x[:expected_rows], y[:expected_rows])], nn.MSELoss(), mode, None
    )
    for left, right in zip(actual, expected):
        np.testing.assert_allclose(left, right, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize('sizes,limit', [([20000], 16), ([10, 20000], 16), ([8, 8, 10], 16)])
def test_cat_does_not_copy_samples_beyond_budget(monkeypatch, sizes, limit):
    batches = [data(n) for n in sizes]
    original_cat = torch.cat
    records = []
    def checked_cat(tensors, dim=0, **kwargs):
        lengths = [len(t) for t in tensors]
        result = original_cat(tensors, dim=dim, **kwargs)
        records.append((lengths, result.shape, result.untyped_storage().nbytes()))
        assert sum(lengths) <= limit
        return result
    monkeypatch.setattr(torch, 'cat', checked_cat)
    selector().compute_mask_derivatives(batches, nn.MSELoss(), max_samples=limit)
    assert len(records) == 2
    assert records[0][1] == (16, 4)
    assert records[0][2] == 16 * 4 * 8
    assert records[1][2] == 16 * 8


def test_budget_does_not_read_next_batch():
    def loader():
        yield data(8)
        pytest.fail('read after reaching the budget')
    selector().compute_mask_derivatives(loader(), nn.MSELoss(), max_samples=8)


@pytest.mark.parametrize('fail', [False, True])
def test_collection_preserves_inputs_flags_and_existing_gradients(fail):
    model = selector().train()
    params = list(model.model.parameters())
    params[0].grad = torch.ones_like(params[0])
    x, y = data(20)
    original_x, original_y = x.clone(), y.clone()
    original_mask = model.mask.clone()
    def loss(prediction, target):
        if fail:
            raise RuntimeError('controlled failure')
        return nn.functional.mse_loss(prediction, target)
    if fail:
        with pytest.raises(RuntimeError, match='controlled failure'):
            model.compute_mask_derivatives([(x, y)], loss, max_samples=7)
    else:
        model.compute_mask_derivatives([(x, y)], loss, max_samples=7)
    assert model.training
    assert params[0].requires_grad
    assert torch.equal(params[0].grad, torch.ones_like(params[0]))
    assert torch.equal(model.mask, original_mask)
    assert torch.equal(x, original_x) and torch.equal(y, original_y)


@pytest.mark.parametrize('mode', ['full', 'diagonal'])
def test_derivatives_match_linear_mse_formula(mode):
    x, y = data(50)
    model = selector()
    model.mask.copy_(torch.tensor([1.0, 0.0, 1.0, 1.0]))
    gradient, hessian = model.compute_mask_derivatives(
        [(x, y)], nn.MSELoss(), hessian_mode=mode, max_samples=9
    )
    features = x[:9].numpy() * model.model.weight.detach().numpy()
    residual = features @ model.mask.numpy() - y[:9, 0].numpy()
    expected_gradient = 2 * features.T @ residual / 9
    expected_hessian = 2 * features.T @ features / 9
    if mode == 'diagonal':
        expected_hessian = np.diag(np.diag(expected_hessian))
    np.testing.assert_allclose(gradient, expected_gradient, atol=1e-12)
    np.testing.assert_allclose(hessian, expected_hessian, atol=1e-12)
