"""Actual DPLM feature pooling precision and trainable reranker regressions."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn


def _models():
    name = '_dplm_pooling_test_models'
    if name not in sys.modules:
        folder = Path(__file__).resolve().parents[1] / 'example/qdiffusion/dplm/models'
        spec = importlib.util.spec_from_file_location(
            name, folder / '__init__.py', submodule_search_locations=[str(folder)])
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def _pool(hidden, mask):
    _models()
    module = sys.modules['_dplm_pooling_test_models.common']
    return module.masked_mean_pool(hidden, mask)


@pytest.mark.parametrize('dtype,length,value', [
    (torch.float16, 1000, 1000.), (torch.float16, 800, -200.),
    (torch.bfloat16, 257, 7.), (torch.bfloat16, 513, 7.),
    (torch.float32, 1000, 1000.), (torch.float64, 1000, 1000.)])
def test_representable_constant_mean_does_not_overflow_or_round_twice(dtype, length, value):
    hidden = torch.full((2, length, 3), value, dtype=dtype)
    mask = torch.ones(2, length, dtype=torch.bool)
    result = _pool(hidden, mask)
    assert result.dtype == dtype and result.device == hidden.device
    assert torch.isfinite(result).all()
    torch.testing.assert_close(result, torch.full((2, 3), value, dtype=dtype), atol=0, rtol=0)


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_noncontiguous_masked_mean_and_gradient_match_float64_weighted_reference(dtype):
    hidden = torch.linspace(-20, 20, 2 * 3 * 257).reshape(2, 3, 257)
    hidden = hidden.to(dtype).transpose(1, 2).detach().requires_grad_()
    assert not hidden.is_contiguous()
    mask = torch.zeros(2, 257, dtype=torch.bool)
    mask[0, :193] = True
    mask[1, 1::2] = True
    original = hidden.detach().clone()
    expected = torch.stack([hidden[i, mask[i]].double().mean(0) for i in range(2)]).to(dtype)
    actual = _pool(hidden, mask)
    torch.testing.assert_close(actual, expected, atol=2e-6 if dtype == torch.float32 else 0,
                               rtol=2e-7 if dtype == torch.float32 else 0)
    actual.sum().backward()
    gradient = torch.stack([mask[i].double().unsqueeze(-1).expand(257, 3) /
                            int(mask[i].sum()) for i in range(2)]).to(dtype)
    torch.testing.assert_close(hidden.grad, gradient, atol=0, rtol=0)
    torch.testing.assert_close(hidden.detach(), original, atol=0, rtol=0)


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize('pad_value', [float('nan'), float('inf'), -float('inf')])
def test_inactive_nonfinite_padding_is_excluded_with_zero_gradient(dtype, pad_value):
    hidden = torch.tensor([[[2., -4.], [6., 8.], [pad_value, pad_value]],
                           [[pad_value, pad_value]] * 3], dtype=dtype, requires_grad=True)
    mask = torch.tensor([[1, 1, 0], [0, 0, 0]])
    actual = _pool(hidden, mask)
    torch.testing.assert_close(actual, torch.tensor([[4., 2.], [0., 0.]], dtype=dtype))
    actual.sum().backward()
    expected = torch.tensor([[[.5, .5], [.5, .5], [0., 0.]], [[0., 0.]] * 3], dtype=dtype)
    torch.testing.assert_close(hidden.grad, expected, atol=0, rtol=0)


def test_nonfinite_active_tokens_are_not_silently_replaced():
    hidden = torch.tensor([[[float('nan'), 1.], [4., 5.]]])
    actual = _pool(hidden, torch.tensor([[1, 0]]))
    assert torch.isnan(actual[0, 0]) and actual[0, 1] == 1.


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_numeric_weights_and_existing_clamped_denominator_are_preserved(dtype):
    hidden = torch.tensor([[[2., 4.], [8., 10.]], [[6., 8.], [2., 4.]]], dtype=dtype)
    mask = torch.tensor([[.5, 1.5], [.25, 0.]], dtype=dtype)
    expected = torch.tensor([[6.5, 8.5], [1.5, 2.]], dtype=dtype)
    torch.testing.assert_close(_pool(hidden, mask), expected, atol=0, rtol=0)


class _TinyTokenNet(nn.Module):
    """A real trainable network injected through the existing backbone API."""
    mask_id, pad_id, bos_id, eos_id, x_id = 0, 1, 2, 3, 4
    tokenizer = None

    def __init__(self, dtype):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=2)
        self.embedding = nn.Embedding(5, 2, dtype=dtype)
        with torch.no_grad():
            self.embedding.weight.fill_(1000.)

    def forward(self, input_ids, attention_mask=None):
        return {'last_hidden_state': self.embedding(input_ids)}


class _RecordedSampler:
    def solve(self, matrix):
        raise AssertionError('Feature projection training does not require an external solver')


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_real_backbone_encoder_reranker_projection_and_sgd_remain_finite(dtype):
    models = _models()
    backbone_module = sys.modules['_dplm_pooling_test_models.backbone']
    net = _TinyTokenNet(dtype)
    backbone = backbone_module.DPLMBackbone(net=net)
    encoder = models.DPLMFeatureEncoder(backbone)
    model = models.BMConditionedEnergyModel(encoder, 2, 1, sampler=_RecordedSampler()).to(dtype)
    with torch.no_grad():
        model.feature_projector.weight.fill_(.001)
        model.feature_projector.bias.zero_()
    tokens = torch.full((2, 1000), 4, dtype=torch.long)
    mask = torch.ones_like(tokens, dtype=torch.bool)
    before = model.feature_projector.weight.detach().clone()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-7)
    logits = model.build_visible_logits(tokens, tokens, mask)
    expected = (torch.full((2, 4), 1000., dtype=torch.float64) @ before.double().t()).to(dtype)
    torch.testing.assert_close(logits, expected, atol=0, rtol=0)
    loss = logits.float().square().mean()
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(model.feature_projector.weight.grad).all()
    assert torch.isfinite(net.embedding.weight.grad).all()
    # Independent chain rule: d mean(logit^2)/d W_ij = mean(logit_i * feature_j).
    expected_grad = (logits.detach().double().mean(0).unsqueeze(-1) * 1000.).expand(2, 4).to(dtype)
    torch.testing.assert_close(model.feature_projector.weight.grad, expected_grad, atol=0, rtol=0)
    optimizer.step()
    assert not torch.equal(before, model.feature_projector.weight)
    assert torch.isfinite(model.build_visible_logits(tokens, tokens, mask)).all()
