"""QVAE flattening should depend on logical shape rather than strides."""
import contextlib
import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from kaiwu.torch_plugin.qvae import QVAE
from kaiwu.torch_plugin.full_boltzmann_machine import BoltzmannMachine
import kaiwu.torch_plugin.abstract_boltzmann_machine as abstract_module


@pytest.fixture(autouse=True)
def offline_context(monkeypatch):
    monkeypatch.setattr(abstract_module, 'kpp_caller_context', contextlib.nullcontext)


class Sampler:
    def solve(self, matrix):
        return np.ones((2, len(matrix)), dtype=np.float32)


class TinyQVAE(QVAE):
    def _create_encoder(self):
        return nn.Linear(8, 3)

    def _create_decoder(self):
        return nn.Linear(3, 8)

    def _create_bm(self):
        return BoltzmannMachine(3, device='cpu')

    def _create_sampler(self, sampler_type):
        return Sampler()


def model(loss_type):
    torch.manual_seed(123)
    config = SimpleNamespace(num_latent_units=3, loss_type=loss_type,
                             dist_beta=2., kl_beta=.1, weight_decay=.01)
    result = TinyQVAE(8, None, config)
    # Avoid conflating the Bernoulli energy configuration issue covered by #175.
    result.set_dataset_mean(torch.full((8,), .5))
    result.set_train_bias(torch.full((8,), .5))
    return result


def inputs(layout):
    if layout == 'permute':
        x = (torch.arange(24, dtype=torch.float32) / 48).reshape(3, 2, 2, 2).permute(0, 2, 3, 1)
    elif layout == 'transpose':
        x = (torch.arange(24, dtype=torch.float32) / 48).reshape(3, 2, 4).transpose(1, 2)
    elif layout == 'strided_flat':
        x = (torch.arange(48, dtype=torch.float32) / 48).reshape(3, 16)[:, ::2]
    elif layout == 'sliced_image':
        x = (torch.arange(48, dtype=torch.float32) / 48).reshape(3, 2, 8)[:, :, ::2]
    else:
        x = (torch.arange(24, dtype=torch.float32) / 48).reshape(3, 2, 4)
    if layout != 'contiguous':
        assert not x.is_contiguous()
    return x


LAYOUTS = ['contiguous', 'transpose', 'permute', 'strided_flat', 'sliced_image']


@pytest.mark.parametrize('layout', LAYOUTS)
@pytest.mark.parametrize('loss_type', ['mse', 'bernoulli'])
@pytest.mark.parametrize('training', [False, True])
def test_forward_matches_contiguous_reference(layout, loss_type, training):
    actual_model = model(loss_type).train(training)
    expected_model = copy.deepcopy(actual_model)
    x = inputs(layout)
    before = x.clone()
    torch.manual_seed(19)
    actual = actual_model(x)
    torch.manual_seed(19)
    expected = expected_model(x.contiguous())
    for index in [0, 2, 3]:
        torch.testing.assert_close(actual[index], expected[index], rtol=0, atol=0)
    assert torch.equal(actual[1].logit_mu, expected[1].logit_mu)
    assert torch.equal(x, before)
    assert actual_model.training is training


@pytest.mark.parametrize('layout', LAYOUTS)
@pytest.mark.parametrize('loss_type', ['mse', 'bernoulli'])
def test_energy_matches_contiguous_reference(layout, loss_type):
    actual_model = model(loss_type)
    x = inputs(layout)
    actual = actual_model.energy(x)
    expected = actual_model.energy(x.contiguous())
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize('layout', LAYOUTS)
def test_mse_loss_matches_contiguous_reference(layout):
    actual_model = model('mse').eval()
    x = inputs(layout)
    recon, posterior, _, _ = actual_model(x.contiguous())
    actual = actual_model.loss(x, recon, posterior)
    expected = actual_model.loss(x.contiguous(), recon, posterior)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize('layout', ['transpose', 'permute', 'strided_flat'])
@pytest.mark.parametrize('loss_type', ['mse', 'bernoulli'])
def test_encoder_input_and_parameter_gradients_match_reference(layout, loss_type):
    actual_model = model(loss_type)
    expected_model = copy.deepcopy(actual_model)
    x = inputs(layout).detach().requires_grad_()
    reference_x = x.detach().contiguous().requires_grad_()
    # Encoder logits avoid entangling this layout regression with posterior precision.
    actual_model(x)[2].square().sum().backward()
    expected_model(reference_x)[2].square().sum().backward()
    torch.testing.assert_close(x.grad, reference_x.grad, rtol=0, atol=0)
    for left, right in zip(actual_model.encoder.parameters(), expected_model.encoder.parameters()):
        torch.testing.assert_close(left.grad, right.grad, rtol=0, atol=0)


@pytest.mark.parametrize('layout', ['transpose', 'permute', 'strided_flat'])
def test_mse_loss_input_and_decoder_gradients_match_reference(layout):
    actual_model = model('mse').eval()
    expected_model = copy.deepcopy(actual_model)
    x = inputs(layout).detach().requires_grad_()
    reference_x = x.detach().contiguous().requires_grad_()
    torch.manual_seed(29)
    recon, posterior, _, _ = actual_model(x)
    actual_model.loss(x, recon, posterior).backward()
    torch.manual_seed(29)
    recon, posterior, _, _ = expected_model(reference_x)
    expected_model.loss(reference_x, recon, posterior).backward()
    torch.testing.assert_close(x.grad, reference_x.grad, rtol=0, atol=0)
    for left, right in zip(actual_model.parameters(), expected_model.parameters()):
        if left.grad is None or right.grad is None:
            assert left.grad is None and right.grad is None
        else:
            torch.testing.assert_close(left.grad, right.grad, rtol=0, atol=0)
