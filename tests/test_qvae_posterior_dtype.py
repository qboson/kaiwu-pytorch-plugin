"""Posterior samples must remain usable by converted real QVAE networks."""
import importlib
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn
import torch.nn.functional as F

DTYPES = (torch.float16, torch.bfloat16, torch.float32, torch.float64)


@pytest.fixture
def core():
    source = Path(__file__).resolve().parents[1] / "src"
    sys.path.insert(0, str(source))
    import kaiwu
    kaiwu.__path__.insert(0, str(source / "kaiwu"))
    modules = tuple(importlib.import_module(f"kaiwu.torch_plugin.{name}") for name in (
        "qvae", "qvae_dist_util", "restricted_boltzmann_machine",
    ))
    for module in modules:
        assert Path(module.__file__).resolve() == (
            source / "kaiwu/torch_plugin" / f"{module.__name__.split('.')[-1]}.py"
        ).resolve()
    return modules


class LegalOfflineSampler:
    """Supply legal auxiliary-spin vectors without licensed solver execution."""
    def solve(self, matrix):
        assert matrix.shape == (3, 3)
        assert np.all(np.isfinite(matrix))
        return np.array([[1, -1, 1], [-1, 1, 1]], dtype=np.int8)


def make_model(core, dtype):
    qvae, _, rbm = core
    class InjectedQVAE(qvae.QVAE):
        def _create_encoder(self):
            raise AssertionError("Use injected encoder")
        def _create_decoder(self):
            raise AssertionError("Use injected decoder")
        def _create_bm(self):
            raise AssertionError("Use injected BM")
        def _create_sampler(self, sampler_type):
            raise AssertionError("Use injected sampler")
    torch.manual_seed(17)
    config = SimpleNamespace(num_latent_units=2, loss_type="bernoulli", dist_beta=2.,
                             kl_beta=.01, weight_decay=.001)
    model = InjectedQVAE(3, None, config, encoder=nn.Linear(3, 2), decoder=nn.Linear(2, 3),
                         bm=rbm.RestrictedBoltzmannMachine(2, 0, device="cpu"),
                         sampler=LegalOfflineSampler())
    convert = {torch.float16: model.half, torch.bfloat16: model.bfloat16,
               torch.float32: model.float, torch.float64: model.double}[dtype]
    assert convert() is model  # Standard nn.Module._apply, not BM.to.
    assert all(parameter.dtype == dtype for parameter in model.parameters())
    return model


@pytest.mark.parametrize("dtype", DTYPES)
def test_real_qvae_forward_and_reconstruction_training(core, dtype):
    model = make_model(core, dtype)
    inputs = torch.tensor([[0., 1., 0.], [1., 0., 1.], [1., 1., 0.]], dtype=dtype)
    original_encoder = model.encoder.weight.detach().clone()
    original_decoder = model.decoder.weight.detach().clone()
    reconstruction, posterior, logits, latent = model(inputs)
    assert logits.dtype == latent.dtype == reconstruction.dtype == dtype
    assert posterior.logit_mu is logits
    # Reconstruction alone isolates posterior dtype from the separately proposed
    # BM negative-phase sampling precision fix (PR186). No BM method is patched.
    reconstruction_loss = F.binary_cross_entropy_with_logits(reconstruction, inputs)
    reconstruction_loss.backward()
    for component in (model.encoder, model.decoder):
        assert all(parameter.grad is not None and torch.all(torch.isfinite(parameter.grad))
                   for parameter in component.parameters())
        assert any(torch.any(parameter.grad != 0) for parameter in component.parameters())
    torch.optim.SGD(model.parameters(), lr=.05).step()
    assert not torch.equal(original_encoder, model.encoder.weight)
    assert not torch.equal(original_decoder, model.decoder.weight)
    model.eval()
    with torch.no_grad():
        assert model(inputs)[0].dtype == dtype


def test_float32_real_full_loss_and_bm_backward(core):
    model = make_model(core, torch.float32)
    inputs = torch.tensor([[0., 1., 0.], [1., 0., 1.], [1., 1., 0.]])
    reconstruction, posterior, logits, _ = model(inputs)
    loss = model.loss(inputs, reconstruction, posterior)
    assert torch.isfinite(loss)
    loss.backward()
    assert model.encoder.weight.grad is not None
    assert model.decoder.weight.grad is not None
    model.zero_grad(set_to_none=True)
    bm_loss = model.bm_loss(logits.detach())
    assert torch.isfinite(bm_loss)
    bm_loss.backward()
    assert model.encoder.weight.grad is None
    assert model.bm.linear_bias.grad is not None
    assert torch.all(torch.isfinite(model.bm.linear_bias.grad))


@pytest.mark.parametrize("dtype", DTYPES)
def test_bernoulli_bits_preserve_dtype_and_rng(core, dtype):
    _, distributions, _ = core
    logits = torch.tensor([[-.7, .4, 1.2]], dtype=dtype)
    probability = torch.sigmoid(logits)
    torch.manual_seed(31)
    uniforms = torch.rand_like(probability)
    expected_next_draw = torch.rand(5)
    torch.manual_seed(31)
    bits = distributions.FactorialBernoulliUtil(logits).reparameterize(False)
    assert bits.dtype == dtype and bits.device == logits.device
    torch.testing.assert_close(bits.bool(), uniforms < probability)
    torch.testing.assert_close(torch.rand(5), expected_next_draw)
    with pytest.raises(NotImplementedError):
        distributions.FactorialBernoulliUtil(logits).reparameterize(True)


def exponential_cdf(sample):
    return -math.expm1(-2. * sample) / -math.expm1(-2.)


@pytest.mark.parametrize("dtype", [torch.bool, torch.int32, torch.int64])
def test_integral_logits_keep_floating_probability_sampling(core, dtype):
    _, distributions, _ = core
    logits = torch.tensor([[0, 1, 0, 1]], dtype=dtype)
    probability = torch.sigmoid(logits)
    torch.manual_seed(31)
    bits = torch.rand_like(probability) < probability
    expected_next_draw = torch.rand(5)
    torch.manual_seed(31)
    sampled_bits = distributions.FactorialBernoulliUtil(logits).reparameterize(False)
    assert sampled_bits.dtype == probability.dtype == torch.float32
    torch.testing.assert_close(sampled_bits.bool(), bits)
    torch.testing.assert_close(torch.rand(5), expected_next_draw)
    for training in (True, False):
        torch.manual_seed(31)
        bits = torch.rand_like(probability) < probability
        smoothing_uniforms = torch.rand(logits.shape)
        expected_next_draw = torch.rand(5)
        torch.manual_seed(31)
        samples = distributions.MixtureGeneric(logits, 2.).reparameterize(training)
        assert samples.dtype == probability.dtype
        assert torch.all((samples > 0) & (samples < 1))
        torch.testing.assert_close(torch.rand(5), expected_next_draw)
        for sample, bit, uniform in zip(samples.flatten().tolist(),
                                        bits.flatten().tolist(), smoothing_uniforms.flatten().tolist()):
            component_value = 1. - sample if bit else sample
            assert math.isclose(exponential_cdf(component_value), uniform, abs_tol=2e-7)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("training", [True, False])
def test_mixture_samples_match_component_cdf_and_rng(core, dtype, training):
    _, distributions, _ = core
    logits = torch.tensor([[-.7, .4, 1.2]], dtype=dtype, requires_grad=True)
    torch.manual_seed(31)
    bits = torch.rand_like(logits) < torch.sigmoid(logits)
    smoothing_uniforms = torch.rand(logits.shape)
    expected_next_draw = torch.rand(5)
    torch.manual_seed(31)
    samples = distributions.MixtureGeneric(logits, 2.).reparameterize(training)
    assert samples.dtype == dtype and samples.device == logits.device
    assert samples.requires_grad
    torch.testing.assert_close(torch.rand(5), expected_next_draw)
    tolerance = {torch.float16: .002, torch.bfloat16: .02,
                 torch.float32: 2e-7, torch.float64: 2e-7}[dtype]
    for sample, bit, probability in zip(samples.detach().flatten().tolist(),
                                       bits.flatten().tolist(), smoothing_uniforms.flatten().tolist()):
        component_value = 1. - sample if bit else sample
        assert math.isclose(exponential_cdf(component_value), probability, abs_tol=tolerance)
    samples.sum().backward()
    assert logits.grad.dtype == dtype
    assert torch.all(torch.isfinite(logits.grad)) and torch.all(logits.grad > 0)


def mixture_cdf(logit, sample):
    probability = 1. / (1. + math.exp(-logit))
    return ((1. - probability) * exponential_cdf(sample)
            + probability * (1. - exponential_cdf(1. - sample)))


def mixture_quantile(logit, probability):
    lower, upper = 0., 1.
    for _ in range(80):
        middle = (lower + upper) / 2.
        if mixture_cdf(logit, middle) < probability:
            lower = middle
        else:
            upper = middle
    return (lower + upper) / 2.


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_implicit_gradient_matches_independent_inverse_cdf_difference(core, dtype):
    _, distributions, _ = core
    logits = torch.tensor([[-.7, .4, 1.2]], dtype=dtype, requires_grad=True)
    torch.manual_seed(31)
    samples = distributions.MixtureGeneric(logits, 2.).reparameterize(True)
    samples.sum().backward()
    for logit, sample, gradient in zip(logits.detach().flatten().tolist(),
                                      samples.detach().flatten().tolist(), logits.grad.flatten().tolist()):
        probability = mixture_cdf(logit, sample)
        epsilon = 1e-5
        numerical_derivative = (mixture_quantile(logit + epsilon, probability)
                                - mixture_quantile(logit - epsilon, probability)) / (2. * epsilon)
        assert math.isclose(gradient, numerical_derivative, rel_tol=3e-6, abs_tol=1e-8)
