"""Real MNIST QVAE generation, with legal spins at the solver boundary."""

import importlib.util
import math
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

for dependency in ("sklearn", "matplotlib", "seaborn", "gif", "imageio",
                   "tqdm", "torchvision", "torchmetrics"):
    pytest.importorskip(dependency)
import matplotlib
matplotlib.use("Agg")
from PIL import Image


class SpinSampler:
    """Yield configurable numbers of valid gauge-fixed Ising states."""

    def __init__(self, counts=(64,), bits=(0, 1), error=None):
        self.counts = counts
        self.bits = bits
        self.error = error
        self.calls = 0

    def solve(self, matrix):
        if self.error:
            raise self.error
        count = self.counts[min(self.calls, len(self.counts) - 1)]
        self.calls += 1
        assert len(matrix) == len(self.bits) + 1
        spins = np.array(self.bits) * 2 - 1
        states = np.tile(np.append(spins, 1), (count, 1))
        return states


@pytest.fixture
def actual_example(monkeypatch):
    example = Path(__file__).resolve().parents[1] / "example/qvae_mnist"
    package_name = "mnist_generation_test_model"
    spec = importlib.util.spec_from_file_location(
        package_name, example / "model/__init__.py",
        submodule_search_locations=[str(example / "model")],
    )
    package = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, package_name, package)
    spec.loader.exec_module(package)
    spec = importlib.util.spec_from_file_location(
        "mnist_generation_helpers_test", example / "utils/helpers.py"
    )
    helpers = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helpers)
    assert Path(helpers.__file__).resolve() == example / "utils/helpers.py"
    assert Path(sys.modules[package_name + ".model"].__file__).resolve() == example / "model/model.py"
    config = package.Config(encoder_hidden_nodes=[3], decoder_hidden_nodes=[3],
                            num_latent_units=2, dist_beta=.5)
    model = package.MnistQVAE(784, torch.nn.ReLU(), config)
    return helpers, model


def capture_decoder(model):
    inputs, outputs, grad_flags = [], [], []

    def before_decode(unused, arguments):
        inputs.append(arguments[0].detach().clone())
        grad_flags.append(torch.is_grad_enabled())

    def after_decode(unused, arguments, output):
        outputs.append(output.detach().clone())

    model.decoder.register_forward_pre_hook(before_decode)
    model.decoder.register_forward_hook(after_decode)
    return inputs, outputs, grad_flags


@pytest.mark.parametrize("requested,batch_size,counts,expected_calls", [
    (3, 64, (64,), 1), (65, 64, (64,), 2),
    (9, 3, (64,), 1), (8, 5, (2, 1, 6), 3),
])
def test_exact_count_uses_actual_sampler_rows_and_limits_decoder_batches(
        actual_example, monkeypatch, requested, batch_size, counts, expected_calls):
    helpers, model = actual_example
    sampler = SpinSampler(counts)
    monkeypatch.setattr(helpers, "init_qvae_sampler", lambda unused: sampler)
    latent, _, grad_flags = capture_decoder(model)

    images = helpers.generate_qvae_samples(model, .5, n_images=requested, batch_size=batch_size)

    assert images.shape == (requested, 784)
    assert sampler.calls == expected_calls
    assert sum(len(batch) for batch in latent) == requested
    assert max(len(batch) for batch in latent) <= batch_size
    assert not any(grad_flags)
    assert not images.requires_grad
    assert torch.isfinite(images).all()
    assert ((images >= 0) & (images <= 1)).all()


@pytest.mark.parametrize("beta", [.5, 2.])
def test_both_binary_states_follow_independent_truncated_exponential_cdf(
        actual_example, monkeypatch, beta):
    helpers, model = actual_example
    monkeypatch.setattr(helpers, "init_qvae_sampler", lambda unused: SpinSampler((65,)))
    latent, _, _ = capture_decoder(model)
    torch.manual_seed(42)
    expected_uniforms = torch.rand((65, 2)).double().numpy()
    torch.manual_seed(42)

    helpers.generate_qvae_samples(model, beta, n_images=65, batch_size=64)

    observed = torch.cat(latent).double().numpy()
    assert np.isfinite(observed).all()
    assert ((observed >= 0) & (observed <= 1)).all()
    # Independent conditional CDF: F_0(x)=(1-exp(-beta*x))/(1-exp(-beta)).
    # The conditional law for z=1 is the reflection of the law for z=0.
    reflected_noise = np.column_stack((observed[:, 0], 1 - observed[:, 1]))
    observed_uniforms = np.expm1(-beta * reflected_noise) / math.expm1(-beta)
    np.testing.assert_allclose(observed_uniforms, expected_uniforms, rtol=0, atol=1e-6)


@pytest.mark.parametrize("loss_type", ["bernoulli", "mse"])
def test_samples_keep_sigmoid_bias_semantics_with_real_decoder(actual_example, monkeypatch, loss_type):
    helpers, model = actual_example
    model.loss_type = loss_type
    model.set_train_bias(torch.full((784,), .6))
    monkeypatch.setattr(helpers, "init_qvae_sampler", lambda unused: SpinSampler())
    _, decoded, _ = capture_decoder(model)
    images = helpers.generate_qvae_samples(model, .5, n_images=64, batch_size=64)
    expected = torch.sigmoid(torch.cat(decoded) + model._train_bias)
    torch.testing.assert_close(images, expected, atol=0, rtol=0)


@pytest.mark.parametrize("grid_size,counts", [(1, (64,)), (3, (4,))])
def test_grid_uses_same_bounded_law_and_returns_exact_grid_size(
        actual_example, monkeypatch, tmp_path, grid_size, counts):
    helpers, model = actual_example
    sampler = SpinSampler(counts, bits=(1, 1))
    monkeypatch.setattr(helpers, "init_qvae_sampler", lambda unused: sampler)
    latent, decoded, grad_flags = capture_decoder(model)
    torch.manual_seed(42)
    expected_uniforms = torch.rand((grid_size ** 2, 2)).double().numpy()
    torch.manual_seed(42)

    images, filename = helpers.generate_qvae_images(model, tmp_path, grid_size=grid_size)

    assert images.shape == (grid_size ** 2, 784)
    observed = torch.cat(latent).double().numpy()
    assert ((observed >= 0) & (observed <= 1)).all()
    transformed = np.expm1(-.5 * (1 - observed)) / math.expm1(-.5)
    np.testing.assert_allclose(transformed, expected_uniforms, rtol=0, atol=1e-6)
    torch.testing.assert_close(images, torch.sigmoid(torch.cat(decoded) + model._train_bias),
                               atol=0, rtol=0)
    assert not any(grad_flags)
    assert not images.requires_grad
    assert Path(filename) == tmp_path / "generated_x.png"
    with Image.open(filename) as saved_image:
        saved_image.verify()


@pytest.mark.parametrize("name,value", [
    ("n_images", 0), ("n_images", -1), ("n_images", 1.5), ("n_images", True),
    ("batch_size", 0), ("batch_size", -1), ("batch_size", 1.5), ("batch_size", True),
    ("dist_beta", 0), ("dist_beta", -1), ("dist_beta", float("nan")),
    ("dist_beta", float("inf")),
])
def test_invalid_generation_requests_fail_before_sampling(actual_example, monkeypatch, name, value):
    helpers, model = actual_example
    sampler = SpinSampler()
    monkeypatch.setattr(helpers, "init_qvae_sampler", lambda unused: sampler)
    arguments = dict(dist_beta=.5, n_images=3, batch_size=64)
    arguments[name] = value
    with pytest.raises(ValueError, match=name):
        helpers.generate_qvae_samples(model, **arguments)
    assert sampler.calls == 0


def test_empty_sampler_result_is_rejected_without_decoding(actual_example, monkeypatch):
    helpers, model = actual_example
    sampler = SpinSampler((0,))
    monkeypatch.setattr(helpers, "init_qvae_sampler", lambda unused: sampler)
    latent, _, _ = capture_decoder(model)
    with pytest.raises(ValueError, match="empty"):
        helpers.generate_qvae_samples(model, .5, n_images=64, batch_size=64)
    assert sampler.calls == 1
    assert not latent


def test_solver_failure_is_propagated(actual_example, monkeypatch):
    helpers, model = actual_example
    sampler = SpinSampler(error=RuntimeError("offline solver failed"))
    monkeypatch.setattr(helpers, "init_qvae_sampler", lambda unused: sampler)
    with pytest.raises(RuntimeError, match="offline solver failed"):
        helpers.generate_qvae_samples(model, .5, n_images=64, batch_size=64)


@pytest.mark.parametrize("grid_size", [0, -1, 1.5, True])
def test_invalid_grid_size_is_rejected(actual_example, monkeypatch, tmp_path, grid_size):
    helpers, model = actual_example
    sampler = SpinSampler()
    monkeypatch.setattr(helpers, "init_qvae_sampler", lambda unused: sampler)
    with pytest.raises(ValueError, match="grid_size"):
        helpers.generate_qvae_images(model, tmp_path, grid_size=grid_size)
    assert sampler.calls == 0
