"""MNIST network L2 must reach the complete objective and actual tuner updates."""
import importlib.util
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from kaiwu.classical import BruteForceOptimizer


ROOT = Path(__file__).resolve().parents[1]


def _model(coefficient):
    from example.qvae_mnist.model import Config, MnistQVAE

    torch.manual_seed(71)
    config = Config(num_latent_units=4, encoder_hidden_nodes=[6],
                    decoder_hidden_nodes=[6], weight_decay=coefficient,
                    kl_beta=0.02)
    model = MnistQVAE(8, config.activation_fct, config)
    model.sampler = BruteForceOptimizer()
    with torch.no_grad():
        model.bm.quadratic_coef.copy_(torch.linspace(
            -0.2, 0.3, model.bm.quadratic_coef.numel()
        ).reshape_as(model.bm.quadratic_coef))
        model.bm.linear_bias.copy_(torch.linspace(
            -0.1, 0.2, model.bm.linear_bias.numel()
        ).reshape_as(model.bm.linear_bias))
    model.set_dataset_mean(torch.tensor(0.4))
    model.set_train_bias(torch.tensor(0.4))
    return model


def _expected_loss(model, inputs, reconstruction, posterior):
    reconstruction_loss = F.binary_cross_entropy_with_logits(
        reconstruction, inputs, reduction="none"
    ).sum(1).mean()
    network_weights = [layer.weight for network in (model.encoder, model.decoder)
                       for layer in network.modules() if isinstance(layer, nn.Linear)]
    network_l2 = model.weight_decay * sum(weight.square().sum()
                                         for weight in network_weights)
    bm_l2 = model.weight_decay * (
        model.bm.quadratic_coef.square().sum()
        + 0.5 * model.bm.linear_bias.square().sum()
    )
    return (reconstruction_loss + model.kl_beta * model._kl_dist_from(posterior).mean()
            + network_l2 + bm_l2)


@pytest.mark.parametrize("coefficient", [0.0, 0.01])
def test_mnist_complete_loss_and_gradients_include_network_regularization(coefficient):
    model = _model(coefficient)
    inputs = torch.linspace(0.1, 0.9, 32).reshape(4, 8)
    reconstruction, posterior, _, _ = model(inputs)
    actual = model.loss(inputs, reconstruction, posterior)
    expected = _expected_loss(model, inputs, reconstruction, posterior)
    parameters = list(model.parameters())
    actual_gradients = torch.autograd.grad(actual, parameters, retain_graph=True)
    expected_gradients = torch.autograd.grad(expected, parameters)
    torch.testing.assert_close(actual, expected)
    for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients):
        torch.testing.assert_close(actual_gradient, expected_gradient, atol=1e-6, rtol=1e-5)


def test_mnist_tuner_updates_match_independent_regularized_objective(monkeypatch):
    # ModelTuner uses the example's existing absolute utils imports.
    monkeypatch.syspath_prepend(str(ROOT / "example/qvae_mnist"))
    spec = importlib.util.spec_from_file_location(
        "mnist_regularization_tuner", ROOT / "example/qvae_mnist/trainer/model_tuner.py"
    )
    tuner_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tuner_module)
    actual, expected = _model(0.01), _model(0.01)
    inputs = torch.linspace(0.1, 0.9, 32).reshape(4, 8)
    loader = DataLoader(TensorDataset(inputs, torch.arange(4)), batch_size=2)

    def optimizers(model):
        return (torch.optim.Adam(list(model.encoder.parameters())
                                 + list(model.decoder.parameters()), lr=8e-4),
                torch.optim.Adam(model.bm.parameters(), lr=8e-4))

    vae_optimizer, bm_optimizer = optimizers(actual)
    tuner = tuner_module.ModelTuner(actual.config)
    tuner.register_model(actual)
    tuner.register_dataLoaders(loader, loader)
    tuner.register_two_optimisers(vae_optimizer, bm_optimizer)
    torch.manual_seed(123)
    tuner.train(1)

    vae_optimizer, bm_optimizer = optimizers(expected)
    torch.manual_seed(123)
    for batch, _ in loader:
        vae_optimizer.zero_grad()
        reconstruction, posterior, q, _ = expected(batch)
        _expected_loss(expected, batch, reconstruction, posterior).backward()
        vae_optimizer.step()
        bm_optimizer.zero_grad()
        expected.bm_loss(q.detach(), expected.weight_decay).backward()
        bm_optimizer.step()
    for (_, actual_parameter), (_, expected_parameter) in zip(
            actual.named_parameters(), expected.named_parameters()):
        torch.testing.assert_close(actual_parameter, expected_parameter, atol=1e-6, rtol=1e-5)


def test_replacement_networks_without_regularizer_keep_base_bm_term():
    model = _model(0.01)
    model.encoder = nn.Linear(8, 4)
    model.decoder = nn.Linear(4, 8)
    expected = 0.01 * (model.bm.quadratic_coef.square().sum()
                       + 0.5 * model.bm.linear_bias.square().sum())
    torch.testing.assert_close(model._weight_decay_loss(), expected)
