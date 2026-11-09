"""Loss-component metrics used by QVAE training and validation logs."""

import importlib.util
from itertools import product
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from kaiwu.torch_plugin import BoltzmannMachine, QVAE
from kaiwu.torch_plugin.qvae_dist_util import MixtureGeneric


class EnumeratingSampler:
    """Return all binary states through the BM's ordinary Ising solver protocol."""

    def solve(self, matrix):
        spins = np.array(list(product((-1.0, 1.0), repeat=len(matrix) - 1)))
        return np.column_stack([spins, np.ones(len(spins))])


@pytest.fixture(params=["core", "cell"])
def model(request):
    """Use the real core loss, including its single-cell subclass inheritance."""
    if request.param == "cell":
        path = Path(__file__).resolve().parents[1] / "example/qvae_cell/models.py"
        spec = importlib.util.spec_from_file_location("loss_metric_cell_models", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        model_class = module.CellQVAE
        extra = {"n_batches": 1}
    else:
        model_class = QVAE
        extra = {}
    config = SimpleNamespace(
        num_latent_units=2, loss_type="mse", dist_beta=2.0,
        kl_beta=0.3, weight_decay=0.04,
    )
    encoder = torch.nn.Linear(2, 2)
    decoder = torch.nn.Linear(2 + extra.get("n_batches", 0), 2)
    with torch.no_grad():
        encoder.weight.copy_(torch.tensor([[0.3, -0.2], [-0.4, 0.5]]))
        encoder.bias.copy_(torch.tensor([0.1, -0.2]))
        decoder.weight.fill_(0.25)
        decoder.bias.copy_(torch.tensor([-0.3, 0.2]))
    bm = BoltzmannMachine(
        2, quadratic_coef=torch.tensor([[0.0, 0.3], [0.0, 0.0]]),
        linear_bias=torch.tensor([0.2, -0.1]), device=torch.device("cpu"),
    )
    return model_class(
        2, torch.nn.ReLU(), config, encoder=encoder, decoder=decoder,
        bm=bm, sampler=EnumeratingSampler(), **extra,
    )


def reference_terms(x, output, logits, loss_type, bm):
    """Independent probability-space reconstruction, entropy and two-node energy."""
    if loss_type == "mse":
        reconstruction = ((output - x) ** 2).sum() / len(x)
    else:
        probability = torch.sigmoid(output)
        reconstruction = -(
            x * torch.log(probability) + (1.0 - x) * torch.log1p(-probability)
        ).sum() / len(x)
    probability = torch.sigmoid(logits)
    entropy = -(
        probability * torch.log(probability)
        + (1.0 - probability) * torch.log1p(-probability)
    ).sum(dim=1).mean()
    positive_energy = (
        -bm.linear_bias[0] * probability[:, 0]
        - bm.linear_bias[1] * probability[:, 1]
        - bm.quadratic_coef[0, 1] * probability[:, 0] * probability[:, 1]
    ).mean()
    # E(00), E(01), E(10), E(11) = 0, 0.1, -0.2, -0.4.
    negative_energy = -0.5 * bm.linear_bias.sum() - 0.25 * bm.quadratic_coef[0, 1]
    kl = positive_energy - negative_energy - entropy
    return reconstruction, kl


@pytest.mark.parametrize("loss_type", ["mse", "bernoulli"])
def test_loss_records_detached_unweighted_terms_and_preserves_gradients(model, loss_type):
    """Cell epoch logging can read the raw terms without retaining a loss graph."""
    model.config.loss_type = loss_type
    model.eval()
    x = torch.tensor([[0.1, 0.9], [0.4, 0.2], [0.8, 0.3]])
    if hasattr(model, "n_batches"):
        output, posterior, logits, _ = model(x, torch.zeros(len(x), dtype=torch.long))
    else:
        output, posterior, logits, _ = model(x)
    expected_reconstruction, expected_kl = reference_terms(x, output, logits, loss_type, model.bm)
    expected_weight_decay = 0.04 * (
        model.bm.quadratic_coef.square().sum() + 0.5 * model.bm.linear_bias.square().sum()
    )
    expected_total = expected_reconstruction + 0.3 * expected_kl + expected_weight_decay

    actual_total = model.loss(x, output, posterior)

    for metric, expected in (
        (model.last_recon_loss, expected_reconstruction),
        (model.last_kl_loss, expected_kl),
    ):
        assert isinstance(metric, torch.Tensor)
        assert metric.shape == torch.Size([])
        assert not metric.requires_grad
        assert metric.grad_fn is None
        assert metric.device == actual_total.device
        assert metric.dtype == actual_total.dtype
        torch.testing.assert_close(metric, expected)
        assert np.isfinite(metric.item())  # The Cell trainer's logging operation.
    torch.testing.assert_close(actual_total, expected_total)
    actual_gradients = torch.autograd.grad(actual_total, tuple(model.parameters()), retain_graph=True)
    expected_gradients = torch.autograd.grad(expected_total, tuple(model.parameters()))
    for actual, expected in zip(actual_gradients, expected_gradients):
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("loss_type", ["mse", "bernoulli"])
def test_each_loss_call_refreshes_the_metrics_in_training_and_validation(model, loss_type):
    """Logging values belong to the current batch, including no-grad validation."""
    model.config.loss_type = loss_type
    x = torch.tensor([[0.1, 0.9], [0.4, 0.2]])
    first_logits = torch.tensor([[-0.5, 0.5], [-0.5, 0.5]], requires_grad=True)
    model.loss(x, torch.zeros_like(x, requires_grad=True), MixtureGeneric(first_logits, 2.0))
    first_reconstruction, first_kl = model.last_recon_loss, model.last_kl_loss

    output = torch.full_like(x, 2.0)
    logits = torch.tensor([[0.2, 1.4], [0.2, 1.4]])
    expected_reconstruction, expected_kl = reference_terms(x, output, logits, loss_type, model.bm)
    model.eval()
    with torch.no_grad():
        model.loss(x, output, MixtureGeneric(logits, 2.0))

    torch.testing.assert_close(model.last_recon_loss, expected_reconstruction)
    torch.testing.assert_close(model.last_kl_loss, expected_kl)
    assert model.last_recon_loss is not first_reconstruction
    assert model.last_kl_loss is not first_kl
    assert model.last_recon_loss.item() != first_reconstruction.item()
    assert model.last_kl_loss.item() != first_kl.item()
