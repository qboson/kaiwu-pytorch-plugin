"""Energy scoring depends on logits, never posterior smoothing samples."""
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from kaiwu.torch_plugin import BoltzmannMachine
from example.qvae_cell.models import CellQVAE


@pytest.mark.parametrize("loss_type", ["mse", "bernoulli"])
def test_energy_avoids_sampling_and_preserves_rng(loss_type):
    config = SimpleNamespace(num_latent_units=2, loss_type=loss_type, dist_beta=1e-8, kl_beta=0., weight_decay=0.)
    model = CellQVAE(3, None, config, encoder=torch.nn.Linear(3, 2),
                     decoder=torch.nn.Linear(2, 3), bm=BoltzmannMachine(2, device="cpu"),
                     sampler=object()).eval()
    model.set_dataset_mean(torch.tensor([0.1, 0.3, 0.5]))
    x = torch.tensor([[0.2, 0.6, 0.8], [0.9, 0.4, 0.2]])
    centered = x - model._dataset_mean if loss_type == "bernoulli" else x
    expected = model.bm((model.encoder(centered) > 0).float())
    rng = torch.get_rng_state().clone()
    torch.testing.assert_close(model.energy(x, loss_type), expected)
    assert torch.equal(rng, torch.get_rng_state())
    model.posterior = Mock(side_effect=AssertionError("energy must not sample"))
    actual = model.energy(x, loss_type)
    torch.testing.assert_close(actual, expected)
    assert torch.equal(rng, torch.get_rng_state())
    with pytest.raises(ValueError, match="loss_type"):
        model.energy(x, "other")


def test_forward_still_uses_posterior_samples():
    config = SimpleNamespace(num_latent_units=2, loss_type="mse", dist_beta=1., kl_beta=0., weight_decay=0.)
    model = CellQVAE(3, None, config, encoder=torch.nn.Linear(3, 2),
                     decoder=torch.nn.Linear(2, 3), bm=BoltzmannMachine(2, device="cpu"),
                     sampler=object())
    sentinel = torch.full((2, 2), 0.25)
    model.posterior = Mock(return_value=(object(), sentinel))
    output, _, q, zeta = model(torch.ones(2, 3))
    model.posterior.assert_called_once_with(q, config.dist_beta)
    torch.testing.assert_close(zeta, sentinel)
    torch.testing.assert_close(output, model.decoder(sentinel))

