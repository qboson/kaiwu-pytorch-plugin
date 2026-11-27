"""Setting the QVAE dataset mean must also apply the Bernoulli train bias."""

import types

import torch

from kaiwu.torch_plugin import QVAE, RestrictedBoltzmannMachine


class OnesEncoder(torch.nn.Module):
    """Return positive latent logits for every sample."""

    def forward(self, x):
        return torch.ones(x.size(0), 4, dtype=x.dtype, device=x.device)


class ZeroDecoder(torch.nn.Module):
    """Return zero reconstruction logits for every sample."""

    def forward(self, z):
        return torch.zeros(z.size(0), 8, dtype=z.dtype, device=z.device)


class UnusedSampler:
    """Sampler placeholder; these tests never draw BM samples."""

    def solve(self, matrix):
        raise AssertionError("the sampler must not be used by this test")


def make_model():
    config = types.SimpleNamespace(
        num_latent_units=4,
        loss_type="bernoulli",
        dist_beta=1.0,
        kl_beta=1e-6,
        weight_decay=0.01,
    )
    return QVAE(
        input_dimension=8,
        activation_fct=None,
        config=config,
        encoder=OnesEncoder(),
        decoder=ZeroDecoder(),
        bm=RestrictedBoltzmannMachine(2, 2),
        sampler=UnusedSampler(),
    )


def test_set_dataset_mean_refreshes_the_train_bias():
    """The documented bias correction is applied by set_dataset_mean alone."""
    model = make_model()
    model.set_dataset_mean(torch.full((8,), 0.25))

    expected = torch.full((8,), -torch.log(torch.tensor(1 / 0.25 - 1)))
    torch.testing.assert_close(model._train_bias, expected)


def test_set_dataset_mean_accepts_a_scalar_and_is_idempotent():
    """Scalar means broadcast and a repeated set_train_bias keeps the same bias."""
    model = make_model()
    model.set_dataset_mean(0.25)
    first = model._train_bias.clone()
    model.set_train_bias(0.25)

    torch.testing.assert_close(model._train_bias, first)


def test_forward_uses_the_bias_from_set_dataset_mean():
    """Reconstruction logits include the bias without a separate setter call."""
    model = make_model()
    model.set_dataset_mean(torch.full((8,), 0.25))

    recon, _, _, _ = model(torch.zeros(2, 8))
    expected = model._train_bias.expand_as(recon)
    torch.testing.assert_close(recon, expected)


def test_set_dataset_mean_none_clears_the_mean_without_touching_the_bias():
    """None only clears the mean, matching the base setter's contract."""
    model = make_model()
    model.set_dataset_mean(0.25)
    bias = model._train_bias.clone()

    model.set_dataset_mean(None)

    assert model._dataset_mean is None
    torch.testing.assert_close(model._train_bias, bias)