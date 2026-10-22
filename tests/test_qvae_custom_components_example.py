"""Contract tests for the QVAE custom-components example."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import torch

EXAMPLE_PATH = (
    Path(__file__).resolve().parents[1]
    / "example"
    / "qvae_custom_components"
    / "custom_qvae.py"
)


def _load_example():
    spec = importlib.util.spec_from_file_location(
        "qvae_custom_components_example", EXAMPLE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


example = _load_example()


def test_injected_components_are_reused() -> None:
    encoder = example.ToyEncoder()
    decoder = example.ToyDecoder()
    num_visible = example.LATENT_DIM // 2
    bm = example.RestrictedBoltzmannMachine(
        num_visible=num_visible, num_hidden=example.LATENT_DIM - num_visible
    )
    sampler = example.ConstantSampler()
    model = example.InjectedQVAE(
        example.INPUT_DIM,
        None,
        example.build_config(),
        encoder=encoder,
        decoder=decoder,
        bm=bm,
        sampler=sampler,
    )
    assert model.encoder is encoder
    assert model.decoder is decoder
    assert model.bm is bm
    assert model.sampler is sampler


def test_forward_shapes() -> None:
    model = example.build_model()
    inputs = torch.rand(example.BATCH, example.INPUT_DIM)
    recon, posterior, q, zeta = model(inputs)
    assert recon.shape == (example.BATCH, example.INPUT_DIM)
    assert q.shape == (example.BATCH, example.LATENT_DIM)
    assert zeta.shape == (example.BATCH, example.LATENT_DIM)
    assert posterior.logit_mu.shape == (example.BATCH, example.LATENT_DIM)


def test_encoder_sees_mean_centered_input() -> None:
    model = example.build_model()
    model.set_dataset_mean(torch.full((example.INPUT_DIM,), 0.5))
    model.set_train_bias(torch.full((example.INPUT_DIM,), 0.5))
    inputs = torch.rand(example.BATCH, example.INPUT_DIM)
    model.encoder.seen.clear()
    model(inputs)
    centered = model.encoder.seen[-1]
    assert torch.allclose(centered, inputs - 0.5)


def test_loss_backpropagates_to_encoder_and_decoder() -> None:
    model = example.build_model()
    inputs = torch.rand(example.BATCH, example.INPUT_DIM)
    recon, posterior, _, _ = model(inputs)
    loss = model.loss(inputs, recon, posterior)
    assert torch.isfinite(loss)
    loss.backward()
    assert all(parameter.grad is not None for parameter in model.encoder.parameters())
    assert all(parameter.grad is not None for parameter in model.decoder.parameters())


def test_energy_uses_the_latent_bernoulli_states() -> None:
    model = example.build_model()
    model.set_dataset_mean(torch.zeros(example.INPUT_DIM))
    model.set_train_bias(torch.zeros(example.INPUT_DIM))
    energy = model.energy(torch.rand(example.BATCH, example.INPUT_DIM))
    assert energy.shape == (example.BATCH,)
    assert bool(torch.isfinite(energy).all())
