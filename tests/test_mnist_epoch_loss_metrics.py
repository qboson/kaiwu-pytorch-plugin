"""Actual QVAE epoch metrics are sample means, including uneven final batches."""

from itertools import product
from pathlib import Path
import json
import math
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Sampler, SubsetRandomSampler, TensorDataset

from kaiwu.torch_plugin import QVAE, RestrictedBoltzmannMachine


class UniformSpinAdapter:
    """Given legal negative samples; exact uniform Boltzmann only for zero weights."""

    def __init__(self, **kwargs):
        self.calls = 0

    def solve(self, matrix):
        self.calls += 1
        assert np.isfinite(matrix).all()
        return np.array([list(spins) + [1]
                         for spins in product((-1, 1), repeat=len(matrix) - 1)])


class InjectedQVAE(QVAE):
    """Use real neural networks and BM through the core component injection API."""

    def _create_encoder(self):
        raise AssertionError("Injected")
    def _create_decoder(self):
        raise AssertionError("Injected")
    def _create_bm(self):
        raise AssertionError("Injected")
    def _create_sampler(self, sampler_type):
        raise AssertionError("Injected")


@pytest.fixture
def example(monkeypatch):
    folder = Path(__file__).resolve().parents[1] / "example/qvae_mnist"
    monkeypatch.syspath_prepend(str(folder))
    roots = {"model", "trainer", "utils", "downstream"}
    saved = {name: module for name, module in sys.modules.items()
             if name.split(".")[0] in roots}
    for name in saved:
        del sys.modules[name]
    try:
        import model.model as mnist_module
        import trainer.trainer as trainer_module
        import trainer.model_tuner as tuner_module
        import kaiwu.torch_plugin.qvae as core_module
        from model import Config
        for module, path in [(mnist_module, "model/model.py"),
                             (trainer_module, "trainer/trainer.py"),
                             (tuner_module, "trainer/model_tuner.py")]:
            assert Path(module.__file__).resolve() == folder / path
        assert Path(core_module.__file__).resolve().is_relative_to(folder.parents[1] / "src")
        monkeypatch.setattr(mnist_module, "SimulatedAnnealingOptimizer", UniformSpinAdapter)
        monkeypatch.setattr(plt, "show", lambda: None)
        yield Config, trainer_module.Trainer, tuner_module.ModelTuner
    finally:
        for name in list(sys.modules):
            if name.split(".")[0] in roots:
                del sys.modules[name]
        sys.modules.update(saved)
        plt.close("all")


def _rows(count):
    return torch.tensor([[0., 0., 0.], [1., 1., 1.], [0., 1., 0.],
                         [1., 0., 1.], [1., 0., 0.], [0., 0., 1.], [1., 1., 0.]])[:count]


def _analytic_mean(rows, biases):
    """Independent per-example Bernoulli NLL using Python arithmetic."""
    return sum(sum(math.log1p(math.exp(bias)) - target * bias
                   for bias, target in zip(biases, row)) for row in rows.tolist()) / len(rows)


def _fixed_model(config, nonzero_bm=False):
    model = InjectedQVAE(3, None, config, encoder=nn.Linear(3, 2), decoder=nn.Linear(2, 3),
        bm=RestrictedBoltzmannMachine(1, 1, device="cpu"), sampler=UniformSpinAdapter())
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.decoder.bias.copy_(torch.tensor([.7, -.4, .1]))
        if nonzero_bm:
            model.encoder.bias.copy_(torch.tensor([100., -100.]))
            model.bm.linear_bias.copy_(torch.tensor([.2, -.3]))
    return model


def _tuner(example, rows, batch_size, two_stage, nonzero_bm=False):
    config_class, _, tuner_class = example
    config = config_class(kl_beta=0., weight_decay=.2 if nonzero_bm else 0., num_latent_units=2)
    model = _fixed_model(config, nonzero_bm)
    tuner = tuner_class(config)
    tuner.register_model(model)
    loader = DataLoader(TensorDataset(rows, torch.arange(len(rows)) % 2),
                        batch_size=batch_size, shuffle=False)
    tuner.register_dataLoaders(loader, loader)
    if two_stage:
        tuner.register_two_optimisers(torch.optim.Adam(
            list(model.encoder.parameters()) + list(model.decoder.parameters()), lr=0),
            torch.optim.Adam(model.bm.parameters(), lr=0))
    else:
        tuner.register_optimiser(torch.optim.SGD(model.parameters(), lr=0))
    return tuner, model


@pytest.mark.parametrize("count", [5, 7])
@pytest.mark.parametrize("batch_size", [1, 2, 4, 8])
@pytest.mark.parametrize("two_stage", [False, True])
def test_tuner_reports_analytic_sample_mean_in_both_optimizer_modes(
        example, count, batch_size, two_stage):
    rows = _rows(count)
    tuner, model = _tuner(example, rows, batch_size, two_stage)
    initial = {key: value.clone() for key, value in model.state_dict().items()}

    training = tuner.train(1)
    testing, inputs, reconstruction, labels = tuner.test()

    expected = _analytic_mean(rows, [.7, -.4, .1])
    assert training == pytest.approx(expected, rel=1e-6)
    assert testing.item() == pytest.approx(expected, rel=1e-6)
    last_count = count % batch_size or min(count, batch_size)
    assert inputs.shape == reconstruction.shape == (last_count, 3)
    assert labels is None
    assert all(torch.equal(value, initial[key]) for key, value in model.state_dict().items())
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               for parameter in model.parameters())
    assert model.sampler.calls == len(tuner.train_loader) * (3 if two_stage else 2)


@pytest.mark.parametrize("batch_size", [1, 2, 4, 8])
def test_two_stage_training_weights_real_bm_objective_separately(example, batch_size):
    """Hard [1,0] positives and supplied uniform negatives have energy gap -.25."""
    rows = _rows(7)
    tuner, model = _tuner(example, rows, batch_size, True, nonzero_bm=True)
    initial = {key: value.clone() for key, value in model.state_dict().items()}

    training = tuner.train(1)
    testing = tuner.test()[0].item()

    # The negative set is specified by the offline adapter, not claimed to be a
    # nonzero-parameter Boltzmann equilibrium distribution. Its mean energy is .05;
    # the real core bm_loss uses the hard positive energy -.2 and L2 penalty .013.
    penalty = .2 * .5 * (.2**2 + .3**2)
    vae = _analytic_mean(rows, [.7, -.4, .1]) + penalty
    bm = -.2 - .05 + penalty
    assert training == pytest.approx(vae + bm, rel=1e-6)
    assert testing == pytest.approx(vae, rel=1e-6)
    assert all(torch.equal(value, initial[key]) for key, value in model.state_dict().items())
    assert torch.isfinite(model.bm.linear_bias.grad).all()


class FixedIndicesSampler(Sampler):
    """An ordinary Torch sampler yielding repeated observation indices."""

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


def _sampled_loader(dataset, sampling):
    if sampling == "drop_last":
        return DataLoader(dataset, batch_size=3, drop_last=True), [0, 1, 2, 3, 4, 5]
    indices = [0, 2, 5] if sampling == "subset" else [1, 1, 4, 0, 4, 1]
    sampler = SubsetRandomSampler(indices) if sampling == "subset" else FixedIndicesSampler(indices)
    return DataLoader(dataset, batch_size=2, sampler=sampler), indices


@pytest.mark.parametrize("sampling", ["drop_last", "subset", "repeated"])
@pytest.mark.parametrize("two_stage", [False, True])
def test_registered_loaders_normalize_over_processed_observations(example, sampling, two_stage):
    rows = _rows(7)
    tuner, model = _tuner(example, rows, 2, two_stage)
    loader, indices = _sampled_loader(tuner.train_loader.dataset, sampling)
    tuner.register_dataLoaders(loader, loader)
    expected = _analytic_mean(rows[indices], [.7, -.4, .1])

    training = tuner.train(1)
    testing = tuner.test()[0].item()

    assert training == pytest.approx(expected, rel=1e-6)
    assert testing == pytest.approx(expected, rel=1e-6)
    assert model.sampler.calls == len(loader) * (3 if two_stage else 2)


def test_repeated_observations_normalize_both_dual_optimizer_terms(example):
    rows = _rows(7)
    tuner, _ = _tuner(example, rows, 2, True, nonzero_bm=True)
    loader, indices = _sampled_loader(tuner.train_loader.dataset, "repeated")
    tuner.register_dataLoaders(loader, loader)
    penalty = .2 * .5 * (.2**2 + .3**2)
    vae = _analytic_mean(rows[indices], [.7, -.4, .1]) + penalty
    bm = -.2 - .05 + penalty

    training = tuner.train(1)
    testing = tuner.test()[0].item()

    assert training == pytest.approx(vae + bm, rel=1e-6)
    assert testing == pytest.approx(vae, rel=1e-6)


@pytest.mark.parametrize("batch_size", [8, 32])
def test_full_trainer_histories_and_saved_metrics_are_analytic_means(example, tmp_path, batch_size):
    config_class, trainer_class, _ = example

    class FixedPredictionTrainer(trainer_class):
        """Only initialize actual parameters after normal model/optimizer setup."""
        def _setup_tuner(self):
            tuner = super()._setup_tuner()
            with torch.no_grad():
                for parameter in self.model.parameters():
                    parameter.zero_()
                self.model.decoder._layers[-1].bias.fill_(.7)
                self.model._train_bias.zero_()
            self.initial_state = {key: value.clone() for key, value in self.model.state_dict().items()}
            return tuner

    # 21 rows give a genuine uneven final batch of five at batch_size=8, also
    # satisfying the existing reconstruction helper's minimum of five images.
    pixels = np.repeat(np.resize(_rows(5)[:, :1].numpy(), (21, 1)), 784, axis=1)
    labels = np.arange(len(pixels)) % 2
    config = config_class(num_epochs=10, batch_size=batch_size, lr=0, bm_lr=0,
        kl_beta=0, weight_decay=0, num_latent_units=2, encoder_hidden_nodes=[],
        decoder_hidden_nodes=[], dist_beta=2., output_dir=str(tmp_path))
    trainer = FixedPredictionTrainer(config, (pixels, labels), (pixels, labels))

    model, training, testing = trainer.train()
    trainer.save_results()

    expected = 784 * (math.log1p(math.exp(.7)) - float(pixels.mean()) * .7)
    assert training == pytest.approx([expected] * 10, rel=1e-6)
    assert testing == pytest.approx([expected] * 10, rel=1e-6)
    assert all(torch.equal(value, trainer.initial_state[key]) for key, value in model.state_dict().items())
    saved = json.loads((tmp_path / "results.json").read_text())
    assert saved["train_losses"] == training and saved["test_losses"] == testing
    assert (tmp_path / "qvae_training_curve.png").stat().st_size > 0
    assert len(list(tmp_path.glob("reconstruction_epoch_*.png"))) == 10
    checkpoint = torch.load(tmp_path / "model_final_QVAE.pt", weights_only=True)
    assert all(torch.equal(value, checkpoint[key]) for key, value in model.state_dict().items())
    for optimizer in (trainer.tuner._vae_optimiser, trainer.tuner._bm_optimiser):
        assert optimizer.state
        assert all(state["step"].item() == 10 * len(trainer.train_loader)
                   for state in optimizer.state.values())
