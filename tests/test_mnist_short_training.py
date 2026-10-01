"""Actual short MNIST training and reconstruction/checkpoint scheduling."""

from itertools import product
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class OfflineSampler:
    """Enumerate legal Ising states at the external solver boundary."""

    def __init__(self, **kwargs):
        self.calls = 0

    def solve(self, matrix):
        self.calls += 1
        spins = np.array(list(product((-1., 1.), repeat=len(matrix) - 1)))
        states = np.column_stack((spins, np.ones(len(spins))))
        energies = -np.einsum('bi,ij,bj->b', states, matrix, states)
        return states[np.argsort(energies)[:2]]


@pytest.fixture
def example(monkeypatch):
    folder = Path(__file__).resolve().parents[1] / 'example/qvae_mnist'
    monkeypatch.syspath_prepend(str(folder))
    roots = {'model', 'trainer', 'utils', 'downstream'}
    saved_modules = {name: module for name, module in sys.modules.items()
                     if name.split('.')[0] in roots}
    for name in saved_modules:
        del sys.modules[name]
    try:
        import model.model as mnist_module
        import trainer.trainer as trainer_module
        import trainer.model_tuner as tuner_module
        import kaiwu.torch_plugin.qvae as qvae_module

        assert Path(mnist_module.__file__).resolve() == folder / 'model/model.py'
        assert Path(trainer_module.__file__).resolve() == folder / 'trainer/trainer.py'
        assert Path(tuner_module.__file__).resolve() == folder / 'trainer/model_tuner.py'
        source = folder.parents[1] / 'src'
        assert Path(qvae_module.__file__).resolve().is_relative_to(source)
        monkeypatch.setattr(mnist_module, 'SimulatedAnnealingOptimizer', OfflineSampler)
        monkeypatch.setattr(plt, 'show', lambda: None)
        yield mnist_module, trainer_module, tuner_module
    finally:
        for name in list(sys.modules):
            if name.split('.')[0] in roots:
                del sys.modules[name]
        sys.modules.update(saved_modules)
        plt.close('all')


@pytest.mark.parametrize('epochs', [*range(1, 10), 10, 21, 50])
def test_actual_training_saves_reconstructions_and_final_checkpoint(example, tmp_path,
                                                                  monkeypatch, epochs):
    mnist_module, trainer_module, tuner_module = example
    from model import Config

    torch.manual_seed(7)
    pixels = np.linspace(.05, .95, 8 * 784).reshape(8, 784)
    labels = np.arange(8) % 2
    config = Config(num_epochs=epochs, batch_size=8, output_dir=str(tmp_path),
                    encoder_hidden_nodes=[4], decoder_hidden_nodes=[4],
                    num_latent_units=2, dist_beta=.5, kl_beta=.01)
    trainer = trainer_module.Trainer(config, (pixels, labels), (pixels, labels))
    initial_parameters = {}
    setup_tuner = trainer._setup_tuner

    def record_initial_parameters():
        tuner = setup_tuner()
        initial_parameters.update({name: value.detach().clone()
                                   for name, value in trainer.model.named_parameters()})
        return tuner

    monkeypatch.setattr(trainer, '_setup_tuner', record_initial_parameters)
    saved_epochs = []
    save_reconstruction = trainer._save_reconstruction

    def record_reconstruction(epoch, input_data, output_data):
        saved_epochs.append(epoch)
        save_reconstruction(epoch, input_data, output_data)
        plt.close('all')

    monkeypatch.setattr(trainer, '_save_reconstruction', record_reconstruction)
    model, training_losses, validation_losses = trainer.train()

    assert isinstance(model, mnist_module.MnistQVAE)
    assert isinstance(trainer.tuner, tuner_module.ModelTuner)
    assert len(training_losses) == len(validation_losses) == epochs
    assert np.isfinite(training_losses).all()
    assert np.isfinite(validation_losses).all()
    assert trainer.tuner._use_two_optimisers
    for optimiser in [trainer.tuner._vae_optimiser, trainer.tuner._bm_optimiser]:
        assert isinstance(optimiser, torch.optim.Adam)
        assert optimiser.state
        assert all(state['step'].item() == epochs for state in optimiser.state.values())
    for prefix in ['encoder.', 'decoder.', 'bm.']:
        parameters = [(name, value) for name, value in model.named_parameters()
                      if name.startswith(prefix)]
        assert all(value.grad is not None and torch.isfinite(value.grad).all()
                   for _, value in parameters)
        assert any(not torch.equal(value, initial_parameters[name])
                   for name, value in parameters)
    assert model.sampler.calls == 3 * epochs

    if epochs < 10:
        expected_epochs = list(range(1, epochs + 1))
    else:
        expected_epochs = list(range(epochs // 10, epochs + 1, epochs // 10))
        if expected_epochs[-1] != epochs:
            expected_epochs.append(epochs)
    assert saved_epochs == expected_epochs
    assert sorted(int(path.stem.removeprefix('reconstruction_epoch_'))
                  for path in tmp_path.glob('reconstruction_epoch_*.png')) == expected_epochs
    assert all((tmp_path / f'reconstruction_epoch_{epoch}.png').stat().st_size > 0
               for epoch in expected_epochs)
    assert (tmp_path / 'qvae_training_curve.png').stat().st_size > 0
    checkpoint = torch.load(tmp_path / 'model_final_QVAE.pt', weights_only=True)
    assert set(checkpoint) == set(model.state_dict())
    for name, value in model.state_dict().items():
        torch.testing.assert_close(checkpoint[name], value, rtol=0, atol=0)
