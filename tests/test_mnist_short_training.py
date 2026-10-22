"""Actual short MNIST training and reconstruction/checkpoint scheduling."""

from itertools import product
from pathlib import Path
import sys

import numpy as np
import pytest
import torch
from PIL import Image

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
        assert Path(trainer_module.plot_MNIST_output.__code__.co_filename).resolve() == (
            folder / 'utils/helpers.py')
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


def _assert_png(path):
    with Image.open(path) as image:
        assert image.format == 'PNG'
        image.verify()
    with Image.open(path) as image:
        image.load()
        assert image.size == (1000, 450)
        assert np.asarray(image).std() > 0


def test_actual_reconstruction_helper_supports_a_single_column(example, tmp_path):
    _, trainer_module, _ = example
    original = np.linspace(0, 1, 784).reshape(1, 784)
    reconstructed = original[:, ::-1].copy()
    path = tmp_path / 'single-column.png'

    trainer_module.plot_MNIST_output(original, reconstructed, n_samples=1, output=path)

    axes = plt.gcf().axes
    assert len(axes) == 2
    np.testing.assert_array_equal(axes[0].images[0].get_array(), original.reshape(28, 28))
    np.testing.assert_array_equal(axes[1].images[0].get_array(), reconstructed.reshape(28, 28))
    _assert_png(path)


@pytest.mark.parametrize('last_batch_count', [1, 2, 3, 4, 5])
def test_actual_small_final_batch_training_and_checkpoint_reload(example, tmp_path,
                                                               last_batch_count):
    mnist_module, trainer_module, tuner_module = example
    from model import Config

    class ObservedTrainer(trainer_module.Trainer):
        """Observe actual saved figures without replacing training or plotting."""

        def _setup_tuner(self):
            tuner = super()._setup_tuner()
            self.initial = {name: value.detach().clone()
                            for name, value in self.model.named_parameters()}
            self.saved_epochs = []
            return tuner

        def _save_reconstruction(self, epoch, input_data, output_data):
            super()._save_reconstruction(epoch, input_data, output_data)
            axes = plt.gcf().axes
            assert len(input_data) == len(output_data) == last_batch_count
            assert len(axes) == 2 * last_batch_count
            for index in range(last_batch_count):
                np.testing.assert_array_equal(axes[2 * index].images[0].get_array(),
                                              input_data[index].numpy().reshape(28, 28))
                np.testing.assert_array_equal(axes[2 * index + 1].images[0].get_array(),
                                              output_data[index].numpy().reshape(28, 28))
            _assert_png(Path(self.output_dir) / f'reconstruction_epoch_{epoch}.png')
            self.saved_epochs.append(epoch)
            plt.close('all')

    torch.manual_seed(7)
    count = 8 + last_batch_count
    pixels = np.linspace(.05, .95, count * 784).reshape(count, 784)
    labels = np.arange(count) % 2
    config = Config(num_epochs=2, batch_size=8, output_dir=str(tmp_path / 'training'),
                    encoder_hidden_nodes=[4], decoder_hidden_nodes=[4],
                    num_latent_units=2, dist_beta=.5, kl_beta=.01)
    trainer = ObservedTrainer(config, (pixels, labels), (pixels, labels))

    model, training, validation = trainer.train()

    assert isinstance(model, mnist_module.MnistQVAE)
    assert isinstance(trainer.tuner, tuner_module.ModelTuner)
    assert len(training) == len(validation) == 2
    assert np.isfinite(training).all() and np.isfinite(validation).all()
    assert trainer.saved_epochs == [1, 2]
    assert model.sampler.calls == 12
    assert all(state['step'].item() == 4 for optimizer in (
        trainer.tuner._vae_optimiser, trainer.tuner._bm_optimiser)
        for state in optimizer.state.values())
    for prefix in ['encoder.', 'decoder.', 'bm.']:
        assert any(not torch.equal(value, trainer.initial[name])
                   for name, value in model.named_parameters() if name.startswith(prefix))
    checkpoint_path = Path(trainer.output_dir) / 'model_final_QVAE.pt'
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(checkpoint[name], value, rtol=0, atol=0)

    # Existing ModelTuner loading restores model weights; it does not restore Adam
    # state or promise identical optimizer continuation across new Trainer objects.
    config.output_dir = str(tmp_path / 'reloaded')
    reloaded = ObservedTrainer(config, (pixels, labels), (pixels, labels))
    reloaded._setup_data()
    reloaded._create_model()
    reloaded._setup_tuner()
    reloaded.tuner.infile = str(checkpoint_path)
    reloaded.tuner.load_model()
    assert not reloaded.model.training
    for name, value in reloaded.model.state_dict().items():
        torch.testing.assert_close(checkpoint[name], value, rtol=0, atol=0)

    train_loss = reloaded.tuner.train(3)
    test_loss, inputs, outputs, _ = reloaded.tuner.test()
    reloaded._save_reconstruction(3, inputs, outputs)
    reloaded.tuner.save_model(config_string='resumed_QVAE')

    assert np.isfinite(train_loss) and torch.isfinite(test_loss)
    assert reloaded.saved_epochs == [3]
    assert reloaded.model.sampler.calls == 6
    assert any(not torch.equal(value, checkpoint[name])
               for name, value in reloaded.model.state_dict().items())
    resumed = torch.load(Path(reloaded.output_dir) / 'model_resumed_QVAE.pt', weights_only=True)
    for name, value in reloaded.model.state_dict().items():
        torch.testing.assert_close(resumed[name], value, rtol=0, atol=0)
