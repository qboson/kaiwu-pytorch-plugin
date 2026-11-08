"""Custom-data labels must not make unsupervised QVAE diagnostics fail."""

import importlib
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch


def offline_sampler(**kwargs):
    """Replace only the external Ising solver with legal deterministic states."""
    del kwargs

    def solve(matrix):
        """Return both constant spin assignments with a positive auxiliary spin."""
        states = np.ones((2, len(matrix)), dtype=np.float32)
        states[0, :-1] = -1
        return states

    return SimpleNamespace(solve=solve)


@pytest.fixture(name="example")
def example_modules(monkeypatch):
    """Import real example modules without a licensed sampler or model downloads."""
    for dependency in ("matplotlib", "pandas", "seaborn", "sklearn", "tqdm",
                       "torchvision", "torchmetrics", "gif", "imageio"):
        pytest.importorskip(dependency)
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    plt = pytest.importorskip("matplotlib.pyplot")

    folder = Path(__file__).resolve().parents[1] / "example" / "qvae_mnist"
    monkeypatch.syspath_prepend(str(folder))
    roots = {"model", "trainer", "utils"}
    saved = {name: module for name, module in sys.modules.items()
             if name.split(".")[0] in roots}
    for name in saved:
        del sys.modules[name]
    for name, exports in (
        ("kaiwu.classical", ("SimulatedAnnealingOptimizer",)),
        ("kaiwu.cim", ("CIMOptimizer", "PrecisionReducer")),
    ):
        boundary = ModuleType(name)
        for export in exports:
            setattr(boundary, export, offline_sampler)
        monkeypatch.setitem(sys.modules, name, boundary)
    try:
        trainer = importlib.import_module("trainer.trainer")
        config = importlib.import_module("model.config").Config
        assert Path(trainer.__file__).resolve() == folder / "trainer" / "trainer.py"
        # Rendering is unrelated to the diagnostic and expects 28x28 images.
        monkeypatch.setattr(trainer.Trainer, "_save_reconstruction", lambda *args: None)
        monkeypatch.setattr(trainer.Trainer, "_plot_training_curve", lambda *args: None)
        with torch.random.fork_rng(devices=[]):
            yield trainer, config
    finally:
        for name in list(sys.modules):
            if name.split(".")[0] in roots:
                del sys.modules[name]
        sys.modules.update(saved)
        plt.close("all")


@pytest.mark.parametrize("labels", [(-3, 5), (2, 1000000), (0, 1)])
def test_custom_label_diagnostics_preserve_training(example, tmp_path, monkeypatch, labels):
    """Label identifiers affect reporting, not the real unsupervised training path."""
    trainer_module, config_class = example
    messages = []
    monkeypatch.setattr(trainer_module.logger, "info", messages.append)
    inputs = np.linspace(0.1, 0.9, 32, dtype=np.float32).reshape(8, 4)
    original_labels = np.array([labels[0], labels[1]] * 4)
    original_copy = original_labels.copy()

    def train(label_values, output):
        torch.manual_seed(17)
        config = config_class(
            num_epochs=10, batch_size=8, output_dir=str(output),
            encoder_hidden_nodes=[3], decoder_hidden_nodes=[3],
            num_latent_units=2, dist_beta=0.5, kl_beta=0.01,
        )
        instance = trainer_module.Trainer(
            config, (inputs, label_values), (inputs[:4], label_values[:4]),
        )
        model, train_losses, test_losses = instance.train()
        assert len(train_losses) == len(test_losses) == 10
        assert np.isfinite(train_losses + test_losses).all()
        assert (output / "model_final_QVAE.pt").is_file()
        return model.state_dict(), train_losses, test_losses

    state, train_losses, test_losses = train(original_labels, tmp_path / "custom")
    expected = f"Test labels distribution: {{{labels[0]}: 2, {labels[1]}: 2}}"
    assert expected in messages
    reference, reference_train, reference_test = train(np.arange(8) % 2, tmp_path / "reference")
    assert train_losses == reference_train
    assert test_losses == reference_test
    for name, value in state.items():
        torch.testing.assert_close(value, reference[name], rtol=0, atol=0)
    np.testing.assert_array_equal(original_labels, original_copy)
