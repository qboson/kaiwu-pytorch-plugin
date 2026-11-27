"""A failed MNIST save must leave the preceding checkpoint usable."""

import importlib.util
from pathlib import Path
import sys
from types import ModuleType

import numpy as np
import pytest
import torch
from torch import nn


def load_example(monkeypatch, relative_path):
    """Load the checkout's actual writer without importing notebook entry points."""
    root = Path(__file__).resolve().parents[1] / "example/qvae_mnist"
    monkeypatch.syspath_prepend(str(root))
    path = root / relative_path
    spec = importlib.util.spec_from_file_location(f"kpp_test_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def model_tuner(monkeypatch, output_dir):
    module = load_example(monkeypatch, "trainer/model_tuner.py")
    tuner = module.ModelTuner()
    tuner.outpath = str(output_dir)
    tuner.register_model(nn.Linear(2, 1))
    return tuner


@pytest.mark.parametrize("writer", ["qvae", "mlp"])
def test_failed_mnist_save_preserves_existing_checkpoint(monkeypatch, tmp_path, writer):
    """Inject an I/O error after the writer has started producing bytes."""
    if writer == "qvae":
        save = model_tuner(monkeypatch, tmp_path).save_model
        destination = tmp_path / "model_test.pt"
    else:
        pytest.importorskip("sklearn")
        pytest.importorskip("tqdm")
        # Plot dependencies are unrelated to the checkpoint writer.
        plot_helpers = ModuleType("utils.helpers")
        plot_helpers.plot_training_curves = lambda **_kwargs: None
        monkeypatch.setitem(sys.modules, "utils.helpers", plot_helpers)
        module = load_example(monkeypatch, "downstream/classifier.py")
        estimator = module.MLPClassifier(
            hidden_dims=[3], output_dim=2, epochs_mlp=1,
            device="cpu", save_path=str(tmp_path),
        )
        monkeypatch.setattr(estimator, "_eval_mlp_epoch", lambda **_kwargs: (100.0, 0.1))
        data = np.arange(32, dtype=np.float32).reshape(16, 2) / 32
        labels = np.arange(16) % 2
        def save():
            estimator.fit(data, labels)

        destination = tmp_path / "best_mlp_classifier.pth"

    preceding = {"weight": torch.tensor([[1.0, 2.0]])}
    torch.save(preceding, destination)

    def fail_after_writing(_state, handle):
        if hasattr(handle, "write"):
            handle.write(b"interrupted")
        else:
            Path(handle).write_bytes(b"interrupted")
        raise OSError("simulated save failure")

    monkeypatch.setattr(torch, "save", fail_after_writing)
    with pytest.raises(OSError, match="simulated save failure"):
        save()

    restored = torch.load(destination, weights_only=True)
    torch.testing.assert_close(restored["weight"], preceding["weight"])
    assert list(tmp_path.iterdir()) == [destination]


def test_successful_save_replaces_the_checkpoint(monkeypatch, tmp_path):
    """An ordinary save still writes the registered model's state dictionary."""
    tuner = model_tuner(monkeypatch, tmp_path)
    destination = tmp_path / "model_test.pt"
    torch.save({"old": torch.tensor([0.0])}, destination)
    tuner.save_model()
    restored = torch.load(destination, weights_only=True)
    torch.testing.assert_close(restored, tuner._model.state_dict())
    assert list(tmp_path.iterdir()) == [destination]
