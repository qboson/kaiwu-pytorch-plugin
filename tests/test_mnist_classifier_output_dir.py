"""Standalone MLP training creates its requested artifact directory."""

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest
import torch
from PIL import Image


def load_source(name, path):
    """Load checked-out example source without importing the full training pipeline."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(name="classifier")
def fixture_classifier(monkeypatch):
    """Use real plotting helpers, neural modules and optimizer on CPU."""
    for dependency in ("matplotlib", "pandas", "sklearn", "torchvision",
                       "torchmetrics", "gif", "imageio"):
        pytest.importorskip(dependency)
    folder = Path(__file__).resolve().parents[1] / "example/qvae_mnist"
    helpers = load_source("mlp_directory_helpers", folder / "utils/helpers.py")
    monkeypatch.setitem(sys.modules, "utils.helpers", helpers)
    helpers.plt.switch_backend("Agg")
    module = load_source("mlp_directory_classifier", folder / "downstream/classifier.py")
    with torch.random.fork_rng(devices=[]):
        yield module.MLPClassifier
    helpers.plt.close("all")


@pytest.mark.parametrize("directory", ["new", "existing", "current"])
def test_fit_saves_real_checkpoint_and_curve_in_requested_directory(
        classifier, tmp_path, monkeypatch, directory):
    """A fresh nested directory works like one prepared by an enclosing pipeline."""
    output = tmp_path / "nested" / "classifier"
    if directory == "current":
        monkeypatch.chdir(tmp_path)
        output = tmp_path
    if directory == "existing":
        output.mkdir(parents=True)
        (output / "keep.txt").write_text("unrelated artifact", encoding="utf-8")
    features = np.tile([[-2., -1.], [2., 1.]], (20, 1)).astype(np.float32)
    labels = np.tile([0, 1], 20)
    model = classifier(input_dim=2, hidden_dims=[], output_dim=2, epochs_mlp=2,
                       batch_size_mlp=8, lr_mlp=.1, device="cpu",
                       save_path="" if directory == "current" else str(output), random_state=5)

    assert model.fit(features, labels) is model

    checkpoint = torch.load(output / "best_mlp_classifier.pth", weights_only=True)
    assert checkpoint.keys() == model.model.state_dict().keys()
    for name, value in checkpoint.items():
        assert value.shape == model.model.state_dict()[name].shape
        assert torch.isfinite(value).all()
    assert model.predict(features).shape == labels.shape
    assert model.score(features, labels) >= .5
    with Image.open(output / "mlp_training_curves_epochs_2.png") as curve:
        assert curve.format == "PNG"
        curve.verify()
    if directory == "existing":
        assert (output / "keep.txt").read_text(encoding="utf-8") == "unrelated artifact"
