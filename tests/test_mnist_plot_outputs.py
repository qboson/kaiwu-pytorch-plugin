"""Actual MNIST PNG artifacts and classifier completion in fresh directories."""

from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import pytest
import torch


@pytest.fixture
def actual_example(monkeypatch):
    folder = Path(__file__).resolve().parents[1] / "example/qvae_mnist"
    monkeypatch.syspath_prepend(str(folder))
    roots = {"model", "trainer", "utils", "downstream"}
    saved_modules = {name: module for name, module in sys.modules.items()
                     if name.split(".")[0] in roots}
    for name in saved_modules:
        del sys.modules[name]
    try:
        import utils.helpers as helpers
        import downstream.classifier as classifier
        import kaiwu.torch_plugin as plugin

        assert Path(helpers.__file__).resolve() == folder / "utils/helpers.py"
        assert Path(classifier.__file__).resolve() == folder / "downstream/classifier.py"
        assert Path(plugin.__file__).resolve().is_relative_to(folder.parents[1] / "src")
        assert classifier.plot_training_curves is helpers.plot_training_curves
        monkeypatch.setattr(plt, "show", lambda: None)
        yield helpers, classifier
    finally:
        for name in list(sys.modules):
            if name.split(".")[0] in roots:
                del sys.modules[name]
        sys.modules.update(saved_modules)
        plt.close("all")


def verify_png(path):
    with Image.open(path) as picture:
        assert picture.format == "PNG"
        assert picture.width > 0 and picture.height > 0
        picture.verify()


def prepare_destination(tmp_path, kind):
    """Keep a real unrelated artifact in an existing ancestor directory."""
    ancestor = tmp_path / "existing"
    ancestor.mkdir()
    sentinel = ancestor / "keep.txt"
    sentinel.write_bytes(b"existing artifact remains unchanged")
    if kind == "default":
        destination = None
    elif kind == "bare":
        destination = "artifact.png"
    else:
        destination = ancestor / "new/plots/artifact.png"
        if kind == "nested-string":
            destination = str(destination)
    return destination, sentinel


@pytest.mark.parametrize("plotter", ["curves", "reconstruction", "generated"])
@pytest.mark.parametrize("destination_kind", ["default", "nested-string", "nested-path", "bare"])
def test_public_png_plotters_save_valid_artifacts_in_fresh_directories(
        actual_example, monkeypatch, tmp_path, plotter, destination_kind):
    helpers, _ = actual_example
    monkeypatch.chdir(tmp_path)
    destination, sentinel = prepare_destination(tmp_path, destination_kind)
    pixels = np.linspace(0., 1., 5 * 784).reshape(5, 784)
    original_pixels = pixels.copy()

    if plotter == "curves":
        helpers.plot_training_curves([1., .5], [.9, .6], [50., 75.], [60., 80.],
                                     save_path=destination, show=False)
        assert not plt.get_fignums()
    else:
        arguments = {} if destination is None else {"output": destination}
        if plotter == "reconstruction":
            helpers.plot_MNIST_output(pixels, pixels, **arguments)
        else:
            helpers.plot_generative_output(pixels, n_samples=5, **arguments)

    if destination is None:
        if plotter == "curves":
            artifacts = list((tmp_path / "results").glob("mlp_training_curves_*.png"))
            assert len(artifacts) == 1
            filename = artifacts[0]
        else:
            filename = tmp_path / "output/testVAE.png"
    else:
        filename = Path(destination)
    verify_png(filename)
    np.testing.assert_array_equal(pixels, original_pixels)
    assert sentinel.read_bytes() == b"existing artifact remains unchanged"


@pytest.mark.parametrize("destination_kind", ["nested-string", "nested-path", "bare"])
def test_flattened_grid_accepts_nested_and_bare_file_destinations(
        actual_example, monkeypatch, tmp_path, destination_kind):
    helpers, _ = actual_example
    monkeypatch.chdir(tmp_path)
    destination, sentinel = prepare_destination(tmp_path, destination_kind)
    features = torch.linspace(0., 1., 4 * 784).reshape(4, 784)
    original_features = features.clone()

    helpers.plot_flattened_images_grid(features, grid_size=2, save_path=destination)

    verify_png(destination)
    torch.testing.assert_close(features, original_features, rtol=0, atol=0)
    assert not plt.get_fignums()
    assert sentinel.read_bytes() == b"existing artifact remains unchanged"


def test_grid_without_destination_keeps_display_only_behavior(actual_example, monkeypatch, tmp_path):
    helpers, _ = actual_example
    monkeypatch.chdir(tmp_path)
    helpers.plot_flattened_images_grid(torch.zeros(4, 784), grid_size=2)
    assert not list(tmp_path.iterdir())
    assert not plt.get_fignums()


class FeatureProtocol(torch.nn.Module):
    """Supply unit feature arrays through the public QVAE visualization protocol."""

    def __init__(self):
        super().__init__()
        self.offset = torch.nn.Parameter(torch.tensor(0.))

    def forward(self, features):
        return None, None, None, features + self.offset


@pytest.mark.parametrize("destination_kind", ["default", "nested-string", "nested-path", "bare"])
def test_actual_tsne_saves_png_and_preserves_returned_feature_data(
        actual_example, monkeypatch, tmp_path, destination_kind):
    helpers, _ = actual_example
    monkeypatch.chdir(tmp_path)
    destination, sentinel = prepare_destination(tmp_path, destination_kind)
    features = torch.tensor(np.random.default_rng(42).normal(size=(40, 3)), dtype=torch.float32)
    labels = torch.arange(40) % 2
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(features, labels),
                                       batch_size=10)

    data, filename, status = helpers.t_SNE(loader, FeatureProtocol(), epochs=2,
                                         save_path=destination, show=False)

    assert status == "epochs_2"
    if destination is None:
        assert Path(filename).parent == Path("results")
        assert Path(filename).name.startswith("t-SNE_QVAE_epochs_2_")
    else:
        assert filename == destination
    np.testing.assert_array_equal(data[["dim_0", "dim_1", "dim_2"]].values, features.numpy())
    assert data["label"].tolist() == [str(value) for value in labels.tolist()]
    assert np.isfinite(data[["x-tsne", "y-tsne"]].values).all()
    verify_png(filename)
    assert not plt.get_fignums()
    assert sentinel.read_bytes() == b"existing artifact remains unchanged"


def test_actual_default_classifier_fit_finishes_and_saves_training_curve(
        actual_example, monkeypatch, tmp_path):
    _, classifier = actual_example
    monkeypatch.chdir(tmp_path)

    class CorrectInitialMLP(classifier.MLPClassifier):
        """Initialize a valid single linear layer with 100% validation accuracy."""

        def _create_model(self):
            model = super()._create_model()
            with torch.no_grad():
                model[0].weight.copy_(torch.tensor([[-1.], [1.]]))
                model[0].bias.zero_()
            return model

    features = np.repeat(np.array([[-1.], [1.]], dtype=np.float32), 10, axis=0)
    labels = np.repeat(np.array([0, 1]), 10)
    model = CorrectInitialMLP(input_dim=1, hidden_dims=[], output_dim=2, lr_mlp=0.,
                              weight_decay=0., batch_size_mlp=20, epochs_mlp=1, device="cpu")

    fitted = model.fit(features, labels)

    assert fitted is model
    assert model.save_path is None
    np.testing.assert_array_equal(model.predict(features), labels)
    assert model.score(features, labels) == 1.
    artifacts = list((tmp_path / "results").glob("mlp_training_curves_*.png"))
    assert len(artifacts) == 1
    verify_png(artifacts[0])
