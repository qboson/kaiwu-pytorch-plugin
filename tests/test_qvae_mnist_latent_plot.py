"""Latent-space plot contract for the QVAE-MNIST helpers."""

import importlib.util
import os

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_spec = importlib.util.spec_from_file_location(
    "qvae_mnist_helpers_latent",
    os.path.join(REPO_ROOT, "example", "qvae_mnist", "utils", "helpers.py"),
)
helpers = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(helpers)


def build_data():
    """Twenty two-dimensional points with two classes."""
    zeta = np.arange(40, dtype=np.float32).reshape(20, 2)
    labels = [index % 2 for index in range(20)]
    return zeta, labels


def test_default_dimensions_write_a_pdf_next_to_the_output_prefix(tmp_path):
    """The default call saves ``<output>.pdf`` with the multi-panel layout."""
    target = tmp_path / "latent"
    helpers.plot_latent_space(*build_data(), output=str(target))
    pdf = tmp_path / "latent.pdf"
    assert pdf.is_file()
    assert pdf.stat().st_size > 0


def test_single_panel_mode_writes_the_same_artifact(tmp_path):
    """``dimensions=0`` uses the single-panel branch but keeps the file name."""
    target = tmp_path / "single"
    helpers.plot_latent_space(*build_data(), output=str(target), dimensions=0)
    assert (tmp_path / "single.pdf").is_file()
