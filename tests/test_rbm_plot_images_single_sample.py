# Copyright (C) 2026 Beijing QBoson Quantum Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Real CPU model and Agg rendering; solver response is an explicit boundary fixture."""
import importlib.util
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import pytest
import torch


@pytest.fixture
def runner_module():
    """Import the complete example and its real dependencies, without SDK shims."""
    path = (
        Path(__file__).resolve().parents[1] / "example" / "rbm_digits" / "rbm_digits.py"
    )
    spec = importlib.util.spec_from_file_location("rbm_digits_single_image_contract", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class LocalSpinResponse:
    """A legal fixed Ising response at solve(), not a real SDK/hardware sample.

    Leave the real RBM's matrix generation, sample decoding, tensor conversion
    and plotting intact. No Kaiwu class, renderer, axes or file writer is replaced.
    """

    def __init__(self, rows):
        self.rows = rows
        self.calls = 0

    def solve(self, ising_matrix):
        self.calls += 1
        assert ising_matrix.shape == (67, 67)  # 64 visible + 2 hidden + constant spin
        return np.ones((self.rows, 67), dtype=np.float64)


@pytest.mark.parametrize("sample_count", [1, 2])
def test_plot_images_exports_single_and_multiple_digits(
    runner_module, tmp_path, monkeypatch, sample_count
):
    monkeypatch.chdir(tmp_path)
    # Avoid constructing an SDK optimizer/licence session. This is a real
    # Runner instance configured with the same real RBM object type fit creates.
    runner = runner_module.RBMRunner.__new__(runner_module.RBMRunner)
    runner.device = torch.device("cpu")
    runner.rbm = runner_module.RestrictedBoltzmannMachine(
        64,
        2,
        quadratic_coef=torch.zeros((64, 2)),
        linear_bias=torch.zeros(66),
        device=torch.device("cpu"),
    )
    runner.sampler = LocalSpinResponse(sample_count)
    images = torch.linspace(0, 1, sample_count * 64).reshape(sample_count, 64)
    labels = np.arange(sample_count)
    existing_figures = set(plt.get_fignums())

    try:
        runner.plot_images(images, labels, save_as="digits", save_pdf=True)
        assert runner.sampler.calls == 1
        figure = plt.gcf()
        assert len(figure.axes) == 2 * sample_count
        for index in range(sample_count):
            np.testing.assert_array_equal(
                figure.axes[index].images[0].get_array(), images[index].numpy().reshape(8, 8)
            )
            np.testing.assert_array_equal(
                figure.axes[sample_count + index].images[0].get_array(), np.ones((8, 8))
            )
            assert figure.axes[index].get_title() == f"Label: {labels[index]}"

        pdf = tmp_path / "results" / "digits.pdf"
        assert pdf.read_bytes().startswith(b"%PDF-")
        assert pdf.stat().st_size > 100

        # Render the same real Figure to a Pillow-readable format as an
        # additional rendering check; this is not a PDF rasterization claim.
        png = tmp_path / "rendered.png"
        figure.savefig(png)
        with Image.open(png) as image:
            image.load()
            assert image.width > 0 and image.height > 0
    finally:
        for number in set(plt.get_fignums()) - existing_figures:
            plt.close(number)


def test_plot_images_retains_uninitialized_model_guard(runner_module):
    runner = runner_module.RBMRunner.__new__(runner_module.RBMRunner)
    runner.rbm = None
    with pytest.raises(ValueError, match="RBM model not trained yet"):
        runner.plot_images(torch.zeros((1, 64)), np.array([0]))
