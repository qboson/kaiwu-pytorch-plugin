"""Exercise reconstruction layouts and PDF placement with real CPU rendering."""
import importlib.util
import math
import os
from pathlib import Path
import tempfile
import unittest

for dependency in ("matplotlib", "seaborn"):
    if importlib.util.find_spec(dependency) is None:
        raise unittest.SkipTest(f"DBN plotting tests require {dependency}")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from example.dbn_digits.supervised_dbn_digits import RBMVisualizer


class TorchReconstructionFixture:
    """Deterministic Torch reconstruction for plotting tests."""

    def __init__(self, n_features=4):
        self.calls = 0
        self.weight = torch.tensor(
            [[0.4, 0.1, -0.2, 0.3], [0.2, -0.3, 0.5, 0.1]],
            dtype=torch.float32,
        )
        self.weight = self.weight.repeat(1, (n_features + 3) // 4)[:, :n_features]

    def reconstruct(self, X, layer_index):
        if layer_index != 0:
            raise ValueError("This fixture has one reconstruction layer")
        self.calls += 1
        with torch.no_grad():
            source = torch.as_tensor(X, dtype=torch.float32)
            hidden = torch.sigmoid(source @ self.weight.T)
            reconstruction = torch.sigmoid(hidden @ self.weight)
            errors = torch.mean((source - reconstruction) ** 2, dim=1)
        return reconstruction.numpy(), errors.numpy()


class TestReconstructionPDFDirectory(unittest.TestCase):
    def setUp(self):
        self.previous_cwd = Path.cwd()
        self.tempdir = tempfile.TemporaryDirectory()
        self.workdir = Path(self.tempdir.name)
        os.chdir(self.workdir)
        self.X = np.array([[0.0, 0.2, 0.8, 1.0], [1.0, 0.7, 0.3, 0.0]], dtype=np.float32)
        self.y = np.array([2, 5])
        self.fixture = TorchReconstructionFixture()

    def tearDown(self):
        plt.close("all")
        os.chdir(self.previous_cwd)
        self.tempdir.cleanup()

    def _plot(self, output, save_pdf=True):
        visualizer = RBMVisualizer(result_dir=output)
        self.assertTrue(output.is_dir())
        try:
            errors = visualizer.plot_reconstructed_images(
                self.fixture, self.X, self.y, layer_index=0, n_images=2,
                title_suffix="cpu output", save_pdf=save_pdf, img_shape=(2, 2),
            )
        except OSError as error:
            self.fail(f"Reconstruction output must use the configured directory: {error}")
        self.assertEqual(self.fixture.calls, 1)
        self.assertEqual(errors.shape, (2,))
        self.assertTrue(np.isfinite(errors).all())
        return output / "reconstructed_images_cpu output.pdf"

    def _assert_pdf(self, path):
        self.assertTrue(path.is_file(), f"Expected reconstruction PDF at {path}")
        data = path.read_bytes()
        self.assertGreater(len(data), 1000)
        self.assertTrue(data.startswith(b"%PDF-"))
        self.assertTrue(data.rstrip().endswith(b"%%EOF"))

    def test_creates_missing_custom_directory(self):
        output = Path("custom figures") / "nested"
        self.assertFalse(output.exists())
        self._assert_pdf(self._plot(output))
        self.assertFalse(Path("results").exists())
        self.assertEqual(list(output.iterdir()), [output / "reconstructed_images_cpu output.pdf"])

    def test_existing_custom_directory_does_not_write_default_directory(self):
        output = self.workdir / "existing figures"
        output.mkdir()
        sentinel = output / "keep.txt"
        sentinel.write_text("preserve")
        Path("results").mkdir()
        self._assert_pdf(self._plot(output))
        self.assertEqual(sentinel.read_text(), "preserve")
        self.assertEqual(list(Path("results").iterdir()), [])
        self.assertEqual(set(p.name for p in output.iterdir()), {"keep.txt", "reconstructed_images_cpu output.pdf"})

    def test_save_false_does_not_create_pdf(self):
        output = Path("unsaved")
        path = self._plot(output, save_pdf=False)
        self.assertFalse(path.exists())
        self.assertEqual(list(output.iterdir()), [])
        self.assertFalse(Path("results").exists())


class TestReconstructionImageShape(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.workdir = Path(self.tempdir.name)

    def tearDown(self):
        plt.close("all")
        self.tempdir.cleanup()

    def _check_layout(self, width, shape=None):
        source = np.array(
            [[(row * width + col) / (3 * width) for col in range(width)] for row in range(3)],
            dtype=np.float32,
        )
        fixture = TorchReconstructionFixture(width)
        suffix = f"auto{width}" if shape is None else f"explicit{width}"
        output = self.workdir / suffix
        visualizer = RBMVisualizer(result_dir=output)
        errors = visualizer.plot_reconstructed_images(
            fixture, source, np.array([2, 5, 7]), n_images=2,
            title_suffix=suffix, save_pdf=True, img_shape=shape,
        )
        self.assertEqual(fixture.calls, 1)

        # Scalar arithmetic supplies an independent oracle for the fixture output.
        patterns = ((0.4, 0.1, -0.2, 0.3), (0.2, -0.3, 0.5, 0.1))
        expected = []
        for row in source[:2].tolist():
            hidden = [1.0 / (1.0 + math.exp(-sum(value * p[col % 4]
                      for col, value in enumerate(row)))) for p in patterns]
            expected.append([1.0 / (1.0 + math.exp(-sum(hidden[k] * patterns[k][col % 4]
                             for k in range(2)))) for col in range(width)])
        expected_errors = [sum((value - recon) ** 2 for value, recon in zip(row, reconstruction)) / width
                           for row, reconstruction in zip(source[:2].tolist(), expected)]
        self.assertEqual(errors.shape, (2,))
        np.testing.assert_allclose(errors, expected_errors, rtol=1e-5, atol=1e-7)

        figure = plt.gcf()
        self.assertEqual(len(figure.axes), 4)
        for index in range(2):
            original_axis, reconstructed_axis = figure.axes[index], figure.axes[index + 2]
            self.assertEqual(len(original_axis.images), 1)
            self.assertEqual(len(reconstructed_axis.images), 1)
            original = np.asarray(original_axis.images[0].get_array())
            reconstructed = np.asarray(reconstructed_axis.images[0].get_array())
            self.assertEqual(original.size, width)
            self.assertEqual(original.shape, reconstructed.shape)
            if shape is not None:
                self.assertEqual(original.shape, shape)
            elif width == 16:
                self.assertEqual(original.shape, (4, 4))
            np.testing.assert_array_equal(original.ravel(), source[index])
            np.testing.assert_allclose(reconstructed.ravel(), expected[index], rtol=1e-5, atol=1e-7)
            self.assertEqual(original_axis.get_title(), f"Label: {[2, 5][index]}")
        figure.canvas.draw()
        pdf = output / f"reconstructed_images_{suffix}.pdf"
        data = pdf.read_bytes()
        self.assertTrue(data.startswith(b"%PDF-"))
        self.assertTrue(data.rstrip().endswith(b"%%EOF"))

    def test_auto_layout_preserves_all_pixels(self):
        for width in (12, 16, 128):
            with self.subTest(width=width):
                self._check_layout(width)

    def test_explicit_rectangular_shape_takes_precedence(self):
        self._check_layout(12, (2, 6))


if __name__ == "__main__":
    unittest.main()
