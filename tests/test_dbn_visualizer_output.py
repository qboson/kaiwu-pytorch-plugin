"""Exercise reconstruction PDF placement with real CPU rendering."""
import importlib.util
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
    """Deterministic test computation; no SDK or trained-model claim."""

    def __init__(self):
        self.calls = 0
        self.weight = torch.tensor(
            [[0.4, 0.1, -0.2, 0.3], [0.2, -0.3, 0.5, 0.1]],
            dtype=torch.float32,
        )

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


if __name__ == "__main__":
    unittest.main()
