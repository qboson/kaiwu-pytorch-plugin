"""Exercise the public image helper through its normal example import."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


WORKFLOW = r'''
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import traceback

optional = {"pandas", "matplotlib", "gif", "imageio", "PIL", "torchvision", "torchmetrics"}
try:
    from utils.helpers import plot_image
except ModuleNotFoundError as exc:
    if exc.name in optional and importlib.util.find_spec(exc.name) is None:
        print("OPTIONAL_DEPENDENCY_MISSING:" + exc.name)
        raise SystemExit(77)
    raise

import matplotlib.pyplot as plt
from matplotlib.image import AxesImage
import numpy as np
from PIL import Image

output = Path(sys.argv[1])
case = sys.argv[2]
if case == "layer0_explicit":
    values = np.arange(36, dtype=np.float32).reshape(3, 12)
    layer, vmin, vmax = 0, 5.0, 30.0
    expected_x, expected_y, expected_limits = [0, 10], [0, 1, 2], [5.0, 30.0]
else:
    assert case == "layer1_automatic"
    values = np.array([[2, 5, 1], [9, 4, 3], [8, 7, 6],
                       [-2, 0, 11], [10, -1, 12]], dtype=np.float64)
    layer, vmin, vmax = 1, None, None
    expected_x, expected_y, expected_limits = [0, 1, 2], [0, 1, 2, 3, 4], [-2.0, 12.0]
original = values.copy()
original_shape, original_dtype = values.shape, values.dtype
before = set(plt.get_fignums())
module = sys.modules[plot_image.__module__]
module_path = Path(module.__file__).resolve()
item = {"case": case, "module_path": str(module_path),
        "module_sha256": hashlib.sha256(module_path.read_bytes()).hexdigest(),
        "loaded_sdk_modules": sorted(n for n in sys.modules if n == "kaiwu" or n.startswith("kaiwu.")),
        "input_shape": list(values.shape), "input_dtype": str(values.dtype)}
exit_code = 0
try:
    artist = plot_image(values, layer, vmin=vmin, vmax=vmax)
    assert isinstance(artist, AxesImage)
    axes, figure = artist.axes, artist.figure
    assert axes.figure is figure and artist in axes.images
    np.testing.assert_array_equal(artist.get_array(), original)
    assert artist.get_array().shape == original_shape
    assert artist.colorbar is not None
    assert artist.colorbar.mappable is artist
    assert artist.colorbar.ax.figure is figure
    assert artist.colorbar.ax is not axes
    assert artist.colorbar.ax.get_ylabel() == "Energy (MeV)"
    assert axes.get_xlabel() == r'$\eta$ Cell ID'
    assert axes.get_ylabel() == r'$\phi$ Cell ID'
    np.testing.assert_array_equal(axes.get_xticks(), expected_x)
    np.testing.assert_array_equal(axes.get_yticks(), expected_y)
    np.testing.assert_array_equal(artist.get_clim(), expected_limits)
    np.testing.assert_array_equal(figure.get_size_inches(), [20, 20])
    assert figure.number not in before and figure.number in plt.get_fignums()
    image_path = output / (case + ".png")
    figure.savefig(image_path)
    with Image.open(image_path) as rendered:
        rendered.load()
        assert rendered.format == "PNG"
        assert rendered.width > 100 and rendered.height > 100
        extrema = rendered.convert("RGB").getextrema()
        assert any(low < high for low, high in extrema)
        png = {"filename": image_path.name, "mode": rendered.mode,
               "size": list(rendered.size), "rgb_extrema": extrema}
    item.update({"outcome": "returned", "artist_type": type(artist).__name__,
                 "colorbar_bound_to_artist": True, "displayed_data": artist.get_array().tolist(),
                 "x_ticks": axes.get_xticks().tolist(), "y_ticks": axes.get_yticks().tolist(),
                 "color_limits": list(artist.get_clim()), "png": png})
except Exception as exc:
    exit_code = 1
    item.update({"outcome": "exception", "exception": type(exc).__name__, "message": str(exc)})
    traceback.print_exc()
finally:
    item["input_unchanged"] = bool(values.shape == original_shape and values.dtype == original_dtype
                                   and np.array_equal(values, original))
    for number in set(plt.get_fignums()) - before:
        plt.close(number)
    item["only_owned_figures_closed"] = set(plt.get_fignums()) == before
    item["loaded_sdk_modules_after"] = sorted(n for n in sys.modules if n == "kaiwu" or n.startswith("kaiwu."))
    if not item["input_unchanged"] or not item["only_owned_figures_closed"]:
        exit_code = 1
    print("WORKFLOW_RESULT:" + json.dumps(item))
raise SystemExit(exit_code)
'''


class PlotImageRenderingTests(unittest.TestCase):
    def run_rendering_case(self, case):
        example = Path(__file__).resolve().parents[1] / "example" / "qvae_mnist"
        with tempfile.TemporaryDirectory(prefix="plot-image-") as directory:
            environment = os.environ.copy()
            environment["MPLBACKEND"] = "Agg"
            environment["MPLCONFIGDIR"] = str(Path(directory) / "mplconfig")
            process = subprocess.run(
                [sys.executable, "-E", "-B", "-c", WORKFLOW, directory, case],
                cwd=example, env=environment, capture_output=True, text=True, timeout=90,
            )
            self.workflow_exit_code = process.returncode
            self.workflow_stdout, self.workflow_stderr = process.stdout, process.stderr
            self.workflow_result = None
            for line in process.stdout.splitlines():
                if line.startswith("WORKFLOW_RESULT:"):
                    self.workflow_result = json.loads(line[len("WORKFLOW_RESULT:"):])
            self.png_artifacts = {p.name: p.read_bytes() for p in Path(directory).glob("*.png")}
            if process.returncode == 77 and process.stdout.startswith("OPTIONAL_DEPENDENCY_MISSING:"):
                self.skipTest(process.stdout.strip())
            self.assertEqual(process.returncode, 0, process.stderr)
            self.assertIsNotNone(self.workflow_result)
            self.assertEqual(self.workflow_result["outcome"], "returned")
            self.assertTrue(self.workflow_result["input_unchanged"])
            self.assertTrue(self.workflow_result["only_owned_figures_closed"])
            self.assertEqual(len(self.png_artifacts), 1)

    def test_layer_zero_rectangular_image_and_explicit_limits(self):
        self.run_rendering_case("layer0_explicit")

    def test_other_layer_rectangular_image_and_automatic_limits(self):
        self.run_rendering_case("layer1_automatic")


if __name__ == "__main__":
    unittest.main()
