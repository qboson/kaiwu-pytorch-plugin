"""Exercise the documented classifier entry with real offline CPU features."""
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

# These extras are documented for the QVAE example. Broken installations fail.
optional = {"pandas", "matplotlib", "gif", "imageio", "PIL", "torchvision", "torchmetrics"}
try:
    from downstream import MLPClassifier
except ModuleNotFoundError as exc:
    if exc.name in optional and importlib.util.find_spec(exc.name) is None:
        print("OPTIONAL_DEPENDENCY_MISSING:" + exc.name)
        raise SystemExit(77)
    print(json.dumps({"stage": "normal_import", "exception": type(exc).__name__, "name": exc.name, "message": str(exc)}), file=sys.stderr)
    raise

import downstream
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageStat
import torch
from utils.helpers import plot_training_curves

stages = []
def assert_offline(stage):
    loaded = sorted(name for name in sys.modules if name == "kaiwu" or name.startswith("kaiwu.") or name == "downstream.pipeline")
    assert not loaded, (stage, loaded)
    stages.append(stage)

assert_offline("import")
assert downstream.__all__ == ["MLPClassifier", "get_full_pipeline"]
try:
    getattr(downstream, "not_an_export")
except AttributeError:
    pass
else:
    raise AssertionError("Unknown package attributes must raise AttributeError")

output = Path(sys.argv[1])
X = np.array([[-1.0, -0.5]] * 12 + [[1.0, 0.5]] * 12, dtype=np.float32)
y = np.array([0] * 12 + [1] * 12, dtype=np.int64)
model = MLPClassifier(input_dim=2, hidden_dims=[4], output_dim=2,
                      epochs_mlp=2, batch_size_mlp=8, lr_mlp=0.01,
                      device="cpu", random_state=7, save_path=str(output))
assert model.fit(X, y, validation_split=0.25) is model
assert_offline("fit")
assert all(parameter.device.type == "cpu" for parameter in model.model.parameters())
labels = model.predict(X)
assert_offline("predict")
probabilities = model.predict_proba(X)
assert_offline("predict_proba")
assert labels.shape == (24,)
assert set(labels.tolist()).issubset({0, 1})
assert probabilities.shape == (24, 2)
assert np.isfinite(probabilities).all()
assert ((probabilities >= 0) & (probabilities <= 1)).all()
np.testing.assert_allclose(probabilities.sum(axis=1), 1, atol=1e-6, rtol=0)

expected_labels = ["Training Loss", "Validation Loss", "Training Accuracy", "Validation Accuracy"]
def inspect_curves(figure):
    assert len(figure.axes) == 2
    lines = [line for axis in figure.axes for line in axis.lines]
    assert [line.get_label() for line in lines] == expected_labels
    values = [np.asarray(line.get_ydata()) for line in lines]
    assert all(value.shape == (2,) and np.isfinite(value).all() for value in values)
    return [value.tolist() for value in values]

fit_curves = inspect_curves(plt.gcf())
plt.close("all")
histories = [[1.0, 0.5], [1.2, 0.7], [40, 60], [30, 55]]
plot_training_curves(*histories, save_path=str(output / "direct_curves.png"), show=True)
assert_offline("direct_plot")
direct_curves = inspect_curves(plt.gcf())
for actual, expected in zip(direct_curves, histories):
    np.testing.assert_array_equal(actual, expected)
plt.close("all")

images = []
for name in ("mlp_training_curves_epochs_2.png", "direct_curves.png"):
    path = output / name
    with Image.open(path) as image:
        assert image.format == "PNG"
        image.load()
        assert min(image.size) >= 200
        rgb = image.convert("RGB")
        extrema = rgb.getextrema()
        assert all(high - low > 32 for low, high in extrema)
        assert max(ImageStat.Stat(rgb).var) > 1
        images.append({"name": name, "size": list(image.size), "extrema": extrema,
                       "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})

imports = {}
for name in ("downstream", "downstream.classifier", "utils.helpers"):
    path = Path(sys.modules[name].__file__).resolve()
    imports[name] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
print(json.dumps({"stages_without_sdk_or_pipeline": stages, "imports": imports,
                  "torch_version": torch.__version__, "probability_shape": list(probabilities.shape),
                  "fit_curves": fit_curves, "direct_curves": direct_curves, "images": images}))
'''


class OfflineClassifierTests(unittest.TestCase):
    def test_real_cpu_classifier_and_plots(self):
        example = Path(__file__).resolve().parents[1] / "example/qvae_mnist"
        with tempfile.TemporaryDirectory() as directory:
            env = os.environ.copy()
            env["MPLBACKEND"] = "Agg"
            env["MPLCONFIGDIR"] = str(Path(directory) / "mplconfig")
            result = subprocess.run(
                [sys.executable, "-E", "-B", "-c", WORKFLOW, directory],
                cwd=example, env=env, text=True, capture_output=True, timeout=90,
            )
            self.workflow_stdout = result.stdout
            self.workflow_stderr = result.stderr
            self.workflow_exit_code = result.returncode
            if result.returncode == 77 and result.stdout.startswith("OPTIONAL_DEPENDENCY_MISSING:"):
                self.skipTest(result.stdout.strip())
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.workflow_result = json.loads(result.stdout.splitlines()[-1])
            self.png_artifacts = {
                item["name"]: (Path(directory) / item["name"]).read_bytes()
                for item in self.workflow_result["images"]
            }


if __name__ == "__main__":
    unittest.main()
