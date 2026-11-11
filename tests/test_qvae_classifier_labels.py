"""Check real CPU fits with original labels and an explicit output width."""
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

optional = {"pandas", "matplotlib", "gif", "imageio", "PIL", "torchvision", "torchmetrics"}
try:
    from downstream import MLPClassifier
except ModuleNotFoundError as exc:
    if exc.name in optional and importlib.util.find_spec(exc.name) is None:
        print("OPTIONAL_DEPENDENCY_MISSING:" + exc.name)
        raise SystemExit(77)
    raise

import matplotlib.pyplot as plt
import numpy as np
import torch

stages = []
def assert_offline(stage):
    loaded = sorted(name for name in sys.modules if name == "kaiwu" or name.startswith("kaiwu.") or name == "downstream.pipeline")
    assert not loaded, (stage, loaded)
    stages.append(stage)

assert_offline("import")
output = Path(sys.argv[1])
mode = sys.argv[2]
X = np.array([[-1.0, -0.5]] * 12 + [[1.0, 0.5]] * 12, dtype=np.float32)
original_X = X.copy()
cases = [("contiguous", [0, 1]), ("integers", [-3, 7]), ("strings", ["alpha", "zebra"])]
if mode == "width":
    assert MLPClassifier(device="cpu").output_dim == 10
    cases = [("width_1", [0, 1]), ("width_3", [0, 1])]
observations = []
for name, values in cases:
    y = np.array([values[0]] * 12 + [values[1]] * 12)
    original_y = y.copy()
    directory = output / name
    directory.mkdir()
    width = int(name[-1]) if mode == "width" else 2
    model = MLPClassifier(input_dim=2, hidden_dims=[4], output_dim=width,
                          epochs_mlp=2, batch_size_mlp=8, lr_mlp=0.01,
                          device="cpu", random_state=7, save_path=str(directory))
    item = {"case": name, "output_dim": width}
    try:
        assert model.fit(X, y, validation_split=0.25) is model
        assert_offline(name + ":fit")
        assert all(parameter.device.type == "cpu" for parameter in model.model.parameters())
        predictions = model.predict(X)
        assert_offline(name + ":predict")
        probabilities = model.predict_proba(X)
        assert_offline(name + ":predict_proba")
        item.update({"outcome": "returned", "classes": model.classes_.tolist(),
                     "predictions": predictions.tolist(), "prediction_dtype": str(predictions.dtype),
                     "label_dtype": str(y.dtype), "probabilities": probabilities.tolist()})
    except Exception as exc:
        item.update({"outcome": "exception", "exception": type(exc).__name__, "message": str(exc)})
    finally:
        np.testing.assert_array_equal(X, original_X)
        np.testing.assert_array_equal(y, original_y)
        assert_offline(name + ":finished")
        item["model_created"] = model.model is not None
        item["classes_assigned"] = model.classes_ is not None
        item["files"] = sorted(path.name for path in directory.iterdir())
        plt.close("all")
    observations.append(item)

imports = {}
for name in ("downstream", "downstream.classifier", "utils.helpers"):
    path = Path(sys.modules[name].__file__).resolve()
    imports[name] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
print(json.dumps({"observations": observations, "imports": imports,
                  "stages_without_sdk_or_pipeline": stages, "torch_version": torch.__version__}))
'''


class ClassifierLabelTests(unittest.TestCase):
    def run_workflow(self, mode):
        example = Path(__file__).resolve().parents[1] / "example/qvae_mnist"
        with tempfile.TemporaryDirectory() as directory:
            env = os.environ.copy()
            env["MPLBACKEND"] = "Agg"
            env["MPLCONFIGDIR"] = str(Path(directory) / "mplconfig")
            result = subprocess.run(
                [sys.executable, "-E", "-B", "-c", WORKFLOW, directory, mode],
                cwd=example, env=env, text=True, capture_output=True, timeout=90,
            )
            self.workflow_stdout = result.stdout
            self.workflow_stderr = result.stderr
            self.workflow_exit_code = result.returncode
            if result.returncode == 77 and result.stdout.startswith("OPTIONAL_DEPENDENCY_MISSING:"):
                self.skipTest(result.stdout.strip())
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.workflow_result = json.loads(result.stdout.splitlines()[-1])
            return self.workflow_result["observations"]

    def test_original_labels_and_probability_columns(self):
        import numpy as np

        observations = self.run_workflow("labels")
        reference = observations[0]
        self.assertEqual(reference["outcome"], "returned", reference)
        for item, classes in zip(observations, ([0, 1], [-3, 7], ["alpha", "zebra"])):
            with self.subTest(labels=classes):
                self.assertEqual(item["outcome"], "returned", item)
                self.assertEqual(item["classes"], classes)
                self.assertEqual(item["prediction_dtype"], item["label_dtype"])
                probabilities = np.asarray(item["probabilities"])
                self.assertEqual(probabilities.shape, (24, 2))
                self.assertTrue(np.isfinite(probabilities).all())
                self.assertTrue(((probabilities >= 0) & (probabilities <= 1)).all())
                np.testing.assert_allclose(probabilities.sum(axis=1), 1, rtol=0, atol=1e-6)
                np.testing.assert_array_equal(item["predictions"], np.asarray(classes)[probabilities.argmax(axis=1)])
                np.testing.assert_allclose(probabilities, reference["probabilities"], rtol=1e-6, atol=1e-7)

    def test_explicit_output_width_matches_observed_classes(self):
        for item in self.run_workflow("width"):
            with self.subTest(output_dim=item["output_dim"]):
                self.assertEqual(item["outcome"], "exception", item)
                self.assertEqual(item["exception"], "ValueError", item)
                self.assertIn("output_dim", item["message"])
                self.assertIn(str(item["output_dim"]), item["message"])
                self.assertIn("2", item["message"])
                self.assertIn("class", item["message"])
                self.assertFalse(item["model_created"])
                self.assertFalse(item["classes_assigned"])
                self.assertEqual(item["files"], [])


if __name__ == "__main__":
    unittest.main()
