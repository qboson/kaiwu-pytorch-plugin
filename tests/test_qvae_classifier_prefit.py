"""Exercise pristine prediction errors and a genuine fitted CPU classifier."""
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

# Extras documented for this example; broken installed packages still fail.
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
from sklearn.exceptions import NotFittedError
import torch

stages = []
def assert_offline(stage):
    loaded = sorted(name for name in sys.modules if name == "kaiwu" or name.startswith("kaiwu.") or name == "downstream.pipeline")
    assert not loaded, (stage, loaded)
    stages.append(stage)

assert_offline("import")
output = Path(sys.argv[1])
mode = sys.argv[2]
observations = []
if mode == "prefit":
    X = np.array([[-1.0, -0.5], [1.0, 0.5]], dtype=np.float32)
    original_X = X.copy()
    for method in ("predict", "predict_proba"):
        model = MLPClassifier(input_dim=2, hidden_dims=[4], output_dim=2, device="cpu")
        item = {"method": method}
        try:
            getattr(model, method)(X)
        except Exception as exc:
            item.update({"outcome": "exception", "exception": type(exc).__name__,
                         "message": str(exc), "is_not_fitted_error": isinstance(exc, NotFittedError)})
        else:
            item["outcome"] = "returned"
        np.testing.assert_array_equal(X, original_X)
        assert_offline(method + ":prefit")
        observations.append(item)
else:
    X = np.array([[-1.0, -0.5]] * 12 + [[1.0, 0.5]] * 12, dtype=np.float32)
    y = np.array(["alpha"] * 12 + ["zebra"] * 12)
    original_X, original_y = X.copy(), y.copy()
    model = MLPClassifier(input_dim=2, hidden_dims=[4], output_dim=2,
                          epochs_mlp=2, batch_size_mlp=8, lr_mlp=0.01,
                          device="cpu", random_state=7, save_path=str(output))
    assert model.fit(X, y, validation_split=0.25) is model
    assert_offline("fit")
    assert all(parameter.device.type == "cpu" for parameter in model.model.parameters())
    predictions = model.predict(X)
    assert_offline("predict:fitted")
    probabilities = model.predict_proba(X)
    assert_offline("predict_proba:fitted")
    np.testing.assert_array_equal(model.classes_, ["alpha", "zebra"])
    assert predictions.dtype == y.dtype
    assert predictions.shape == (24,)
    assert probabilities.shape == (24, 2)
    assert np.isfinite(probabilities).all()
    assert ((probabilities >= 0) & (probabilities <= 1)).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1, rtol=0, atol=1e-6)
    np.testing.assert_array_equal(predictions, model.classes_[probabilities.argmax(axis=1)])
    np.testing.assert_array_equal(X, original_X)
    np.testing.assert_array_equal(y, original_y)
    observations.append({"outcome": "returned", "classes": model.classes_.tolist(),
                         "predictions": predictions.tolist(), "probabilities": probabilities.tolist()})
    plt.close("all")

imports = {}
for name in ("downstream", "downstream.classifier", "utils.helpers"):
    path = Path(sys.modules[name].__file__).resolve()
    imports[name] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
print(json.dumps({"observations": observations, "imports": imports,
                  "stages_without_sdk_or_pipeline": stages, "torch_version": torch.__version__}))
'''


class ClassifierPrefitTests(unittest.TestCase):
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

    def test_pristine_prediction_methods_raise_not_fitted(self):
        for item in self.run_workflow("prefit"):
            with self.subTest(method=item["method"]):
                self.assertEqual(item["outcome"], "exception", item)
                self.assertEqual(item["exception"], "NotFittedError", item)
                self.assertTrue(item["is_not_fitted_error"], item)
                self.assertIn("fit", item["message"].lower())

    def test_fitted_predictions_and_probabilities(self):
        observations = self.run_workflow("fitted")
        self.assertEqual(len(observations), 1)
        self.assertEqual(observations[0]["outcome"], "returned")


if __name__ == "__main__":
    unittest.main()
