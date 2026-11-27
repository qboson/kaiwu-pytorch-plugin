"""The MNIST pipeline reports class-aware metrics from one prediction pass."""

import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch


def test_pipeline_report_contains_aligned_counts_and_macro_f1(monkeypatch, tmp_path):
    pytest.importorskip("sklearn", exc_type=ImportError)
    calls = []

    class Pipeline:
        def fit(self, samples, labels):
            np.testing.assert_array_equal(labels, [0, 1])

        def predict(self, samples):
            calls.append(samples.copy())
            return np.array([0, 0, 0, 7])

        def score(self, samples, labels):
            return 0.75

    source = Path(__file__).resolve().parents[1] / "example" / "qvae_mnist"
    model = ModuleType("model")
    model.Config = lambda **kwargs: SimpleNamespace(**kwargs)
    downstream = ModuleType("downstream")
    downstream.get_full_pipeline = lambda _config: Pipeline()
    mnist = ModuleType("utils.loadMNIST")
    mnist.loadMNIST = lambda **kwargs: (
        [(torch.zeros(2, 1), torch.tensor([0, 1]))],
        [(torch.ones(4, 1), torch.tensor([0, 0, 1, 7]))],
    )
    for name, module in (("model", model), ("downstream", downstream), ("utils.loadMNIST", mnist)):
        monkeypatch.setitem(sys.modules, name, module)

    spec = importlib.util.spec_from_file_location("mnist_pipeline_report", source / "run_pipeline.py")
    pipeline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pipeline)
    tmp_path = tmp_path / "run"
    tmp_path.mkdir()
    monkeypatch.setattr(sys, "argv", ["run_pipeline.py", "--output-dir", str(tmp_path)])
    pipeline.main()

    result = json.loads((tmp_path / "results.json").read_text(encoding="utf-8"))
    assert result == {
        "test_accuracy": 0.75,
        "test_macro_f1": pytest.approx(0.6),
        "class_labels": [0, 1, 7],
        "confusion_matrix": [[2, 0, 0], [1, 0, 0], [0, 0, 1]],
    }
    assert len(calls) == 1
