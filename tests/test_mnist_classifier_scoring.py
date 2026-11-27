"""Regression test for scikit-learn scoring of the MNIST MLP example."""

import importlib.util
from pathlib import Path
import sys
from types import ModuleType

import numpy as np
import pytest
import torch
from torch import nn


def test_mlp_classifier_supports_probability_scoring(monkeypatch):
    """Use the configured classifier with scikit-learn's log-loss scorer."""
    metrics = pytest.importorskip("sklearn.metrics")
    pytest.importorskip("tqdm")
    helpers = ModuleType("utils.helpers")
    helpers.plot_training_curves = None  # Scoring does not use the plotting helper.
    monkeypatch.setitem(sys.modules, "utils.helpers", helpers)

    example_path = (
        Path(__file__).resolve().parents[1]
        / "example"
        / "qvae_mnist"
        / "downstream"
        / "classifier.py"
    )
    spec = importlib.util.spec_from_file_location("mnist_mlp_example", example_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    with torch.random.fork_rng():
        classifier = module.MLPClassifier(input_dim=1, output_dim=2, device="cpu")
        classifier.model = nn.Linear(1, 2, bias=False)
    with torch.no_grad():
        classifier.model.weight.copy_(torch.tensor([[0.0], [1.0]]))
    classifier.classes_ = np.array([0, 1])

    features = np.array([[-1.0], [1.0]], dtype=np.float32)
    score = metrics.get_scorer("neg_log_loss")(classifier, features, classifier.classes_)

    assert score == pytest.approx(-np.logaddexp(0.0, -1.0))
