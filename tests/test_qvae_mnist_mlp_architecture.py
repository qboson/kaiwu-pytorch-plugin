"""Contract test for ``MLPClassifier._create_model``."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import torch
from torch import nn

REPO = Path(__file__).resolve().parents[1]
CLASSIFIER = REPO / "example" / "qvae_mnist" / "downstream" / "classifier.py"


def _load_classifier():
    names = ("utils", "utils.helpers")
    saved = {name: sys.modules.get(name) for name in names}
    utils_module = types.ModuleType("utils")
    utils_module.__path__ = []
    helpers_module = types.ModuleType("utils.helpers")
    helpers_module.plot_training_curves = lambda *args, **kwargs: None
    sys.modules["utils"] = utils_module
    sys.modules["utils.helpers"] = helpers_module
    try:
        spec = importlib.util.spec_from_file_location(
            "qvae_mnist_classifier_uut", CLASSIFIER
        )
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        for name, value in saved.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value


def test_builds_the_configured_mlp_stack():
    classifier = _load_classifier()
    model = classifier.MLPClassifier(
        input_dim=5,
        hidden_dims=[4, 3],
        output_dim=2,
        device=torch.device("cpu"),
    )

    network = model._create_model()

    assert isinstance(network, nn.Sequential)
    assert [type(layer) for layer in network] == [
        nn.Linear,
        nn.ReLU,
        nn.Dropout,
        nn.Linear,
        nn.ReLU,
        nn.Dropout,
        nn.Linear,
    ]
    linears = [layer for layer in network if isinstance(layer, nn.Linear)]
    assert [(layer.in_features, layer.out_features) for layer in linears] == [
        (5, 4),
        (4, 3),
        (3, 2),
    ]
    assert [layer.p for layer in network if isinstance(layer, nn.Dropout)] == [0.2, 0.2]
    assert network(torch.zeros(7, 5)).shape == (7, 2)
    assert next(network.parameters()).device.type == "cpu"
