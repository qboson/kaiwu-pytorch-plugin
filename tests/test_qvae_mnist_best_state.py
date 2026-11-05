"""The MLP classifier must reload the weights of the best validation epoch."""

import importlib
from pathlib import Path
import sys

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def classifier_module(monkeypatch):
    saved = {
        name: value
        for name, value in sys.modules.items()
        if name == "classifier" or name == "utils" or name.startswith("utils.")
    }
    for name in saved:
        sys.modules.pop(name)
    monkeypatch.syspath_prepend(str(ROOT / "example" / "qvae_mnist" / "downstream"))
    monkeypatch.syspath_prepend(str(ROOT / "example" / "qvae_mnist"))
    try:
        yield importlib.import_module("classifier")
    finally:
        for name in list(sys.modules):
            if name == "classifier" or name == "utils" or name.startswith("utils."):
                sys.modules.pop(name)
        sys.modules.update(saved)


def test_best_state_is_a_snapshot_not_a_live_reference(classifier_module, monkeypatch):
    monkeypatch.setattr(classifier_module, "plot_training_curves", lambda **kwargs: None)
    classifier = classifier_module.MLPClassifier(
        input_dim=2,
        hidden_dims=[2],
        output_dim=2,
        epochs_mlp=2,
        batch_size_mlp=2,
        save_path=None,
    )
    epoch = {"value": 0}
    accuracies = iter([0.9, 0.1])

    def fake_train(model, data_loader, optimizer, criterion, device):
        del data_loader, optimizer, criterion, device
        epoch["value"] += 1
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.fill_(float(epoch["value"]))
        return 50.0, 0.0

    def fake_eval(model, data_loader, criterion, device):
        del model, data_loader, criterion, device
        return next(accuracies), 0.0

    monkeypatch.setattr(classifier, "_train_mlp_epoch", fake_train)
    monkeypatch.setattr(classifier, "_eval_mlp_epoch", fake_eval)

    classifier.fit(
        np.zeros((10, 2), dtype=np.float32),
        np.array([0, 1] * 5),
    )

    values = {float(parameter.detach().flatten()[0]) for parameter in classifier.model.parameters()}
    assert values == {1.0}
