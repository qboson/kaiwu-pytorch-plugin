"""Check the paired MAE reported by both molecular fingerprint baselines."""

import importlib.util
from pathlib import Path
import warnings

import numpy as np
import pytest
import torch

pytest.importorskip("pandas")


@pytest.fixture(params=["sqa", "cim_sa"])
def baseline(request, monkeypatch):
    root = Path(__file__).resolve().parents[1] / "example" / "molecular_design" / request.param
    monkeypatch.syspath_prepend(str(root))
    spec = importlib.util.spec_from_file_location(
        f"{request.param}_fingerprint_baseline",
        root / "qc_mol_design" / "baselines" / "fingerprint_models.py",
    )
    module = importlib.util.module_from_spec(spec)
    with warnings.catch_warnings():
        spec.loader.exec_module(module)
    return module


def test_mae_aligns_single_column_targets_instead_of_broadcasting(baseline, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    model = torch.nn.Sequential(torch.nn.Linear(1, 1), torch.nn.Flatten(0))
    with torch.no_grad():
        model[0].weight.fill_(1)
        model[0].bias.zero_()
    features = np.array([[0.], [1.], [3.]])
    labels = np.array([[0.], [2.], [3.]])

    assert baseline.evaluate_fingerprint_predictor(model, features, labels) == pytest.approx(1 / 3)


def test_evaluation_uses_live_model_dtype_and_weights_the_tail_batch(baseline):
    class Predictor(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0, dtype=torch.float64))
            self.batch_sizes = []

        def forward(self, values):
            self.batch_sizes.append(len(values))
            assert values.dtype == self.weight.dtype
            return values[:, 0] * self.weight

    model = Predictor()
    features = np.arange(5, dtype=np.float32).reshape(-1, 1)
    targets = np.array([0., 1., 2., 3., 9.])

    assert baseline.evaluate_fingerprint_predictor(model, features, targets, batch_size=2) == pytest.approx(1)
    assert model.batch_sizes == [2, 2, 1]
