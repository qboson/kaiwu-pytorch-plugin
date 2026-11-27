"""The molecular Table 1 evaluator follows the loaded model placements."""

import importlib
from pathlib import Path
import sys

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

pytest.importorskip("pandas")


@pytest.fixture(params=["sqa", "cim_sa"])
def evaluator(request, monkeypatch):
    root = Path(__file__).resolve().parents[1] / "example" / "molecular_design"
    monkeypatch.syspath_prepend(str(root / request.param))
    package = "qc_mol_design"
    previous = {
        key: sys.modules.pop(key)
        for key in list(sys.modules)
        if key == package or key.startswith(f"{package}.")
    }
    try:
        yield importlib.import_module(f"{package}.evaluation.metrics").compute_prediction_mae
    finally:
        for key in list(sys.modules):
            if key == package or key.startswith(f"{package}."):
                sys.modules.pop(key)
        sys.modules.update(previous)


class Fingerprint(nn.Linear):
    def forward(self, atoms, adj, degree):
        assert degree.dtype == torch.long
        return super().forward(atoms + adj)


class ConditionalModel(nn.Linear):
    def encode_unknown(self, fingerprint):
        return self(fingerprint)


class PropertyHeads(nn.Linear):
    def forward(self, hidden):
        values = super().forward(hidden)
        return dict(zip(("qed", "logp", "sas"), values.unbind(dim=1)))


def test_table1_streams_sample_weighted_errors_across_mixed_model_dtypes(evaluator, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    fingerprint = Fingerprint(1, 1, bias=False, dtype=torch.float64)
    conditional = ConditionalModel(1, 1, bias=False, dtype=torch.float32)
    heads = PropertyHeads(1, 3, bias=False, dtype=torch.float64)
    nn.init.ones_(fingerprint.weight)
    nn.init.ones_(conditional.weight)
    nn.init.ones_(heads.weight)
    atoms = torch.arange(1, 6, dtype=torch.float32).unsqueeze(1)
    targets = atoms + torch.tensor([0.0, 1.0, -2.0])
    targets[-1] += torch.tensor([5.0, 5.0, -5.0])
    data = TensorDataset(
        atoms, torch.zeros_like(atoms), torch.zeros(5, dtype=torch.long),
        torch.zeros(5, 3), targets,
    )

    result = evaluator(fingerprint, conditional, heads, DataLoader(data, batch_size=3))

    assert result == {"qed": 1.0, "logp": 2.0, "sas": 3.0}
