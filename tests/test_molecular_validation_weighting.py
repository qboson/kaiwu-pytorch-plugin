"""Validation must weight molecules equally when the tail batch is smaller."""

import importlib
from pathlib import Path
import sys
import warnings

import pytest

import torch


@pytest.fixture(params=["cim_sa", "sqa"])
def modules(request, monkeypatch):
    """Import the real loader from each independent backend."""
    prefix = "qc_mol_design"
    saved = {k: v for k, v in sys.modules.items() if k == prefix or k.startswith(prefix + ".")}
    for name in saved:
        sys.modules.pop(name)
    root = Path(__file__).resolve().parents[1] / "example" / "molecular_design"
    monkeypatch.syspath_prepend(str(root / request.param))
    try:
        with warnings.catch_warnings():
            yield (importlib.import_module("qc_mol_design.training.cd_trainer"),
                   importlib.import_module("qc_mol_design.training.qc_trainer"))
    finally:
        for name in list(sys.modules):
            if name == prefix or name.startswith(prefix + "."):
                sys.modules.pop(name)
        sys.modules.update(saved)



class Fingerprint(torch.nn.Module):
    def forward(self, x, adj, deg):
        return x


class EnergyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.version = torch.nn.Parameter(torch.zeros(()))


class Predictor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.ones(()))

    def forward(self, h):
        return {name: h[:, 0] * self.scale for name in ("qed", "logp", "sas")}


def batch(n):
    return (torch.zeros(n, 1), torch.zeros(n, 1), torch.zeros(n, 1),
            torch.zeros(n, 1), torch.zeros(n, 3))


@pytest.mark.parametrize("kind", ["cd", "qc"])
def test_small_validation_tail_does_not_select_wrong_checkpoint(modules, monkeypatch, kind):
    cd, qc = modules
    module = cd if kind == "cd" else qc
    trainer_type = cd.CDTrainer if kind == "cd" else qc.QCTrainer
    monkeypatch.setattr(cd.CFG, "epochs", 2)
    monkeypatch.setattr(cd.CFG, "patience", 5)
    model = EnergyModel()
    trainer = trainer_type(Fingerprint(), model, Predictor())

    def step(crbm, *args, **kwargs):
        with torch.no_grad():
            crbm.version.add_(1)
        return 0.0

    # Epoch 1 has per-molecule values [0, 0, 9], epoch 2 [4, 4, 4].
    # Their means are 3 and 4, but a mean of batch means gives 4.5 and 4.
    means = iter([0., 9., 4., 4.])
    monkeypatch.setattr(module, "_cd_step" if kind == "cd" else "_qc_step", step)
    monkeypatch.setattr(module, "_reconstruction_free_energy_difference", lambda *args: next(means))
    monkeypatch.setattr(module, "_train_property_predictor", lambda *args: [])
    history = trainer.train([batch(2)], [batch(2), batch(1)])
    assert model.version.item() == 1.0  # restore the first epoch, not the second
    assert [row["val_free_energy_difference"] for row in history] == pytest.approx([3., 4.])


@pytest.mark.parametrize("sizes", [(2, 1), (1, 1, 1), (3,)])
def test_predictor_validation_is_invariant_to_batch_partition(modules, monkeypatch, sizes):
    cd, _ = modules
    monkeypatch.setattr(cd.CFG, "epochs", 1)
    model = EnergyModel()
    model.encode = lambda y, f: f
    model.encode_unknown = lambda f: f
    predictor = Predictor()
    optimizer = torch.optim.SGD(predictor.parameters(), lr=0.)
    values = torch.tensor([[0.], [0.], [3.]])
    batches = []
    start = 0
    for n in sizes:
        batches.append((values[start:start+n], torch.zeros(n, 1), torch.zeros(n, 3)))
        start += n
    history = cd._train_property_predictor(model, predictor, optimizer, batches, batches, "test")
    # Sum of three property MSEs: 3 * (0 + 0 + 9) / 3 = 9.
    assert history[0]["val_pred_loss"] == pytest.approx(9.)
