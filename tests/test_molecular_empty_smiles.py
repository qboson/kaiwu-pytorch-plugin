"""Zero-atom RDKit molecules must not enter molecular training data."""

import importlib
from pathlib import Path
import sys
import warnings

import pytest

pytest.importorskip("rdkit", reason="Reference validation requires the optional RDKit dependency")


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
            yield (importlib.import_module("qc_mol_design.data.loader"),
                   importlib.import_module("qc_mol_design.data.featurizer"))
    finally:
        for name in list(sys.modules):
            if name == prefix or name.startswith(prefix + "."):
                sys.modules.pop(name)
        sys.modules.update(saved)



@pytest.mark.parametrize("smiles", ["", " ", "\t"])
def test_loader_drops_empty_molecules(modules, tmp_path, smiles):
    import pandas as pd
    loader, _ = modules
    path = tmp_path / "molecules.csv"
    pd.DataFrame({"smiles": [smiles, "C", "CC"], "qed": [0.1, 0.2, 0.3],
                  "logp": [1., 2., 3.], "sas": [2., 3., 4.]}).to_csv(path, index=False)
    data = loader.load_zinc12k(path)
    assert data["smiles"].tolist() == ["C", "CC"]
    assert data["qed"].tolist() == [0.2, 0.3]


def test_featurizer_skips_zero_atom_graphs(modules):
    import pandas as pd
    _, featurizer = modules
    assert featurizer.mol_to_graph("") is None
    assert featurizer.mol_to_graph("not-a-smiles") is None
    data = pd.DataFrame({"smiles": ["", "C", "CC"], "qed": [0.1, 0.2, 0.3],
                         "logp": [1., 2., 3.], "sas": [2., 3., 4.]})
    dataset = featurizer.MoleculeDataset(data)
    assert len(dataset) == 2
    batch = featurizer.collate_graphs([dataset[i] for i in range(len(dataset))])
    assert batch[0].shape[:2] == (2, 2)
    assert batch[-1][:, 0].tolist() == pytest.approx([0.2, 0.3])


def test_single_atom_molecule_is_retained(modules):
    _, featurizer = modules
    graph = featurizer.mol_to_graph("C")
    assert graph is not None
    assert graph[0].shape[0] == 1
    assert graph[1].shape == (1, 1)
