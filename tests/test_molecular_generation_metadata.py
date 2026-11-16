"""Offline regression coverage for the strict entry point's solver metadata."""

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")
torch = pytest.importorskip("torch")
pytest.importorskip("rdkit")
pytest.importorskip("sklearn")

from rdkit import Chem
from rdkit.Chem import Crippen, QED
from rdkit.Contrib.SA_Score import sascorer


@pytest.fixture
def generation_entry(monkeypatch, tmp_path):
    # The CIM/SA and SQA examples have separate packages with the same name.
    original_path = sys.path[:]
    original_modules = {
        key: value for key, value in sys.modules.items()
        if key == "qc_mol_design" or key.startswith("qc_mol_design.")
    }
    for key in original_modules:
        del sys.modules[key]
    source = (Path(__file__).resolve().parents[1]
              / "example/molecular_design/sqa/run_generation_strict.py")
    spec = importlib.util.spec_from_file_location("molecular_generation_entry", source)
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)
    monkeypatch.setattr(entry, "PROJECT_ROOT", tmp_path)
    monkeypatch.setenv("QC_MAX_FEASIBILITY_BATCHES", "1")
    monkeypatch.setenv("QC_SOLVER_REPEATS", "1")
    monkeypatch.setenv("QC_QUBO_SOLVER", "offline-fixture")
    monkeypatch.delenv("QC_SQA_BUDGET_FILE", raising=False)
    try:
        yield entry
    finally:
        sys.path[:] = original_path
        for key in list(sys.modules):
            if key == "qc_mol_design" or key.startswith("qc_mol_design."):
                del sys.modules[key]
        sys.modules.update(original_modules)


@pytest.fixture
def offline_generation(generation_entry, monkeypatch, tmp_path):
    from qc_mol_design.config import CFG, ATOM_FEATURE_SCHEMA
    from qc_mol_design.data import load_zinc12k, split_dataset
    from qc_mol_design.models import CRBM, build_fingerprint_model
    from qc_mol_design.optimization import designer

    def unexpected_solver_call(*args, **kwargs):
        pytest.fail("Forced offline exploration must not call a solver")

    monkeypatch.setattr(designer, "solve_qubo", unexpected_solver_call)

    CFG.n_train, CFG.n_valid, CFG.n_test = 1, 1, 25
    CFG.gcn_hidden, CFG.gcn_layers, CFG.fp_dim, CFG.n_hidden = 8, 1, 8, 4
    CFG.data_path = str(tmp_path / "zinc250k.csv")
    rows = []
    for count in range(2, 29):
        smiles = "C" * count
        mol = Chem.MolFromSmiles(smiles)
        rows.append(dict(smiles=smiles, qed=QED.qed(mol),
                         logp=Crippen.MolLogP(mol), sas=sascorer.calculateScore(mol)))
    pd.DataFrame(rows).to_csv(CFG.data_path, index=False)
    _, _, test = split_dataset(load_zinc12k())
    references = [
        dict(reference_id=f"R{index:02}", test_position=index,
             source_index=len(smiles) - 2, smiles=smiles, canonical_smiles=smiles,
             n_heavy_atoms=len(smiles), n_qubo_vars=len(smiles) * (len(smiles) - 1) // 2,
             formal_charge=0, size_stratum="fixture")
        for index, smiles in enumerate(test.smiles)
    ]
    manifest = tmp_path / "references.json"
    manifest.write_text(json.dumps({"references": references}))
    reference = min(references, key=lambda row: row["n_heavy_atoms"])
    size = reference["n_heavy_atoms"]
    bits = "".join("1" if j == i + 1 else "0"
                   for i in range(size) for j in range(i + 1, size))
    exploration = tmp_path / "exploration.json"
    exploration.write_text(json.dumps({f'qed:0:{reference["reference_id"]}': [bits]}))
    checkpoint = tmp_path / "checkpoint.pt"
    torch.manual_seed(17)
    torch.save(dict(format_version=3, feature_schema=ATOM_FEATURE_SCHEMA,
                    gcn=build_fingerprint_model().state_dict(),
                    crbm=CRBM().state_dict()), checkpoint)
    output = tmp_path / "output"

    def run(solver):
        monkeypatch.setattr(sys, "argv", [
            str(generation_entry.SOURCE_ROOT / "run_generation_strict.py"),
            "--solver", solver, "--checkpoint", str(checkpoint),
            "--save-dir", str(output), "--reference-manifest", str(manifest),
            "--reference-ids", reference["reference_id"], "--bands", "0",
            "--n-explore", "1", "--n-optimize", "0",
            "--exploration-file", str(exploration),
        ])
        generation_entry.main()
        assert not pd.read_csv(output / "samples_raw.csv").empty
        return json.loads((output / "run_meta.json").read_text())

    return run


@pytest.mark.parametrize("solver", ["kaiwu", "kaiwu_cim", "neal"])
@pytest.mark.parametrize("irrelevant_budget", [False, True])
def test_non_sqa_generation_ignores_sqa_budget(
    offline_generation, monkeypatch, tmp_path, solver, irrelevant_budget
):
    if irrelevant_budget:
        monkeypatch.setenv("QC_SQA_BUDGET_FILE", str(tmp_path / "absent-budget.json"))
    meta = offline_generation(solver)
    assert meta["solver"] == solver
    assert meta["protocol_version"] == 5
    assert meta["tuning_protocol"] == "matched_integer_inputs_validity_selection_v1"
    assert meta["params"]["sqa_schedule"] is None
    assert meta["params"]["budget_file_sha256"] is None
    assert meta["params"]["adaptive_budget_policy"] == (
        "Only acquire another batch when all candidates so far are invalid; record actual reads"
    )
    assert (meta["cim"] is not None) == (solver == "kaiwu_cim")


def test_sqa_generation_preserves_budget_metadata(offline_generation, monkeypatch, tmp_path):
    budget = tmp_path / "budget.json"
    contents = b'{"fixture": "forced exploration bypasses sampling"}\n'
    budget.write_bytes(contents)
    monkeypatch.setenv("QC_SQA_BUDGET_FILE", str(budget))
    meta = offline_generation("sqa")
    assert meta["protocol_version"] == 6
    assert meta["tuning_protocol"] == "sqa_cim_matched_observed_budget_v1"
    assert meta["params"]["budget_file_sha256"] == hashlib.sha256(contents).hexdigest()
    assert meta["params"]["sqa_schedule"] == (
        "continuous-time PIMC, Hp=geometric automatic beta endpoints, "
        "Hd=geometric(2,0.001) then 0, Gamma=1"
    )
    assert meta["params"]["adaptive_budget_policy"] == (
        "Frozen per-step observed CIM candidate counts; no SQA outcome-dependent resampling"
    )


@pytest.mark.parametrize("budget_kind", ["missing", "absent-file", "directory"])
def test_sqa_budget_fails_before_data_or_model_work(
    generation_entry, monkeypatch, tmp_path, budget_kind
):
    if budget_kind != "missing":
        path = tmp_path if budget_kind == "directory" else tmp_path / "absent.json"
        monkeypatch.setenv("QC_SQA_BUDGET_FILE", str(path))
    # No dataset or checkpoint exists. A late budget check would fail loading data first.
    output = tmp_path / "output"
    monkeypatch.setattr(sys, "argv", ["generation", "--solver", "sqa",
                                     "--save-dir", str(output)])
    with pytest.raises(ValueError, match="QC_SQA_BUDGET_FILE"):
        generation_entry.main()
    assert not output.exists()
