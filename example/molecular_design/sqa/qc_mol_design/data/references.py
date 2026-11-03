"""Fixed reference-molecule manifest support for strict generation runs."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd


@dataclass(frozen=True)
class ReferenceMolecule:
    reference_id: str
    test_position: int
    source_index: int
    smiles: str
    canonical_smiles: str
    n_heavy_atoms: int
    n_qubo_vars: int
    formal_charge: int
    size_stratum: str
    legacy_source: str = ""


def _canonical_smiles(smiles: str) -> str:
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"invalid reference SMILES: {smiles}")
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def load_reference_manifest(
    path: str | Path,
    test_df: Optional[pd.DataFrame] = None,
    expected_count: int = 25,
) -> List[ReferenceMolecule]:
    """Load and fully validate the immutable reference set."""
    manifest_path = Path(path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw_refs = payload.get("references", payload)
    refs = [ReferenceMolecule(**item) for item in raw_refs]

    if len(refs) != expected_count:
        raise ValueError(
            f"reference manifest contains {len(refs)} entries; expected {expected_count}"
        )
    ids = [ref.reference_id for ref in refs]
    if len(set(ids)) != len(ids):
        raise ValueError("reference manifest contains duplicate reference_id values")
    canonical = [ref.canonical_smiles for ref in refs]
    if len(set(canonical)) != len(canonical):
        raise ValueError("reference manifest contains duplicate molecules")

    from rdkit import Chem

    for ref in refs:
        mol = Chem.MolFromSmiles(ref.smiles)
        if mol is None:
            raise ValueError(f"{ref.reference_id} has invalid SMILES")
        observed_canonical = Chem.MolToSmiles(
            mol, canonical=True, isomericSmiles=True
        )
        n_atoms = mol.GetNumHeavyAtoms()
        if observed_canonical != ref.canonical_smiles:
            raise ValueError(f"{ref.reference_id} canonical SMILES mismatch")
        if n_atoms != ref.n_heavy_atoms:
            raise ValueError(f"{ref.reference_id} heavy-atom count mismatch")
        if n_atoms * (n_atoms - 1) // 2 != ref.n_qubo_vars:
            raise ValueError(f"{ref.reference_id} QUBO-variable count mismatch")

    if test_df is not None:
        test_canonical = {_canonical_smiles(str(s)) for s in test_df["smiles"]}
        missing = [r.reference_id for r in refs if r.canonical_smiles not in test_canonical]
        if missing:
            raise ValueError(f"references absent from fixed test split: {missing}")

    return refs


def select_references(
    refs: Iterable[ReferenceMolecule],
    reference_ids: Optional[Iterable[str]] = None,
) -> List[ReferenceMolecule]:
    refs = list(refs)
    if reference_ids is None:
        return refs
    wanted = list(reference_ids)
    by_id = {ref.reference_id: ref for ref in refs}
    unknown = [ref_id for ref_id in wanted if ref_id not in by_id]
    if unknown:
        raise ValueError(f"unknown reference IDs: {unknown}")
    return [by_id[ref_id] for ref_id in wanted]


def manifest_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
