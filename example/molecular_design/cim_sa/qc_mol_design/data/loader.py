"""
ZINC-12k dataset loader.

The paper uses a 12,000-molecule subset of the ZINC database (Irwin & Shoichet 2005)
commonly benchmarked in graph neural network papers (Dwivedi et al. 2022).

This module:
  - Reads the local CSV (columns: type, smiles, qed, sas, logp)
  - Validates SMILES with RDKit
  - Splits into train / valid / test
  - Computes RDKit properties when not pre-computed
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from qc_mol_design.config import CFG, PROP_BINS, N_BINS_PER_PROP

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# Property computation helpers
# ─────────────────────────────────────────────────────────────────────────────

def _compute_properties(smiles_series: pd.Series) -> pd.DataFrame:
    """Compute QED, LogP, SAS for a Series of SMILES strings."""
    from rdkit import Chem
    from rdkit.Chem.QED import qed as rdkit_qed
    from rdkit.Chem.Crippen import MolLogP

    try:
        from rdkit.Contrib.SA_Score import sascorer
        _has_sas = True
    except Exception:
        _has_sas = False

    records = []
    for smi in tqdm(smiles_series, desc="Computing properties", leave=False):
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            records.append(None)
            continue
        try:
            qed  = float(rdkit_qed(mol))
            logp = float(MolLogP(mol))
            sas  = (float(sascorer.calculateScore(mol)) if _has_sas
                    else float(mol.GetNumHeavyAtoms()) / 10.0 + 1.0)
            records.append({"qed": qed, "logp": logp, "sas": sas})
        except Exception:
            records.append(None)

    df = pd.DataFrame([r if r else {"qed": np.nan, "logp": np.nan, "sas": np.nan}
                       for r in records])
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Main loader
# ─────────────────────────────────────────────────────────────────────────────

def load_zinc12k(path: Optional[str] = None) -> pd.DataFrame:
    """
    Load and validate the ZINC-12k molecular dataset.

    Returns a DataFrame with columns: smiles, qed, logp, sas.
    Rows with invalid SMILES or NaN properties are dropped.
    """
    from rdkit import Chem

    csv_path = Path(path or CFG.data_path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at '{csv_path}'.\n"
            "Download from: https://github.com/PEESEgroup/qc-camd"
        )

    df = pd.read_csv(csv_path)

    # Normalise column names: lowercase + strip whitespace
    df.columns = [c.lower().strip() for c in df.columns]
    # zinc250k uses 'logp' (from 'logP') and 'sas' (from 'SAS') after lower()

    # Retain training split when a legacy split column is present
    if "type" in df.columns:
        df = df[df["type"] == "train"].reset_index(drop=True)

    # Ensure SMILES column exists
    smiles_col = next(
        (c for c in df.columns if "smile" in c.lower()), df.columns[0]
    )
    if smiles_col != "smiles":
        df = df.rename(columns={smiles_col: "smiles"})

    # Strip trailing whitespace / newlines from SMILES
    df["smiles"] = df["smiles"].astype(str).str.strip()

    # Check for pre-computed property columns
    required_props = {"qed", "logp", "sas"}
    if required_props.issubset(set(df.columns)):
        data = df[["smiles", "qed", "logp", "sas"]].copy()
    else:
        print("[LOADER] Computing molecular properties via RDKit...")
        prop_df = _compute_properties(df["smiles"])
        data = pd.concat([df[["smiles"]], prop_df], axis=1)

    # Validate SMILES and drop bad rows
    valid_mask = data["smiles"].apply(
        lambda s: Chem.MolFromSmiles(str(s)) is not None
    )
    data = data.loc[valid_mask].dropna(subset=["qed", "logp", "sas"])
    data = data.reset_index(drop=True)
    print(f"[LOADER] {len(data)} valid molecules loaded.")
    return data


def split_dataset(
    data: pd.DataFrame,
    n_train: Optional[int] = None,
    n_valid: Optional[int] = None,
    n_test:  Optional[int] = None,
    seed:    int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Randomly shuffle and split into train / valid / test.
    Uses CFG defaults when sizes are not specified.
    """
    n_tr = n_train or CFG.n_train
    n_vl = n_valid or CFG.n_valid
    n_te = n_test  or CFG.n_test

    total = n_tr + n_vl + n_te
    if len(data) < total:
        raise ValueError(
            f"Dataset has {len(data)} molecules but {total} requested. "
            "Reduce n_train / n_valid / n_test in config."
        )

    data = data.sample(frac=1, random_state=seed).reset_index(drop=True)
    train_df = data.iloc[:n_tr].reset_index(drop=True)
    valid_df = data.iloc[n_tr : n_tr + n_vl].reset_index(drop=True)
    test_df  = data.iloc[n_tr + n_vl : n_tr + n_vl + n_te].reset_index(drop=True)

    print(f"[LOADER] Split → train:{len(train_df)} | valid:{len(valid_df)} | test:{len(test_df)}")
    return train_df, valid_df, test_df


# ─────────────────────────────────────────────────────────────────────────────
# Property discretisation
# ─────────────────────────────────────────────────────────────────────────────

def discretize(values: np.ndarray, prop: str) -> np.ndarray:
    """
    Map continuous property values to bin indices in [0, n_bins-1].

    Uses the bin edges defined in PROP_BINS[prop].
    """
    edges = PROP_BINS[prop]
    # digitize returns 0 for below first edge, len(edges) for above last
    indices = np.digitize(values, edges[1:-1])  # interior edges only
    return indices.clip(0, N_BINS_PER_PROP[prop] - 1)


def make_label_vector(
    qed_val: float, logp_val: float, sas_val: float
) -> np.ndarray:
    """
    Build the concatenated one-hot label vector y for the CRBM.

    y = [one_hot(qed_bin, n_bins_qed) | one_hot(logp_bin, n_bins_logp) | one_hot(sas_bin, n_bins_sas)]
    Length = N_BINS_PER_PROP["qed"] + N_BINS_PER_PROP["logp"] + N_BINS_PER_PROP["sas"]
    """
    from qc_mol_design.config import N_BINS_PER_PROP

    parts = []
    for prop, val in zip(("qed", "logp", "sas"), (qed_val, logp_val, sas_val)):
        n = N_BINS_PER_PROP[prop]
        idx = int(discretize(np.array([val]), prop)[0])
        oh = np.zeros(n, dtype=np.float32)
        oh[idx] = 1.0
        parts.append(oh)
    return np.concatenate(parts)
