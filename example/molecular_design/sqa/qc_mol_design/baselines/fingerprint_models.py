"""
Baseline property prediction models from Table 1 of the paper:
  - Rule-based ECFP (2048-dim)   [Extended Connectivity Fingerprint, radius 2]
  - Rule-based MACCS (512-dim)   [MACCS structural keys, zero-padded to 512]

These fingerprints are converted to numpy arrays and then used as features
for a trained feedforward (PropertyMLP) network, reproducing the
"Rule-based" rows of Table 1.
"""
from __future__ import annotations

import warnings
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from qc_mol_design.config import CFG
from qc_mol_design.models.predictors import DirectPropertyMLP

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# RDKit fingerprint computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_ecfp(smiles_list: List[str], radius: int = 2, nbits: int = 2048) -> np.ndarray:
    """
    Compute ECFP (Extended Connectivity Fingerprint) for a list of SMILES.
    Returns (N, nbits) float32 array.

    Uses rdFingerprintGenerator.MorganGenerator (RDKit ≥ 2022.03)
    to avoid the deprecation warning from AllChem.GetMorganFingerprintAsBitVect.
    """
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator

    morgan_gen = rdFingerprintGenerator.GetMorganGenerator(
        radius=radius, fpSize=nbits
    )
    fps = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            fps.append(np.zeros(nbits, dtype=np.float32))
        else:
            fp = morgan_gen.GetFingerprintAsNumPy(mol).astype(np.float32)
            fps.append(fp)
    return np.stack(fps)


def compute_maccs(smiles_list: List[str], nbits: int = 512) -> np.ndarray:
    """
    Compute MACCS structural keys for a list of SMILES.
    Returns (N, nbits) float32 array.

    Paper Table 1 uses MACCS(512).  RDKit's MACCS is fixed at 167 bits,
    so we zero-pad to nbits=512 to match the paper's dimensionality.
    """
    from rdkit import Chem
    from rdkit.Chem import MACCSkeys

    fps = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            fps.append(np.zeros(nbits, dtype=np.float32))
        else:
            fp = MACCSkeys.GenMACCSKeys(mol)
            arr = np.zeros(nbits, dtype=np.float32)
            arr[:167] = np.array(fp, dtype=np.float32)
            fps.append(arr)
    return np.stack(fps)


# ─────────────────────────────────────────────────────────────────────────────
# Fingerprint Dataset
# ─────────────────────────────────────────────────────────────────────────────

class FingerprintDataset(torch.utils.data.Dataset):
    """Dataset of (fingerprint, property) pairs for a single property target."""

    def __init__(
        self,
        fps:   np.ndarray,      # (N, fp_dim)
        props: np.ndarray,      # (N,) continuous property values
    ) -> None:
        self.fps   = torch.FloatTensor(fps)
        self.props = torch.FloatTensor(props)

    def __len__(self) -> int:
        return len(self.fps)

    def __getitem__(self, idx: int):
        return self.fps[idx], self.props[idx]


# ─────────────────────────────────────────────────────────────────────────────
# Baseline predictor training & evaluation
# ─────────────────────────────────────────────────────────────────────────────

def train_fingerprint_predictor(
    train_fps:   np.ndarray,    # (N_train, fp_dim)
    train_props: np.ndarray,    # (N_train,)
    fp_dim:      int,
    epochs:      int = 50,
    lr:          float = 1e-3,
) -> DirectPropertyMLP:
    """Train a DirectPropertyMLP on pre-computed fingerprints."""
    device = CFG.device
    model  = DirectPropertyMLP(fp_dim).to(device)
    optim  = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=CFG.weight_decay)

    ds     = FingerprintDataset(train_fps, train_props)
    loader = torch.utils.data.DataLoader(ds, batch_size=CFG.batch_size, shuffle=True)

    model.train()
    for epoch in range(epochs):
        for fp_b, prop_b in loader:
            fp_b, prop_b = fp_b.to(device), prop_b.to(device)
            optim.zero_grad()
            loss = F.mse_loss(model(fp_b), prop_b)
            loss.backward()
            optim.step()

    return model


def evaluate_fingerprint_predictor(
    model:       DirectPropertyMLP,
    test_fps:    np.ndarray,
    test_props:  np.ndarray,
) -> float:
    """Return MAE on test set."""
    device = CFG.device
    model.eval()
    fps_t  = torch.FloatTensor(test_fps).to(device)
    with torch.no_grad():
        preds = model(fps_t).cpu().numpy()
    return float(np.abs(preds - test_props).mean())


# ─────────────────────────────────────────────────────────────────────────────
# Table 1 baseline runner
# ─────────────────────────────────────────────────────────────────────────────

def run_fingerprint_baselines(
    train_df: pd.DataFrame,
    test_df:  pd.DataFrame,
    prop:     str,
    epochs:   int = 50,
) -> dict:
    """
    Train and evaluate ECFP and MACCS fingerprint baselines for a given property.

    Returns dict with keys 'ecfp_mae' and 'maccs_mae'.
    """
    train_smiles = train_df["smiles"].tolist()
    test_smiles  = test_df["smiles"].tolist()
    train_y      = train_df[prop].values.astype(np.float32)
    test_y       = test_df[prop].values.astype(np.float32)

    results = {}

    # ECFP baseline
    print(f"  [Baseline] Computing ECFP-2048 fingerprints ...")
    train_ecfp = compute_ecfp(train_smiles)
    test_ecfp  = compute_ecfp(test_smiles)
    ecfp_model = train_fingerprint_predictor(train_ecfp, train_y, 2048, epochs)
    results["ecfp_mae"] = evaluate_fingerprint_predictor(ecfp_model, test_ecfp, test_y)

    # MACCS baseline
    print(f"  [Baseline] Computing MACCS-512 fingerprints ...")
    train_maccs = compute_maccs(train_smiles)
    test_maccs  = compute_maccs(test_smiles)
    maccs_model = train_fingerprint_predictor(train_maccs, train_y, 512, epochs)
    results["maccs_mae"] = evaluate_fingerprint_predictor(maccs_model, test_maccs, test_y)

    return results
