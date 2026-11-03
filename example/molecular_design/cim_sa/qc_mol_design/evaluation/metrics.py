"""
Evaluation metrics reproducing Tables 1 and 2 of the paper.

Table 1 – Property Prediction MAE:
  MAE computed over a 1000-molecule test set for QED, SAS, and LogP.

Table 2 – Molecular Generation Efficacy:
  For each target property range:
    • mean ± std of the property value for generated molecules
    • comparison against training set molecules in the same range

Additional metrics:
  • validity:  fraction of generated SMILES that are valid RDKit molecules
  • uniqueness: fraction of distinct SMILES among valid molecules
  • novelty:    fraction of valid unique SMILES not in the training set
"""
from __future__ import annotations

from typing import Dict, List, Optional, Set

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from qc_mol_design.config import CFG, PROP_BINS
from qc_mol_design.models.crbm import CRBM
from qc_mol_design.models.graph_conv import NeuralFingerprint
from qc_mol_design.models.predictors import MultiPropertyPredictor


# ─────────────────────────────────────────────────────────────────────────────
# Property prediction MAE  (Table 1)
# ─────────────────────────────────────────────────────────────────────────────

def compute_prediction_mae(
    gcn:    NeuralFingerprint,
    crbm:   CRBM,
    pred:   MultiPropertyPredictor,
    loader: DataLoader,
) -> Dict[str, float]:
    """
    Evaluate property prediction Mean Absolute Error on a data loader.

    Returns dict: {'qed': mae_qed, 'logp': mae_logp, 'sas': mae_sas}.
    """
    gcn.eval(); crbm.eval(); pred.eval()
    dev = CFG.device

    all_preds:  Dict[str, List[float]] = {p: [] for p in ("qed", "logp", "sas")}
    all_truths: Dict[str, List[float]] = {p: [] for p in ("qed", "logp", "sas")}

    with torch.no_grad():
        for X, adj, deg, y_vec, props in loader:
            X, adj, deg = (t.to(dev) for t in (X, adj, deg))
            y_vec = y_vec.to(dev)
            props = props.to(dev)   # (B, 3): [qed, logp, sas]

            f    = gcn(X, adj, deg)
            # Methods p.12: for unknown targets first find
            # y*=argmax_y p(y|f), then compute the latent representation.
            h    = crbm.encode_unknown(f)
            phat = pred(h)          # dict: prop → (B,)

            for i, p in enumerate(("qed", "logp", "sas")):
                all_preds[p].extend(phat[p].cpu().tolist())
                all_truths[p].extend(props[:, i].cpu().tolist())

    maes = {}
    for p in ("qed", "logp", "sas"):
        phat_arr = np.array(all_preds[p])
        true_arr = np.array(all_truths[p])
        maes[p]  = float(np.abs(phat_arr - true_arr).mean())
    return maes


# ─────────────────────────────────────────────────────────────────────────────
# Generation metrics  (Table 2)
# ─────────────────────────────────────────────────────────────────────────────

def _summarise_generation_legacy(
    generated:    List[Dict],
    prop:         str,
    target_range: tuple,
    train_df:     pd.DataFrame,
) -> Dict:
    """
    Compute generation efficacy statistics matching Table 2 of the paper.

    Returns a dict with keys:
      qed_mean, qed_std, logp_mean, logp_std, sas_mean, sas_std,
      n_generated, n_in_range, in_range_fraction
    """
    in_range = [m for m in generated if m.get("in_range", False)]

    # Training-set reference: molecules satisfying same target
    lo, hi     = target_range
    train_ref  = train_df[train_df[prop].between(lo, hi)]

    stats: Dict = {
        "n_generated":    len(generated),
        "n_in_range":     len(in_range),
        "in_range_pct":   100.0 * len(in_range) / max(len(generated), 1),
        "train_n_ref":    len(train_ref),
    }

    for key in ("qed", "logp", "sas"):
        gen_vals = [m[key] for m in in_range] if in_range else [float("nan")]
        trn_vals = train_ref[key].values if len(train_ref) else np.array([float("nan")])
        stats[f"gen_{key}_mean"]   = float(np.nanmean(gen_vals))
        stats[f"gen_{key}_std"]    = float(np.nanstd(gen_vals))
        stats[f"train_{key}_mean"] = float(np.nanmean(trn_vals))
        stats[f"train_{key}_std"]  = float(np.nanstd(trn_vals))

    return stats


# ─────────────────────────────────────────────────────────────────────────────
# Diversity / novelty metrics
# ─────────────────────────────────────────────────────────────────────────────

def _generation_quality_metrics_legacy(
    generated:     List[Dict],
    train_smiles:  Set[str],
) -> Dict[str, float]:
    """
    Compute validity, uniqueness, and novelty over generated molecules.
    """
    from rdkit import Chem

    all_smiles   = [m["smiles"] for m in generated]
    valid_smiles = [s for s in all_smiles if Chem.MolFromSmiles(s) is not None]
    unique_valid = list(set(valid_smiles))
    novel        = [s for s in unique_valid if s not in train_smiles]

    n_all    = max(len(all_smiles), 1)
    n_valid  = max(len(valid_smiles), 1)

    return {
        "validity":   len(valid_smiles) / n_all,
        "uniqueness": len(unique_valid) / n_valid,
        "novelty":    len(novel) / max(len(unique_valid), 1),
    }


# ─────────────────────────────────────────────────────────────────────────────

def summarise_generation(
    generated: List[Dict],
    prop: str,
    target_range: tuple,
    train_df: pd.DataFrame,
) -> Dict:
    """Summarise every solver read; invalids and duplicates stay in denominators."""
    lo, hi = target_range
    sampled = len(generated)
    valid = [
        row for row in generated
        if bool(row.get("valid", bool(row.get("smiles"))))
    ]
    hits = [
        row for row in valid
        if bool(row.get("target_hit", row.get("in_range", False)))
    ]
    unique_valid = {
        str(row.get("smiles", "")) for row in valid if row.get("smiles")
    }
    duplicates = max(len(valid) - len(unique_valid), 0)
    train_ref = train_df[(train_df[prop] >= lo) & (train_df[prop] < hi)]

    stats: Dict = {
        "n_sampled": sampled,
        "n_valid": len(valid),
        "n_invalid": sampled - len(valid),
        "n_unique_valid": len(unique_valid),
        "n_duplicates": duplicates,
        "n_in_range": len(hits),
        "validity_pct": 100.0 * len(valid) / max(sampled, 1),
        "uniqueness_pct": 100.0 * len(unique_valid) / max(len(valid), 1),
        "success_all_pct": 100.0 * len(hits) / max(sampled, 1),
        "success_valid_pct": 100.0 * len(hits) / max(len(valid), 1),
        "train_n_ref": len(train_ref),
        # Backward-compatible aliases, now with the honest all-read denominator.
        "n_generated": sampled,
        "in_range_pct": 100.0 * len(hits) / max(sampled, 1),
    }

    for key in ("qed", "logp", "sas"):
        hit_values = [
            float(row[key]) for row in hits
            if row.get(key) is not None and np.isfinite(float(row[key]))
        ]
        train_values = (
            train_ref[key].to_numpy(dtype=float)
            if len(train_ref) else np.array([float("nan")])
        )
        stats[f"gen_{key}_mean"] = (
            float(np.mean(hit_values)) if hit_values else float("nan")
        )
        stats[f"gen_{key}_std"] = (
            float(np.std(hit_values)) if hit_values else float("nan")
        )
        stats[f"train_{key}_mean"] = float(np.nanmean(train_values))
        stats[f"train_{key}_std"] = float(np.nanstd(train_values))
    return stats


def generation_quality_metrics(
    generated: List[Dict],
    train_smiles: Set[str],
) -> Dict[str, float]:
    """Validity/uniqueness/novelty over raw reads without dropping failures."""
    from rdkit import Chem

    canonical_train = set()
    for smiles in train_smiles:
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is not None:
            canonical_train.add(Chem.MolToSmiles(mol, canonical=True))

    valid_smiles: List[str] = []
    for row in generated:
        smiles = str(row.get("smiles", "") or "")
        if not smiles:
            continue
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            valid_smiles.append(Chem.MolToSmiles(mol, canonical=True))

    unique_valid = set(valid_smiles)
    novel = unique_valid - canonical_train
    return {
        "validity": len(valid_smiles) / max(len(generated), 1),
        "uniqueness": len(unique_valid) / max(len(valid_smiles), 1),
        "novelty": len(novel) / max(len(unique_valid), 1),
        "n_sampled": float(len(generated)),
        "n_valid": float(len(valid_smiles)),
        "n_unique_valid": float(len(unique_valid)),
    }


# Table 1 comparison runner
# ─────────────────────────────────────────────────────────────────────────────

def print_table1(results: Dict[str, Dict[str, float]]) -> None:
    """
    Pretty-print Table 1 comparison.

    results: { method_name: {'qed': mae, 'logp': mae, 'sas': mae} }
    """
    header = f"{'Method':<25} {'QED':>10} {'SAS':>10} {'LogP':>10}"
    sep    = "-" * len(header)
    print("\n" + sep)
    print("Table 1 – Mean Absolute Error for Property Prediction")
    print(sep)
    print(header)
    print(sep)
    for method, maes in results.items():
        row = (
            f"{method:<25} "
            f"{maes.get('qed',  float('nan')):>10.4f} "
            f"{maes.get('sas',  float('nan')):>10.4f} "
            f"{maes.get('logp', float('nan')):>10.4f}"
        )
        print(row)
    print(sep)
    print("Reference (paper CD-learning): QED≈0.12 | SAS≈0.75 | LogP≈1.33")
    print("Reference (paper QC-assisted): QED≈0.10 | SAS≈0.66 | LogP≈1.27")
    print(sep + "\n")


def print_table2(
    results:      Dict[str, Dict],
    target_prop:  str,
    target_range: tuple,
) -> None:
    """Pretty-print Table 2 row for a single target condition."""
    prop_label = target_prop.upper()
    lo, hi     = target_range
    print(f"[Table 2] {prop_label} ∈ [{lo:.3f}, {hi:.3f}]")
    for method, stats in results.items():
        if f"gen_{target_prop}_mean" in stats:
            m = stats[f"gen_{target_prop}_mean"]
            s = stats[f"gen_{target_prop}_std"]
            print(f"  {method:<20}: {m:.2f} ± {s:.3f}  "
                  f"(in-range: {stats['n_in_range']}/{stats['n_generated']})")
