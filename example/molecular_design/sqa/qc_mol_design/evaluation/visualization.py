"""
Visualisation utilities reproducing Figures 2, 3, 4, and 5 of the paper.

Figure 2 – t-SNE of latent representations
Figure 3 – KDE plots of LogP/QED for training vs generated, violin plots of SAS
Figure 4 – Property distribution comparison across generation methods
Figure 5 – Learning curves, annealing times, success rates
"""
from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

warnings.filterwarnings("ignore")

try:
    import matplotlib
    matplotlib.use("Agg")   # non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False


def _require_mpl() -> None:
    if not _HAS_MPL:
        raise ImportError("matplotlib is required for visualisation. "
                          "Install with: pip install matplotlib")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2: t-SNE of latent representations
# ─────────────────────────────────────────────────────────────────────────────

def plot_tsne(
    latent_train:     np.ndarray,   # (N_train, n_hid)
    latent_generated: np.ndarray,   # (N_gen,   n_hid)
    gen_props:        np.ndarray,   # (N_gen,) property values for colouring
    prop_name:        str   = "QED",
    save_path:        str   = "fig2_tsne.png",
    max_points:       int   = 2000,
) -> None:
    """Reproduce Figure 2: t-SNE visualisation of chemical space."""
    _require_mpl()
    from sklearn.manifold import TSNE

    # Sub-sample for speed
    n_tr  = min(max_points, len(latent_train))
    n_gen = min(max_points // 2, len(latent_generated))
    lat_tr  = latent_train[np.random.choice(len(latent_train), n_tr, replace=False)]
    lat_gen = latent_generated[np.random.choice(len(latent_generated), n_gen, replace=False)]
    gen_col = gen_props[:n_gen] if len(gen_props) >= n_gen else gen_props

    combined = np.vstack([lat_tr, lat_gen])
    labels   = np.array([0] * n_tr + [1] * n_gen)

    tsne  = TSNE(n_components=2, random_state=42, perplexity=30)
    emb   = tsne.fit_transform(combined)

    fig, ax = plt.subplots(figsize=(8, 7))
    sc_tr  = ax.scatter(emb[:n_tr, 0], emb[:n_tr, 1],
                         c="grey", s=6, alpha=0.4, label="Training")
    sc_gen = ax.scatter(emb[n_tr:, 0], emb[n_tr:, 1],
                         c=gen_col, cmap="viridis", s=10, alpha=0.7,
                         label=f"Generated (coloured by {prop_name})")
    plt.colorbar(sc_gen, ax=ax, label=prop_name)
    ax.legend(markerscale=3, fontsize=10)
    ax.set_title(f"t-SNE: Latent space  (training vs generated molecules)")
    ax.set_xlabel("t-SNE dim 1"); ax.set_ylabel("t-SNE dim 2")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[FIG2] Saved → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3: KDE density plots + violin SAS
# ─────────────────────────────────────────────────────────────────────────────

def plot_kde(
    train_df:     pd.DataFrame,
    generated_df: pd.DataFrame,
    save_path:    str = "fig3_kde.png",
) -> None:
    """Reproduce Figure 3a-b: KDE of LogP and QED for train vs generated."""
    _require_mpl()
    try:
        from scipy.stats import gaussian_kde
    except ImportError:
        print("[VIZ] scipy not available; skipping KDE plot.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    props = [("logp", "LogP"), ("qed", "QED")]

    for ax, (prop, label) in zip(axes, props):
        for df, name, color in [
            (train_df,     "Training",  "steelblue"),
            (generated_df, "Generated", "tomato"),
        ]:
            vals = df[prop].dropna().values
            if len(vals) < 5:
                continue
            kde  = gaussian_kde(vals)
            xmin, xmax = vals.min(), vals.max()
            xs   = np.linspace(xmin, xmax, 300)
            ax.fill_between(xs, kde(xs), alpha=0.4, color=color, label=name)
            ax.plot(xs, kde(xs), color=color, lw=1.5)
        ax.set_xlabel(label, fontsize=12)
        ax.set_ylabel("Density", fontsize=12)
        ax.legend(fontsize=10)
        ax.set_title(f"{label} distribution")

    plt.suptitle("Figure 3: KDE of molecular properties", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[FIG3] Saved → {save_path}")


def plot_violin_sas(
    generated_by_target: Dict[str, List[float]],   # {target_label: [sas_values]}
    save_path: str = "fig3_violin_sas.png",
) -> None:
    """Reproduce Figure 3c-d: violin plots of SAS scores by target condition."""
    _require_mpl()
    if not generated_by_target:
        return

    labels = list(generated_by_target.keys())
    data   = [generated_by_target[k] for k in labels]

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.5), 6))
    parts   = ax.violinplot(data, positions=range(len(labels)),
                             showmedians=True, showextrema=True)
    for pc in parts["bodies"]:
        pc.set_facecolor("skyblue")
        pc.set_alpha(0.7)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Synthetic Accessibility Score (SAS)", fontsize=12)
    ax.set_title("Figure 3c: SAS distribution by target condition")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[FIG3-violin] Saved → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4: Property distributions per target
# ─────────────────────────────────────────────────────────────────────────────

def plot_generation_distributions(
    results_by_target: Dict[str, Dict[str, List[float]]],
    prop: str,
    target_ranges: List[Tuple[float, float]],
    save_path: str = "fig4_gen_dist.png",
) -> None:
    """
    Reproduce Figure 4: stacked histograms comparing generated molecule
    property distributions to the training-set reference for each target.
    """
    _require_mpl()
    n_targets = len(target_ranges)
    if n_targets == 0:
        return

    fig, axes = plt.subplots(1, n_targets, figsize=(4 * n_targets, 4), sharey=False)
    if n_targets == 1:
        axes = [axes]

    for ax, (lo, hi) in zip(axes, target_ranges):
        key  = f"{prop}=[{lo:.3f},{hi:.3f}]"
        data = results_by_target.get(key, {})
        gen_vals = data.get("generated", [])
        trn_vals = data.get("training",  [])

        if gen_vals:
            ax.hist(gen_vals, bins=20, alpha=0.6, color="tomato",   label="QC Generated")
        if trn_vals:
            ax.hist(trn_vals, bins=20, alpha=0.6, color="steelblue", label="Training ref")

        ax.axvline(lo, color="black", linestyle="--", lw=1)
        ax.axvline(hi, color="black", linestyle="--", lw=1)
        ax.set_xlabel(prop.upper())
        ax.set_title(f"Target [{lo:.2f}, {hi:.2f}]", fontsize=10)
        ax.legend(fontsize=8)

    fig.suptitle(f"Figure 4: {prop.upper()} distribution per target condition")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[FIG4] Saved → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 5a: Learning curves
# ─────────────────────────────────────────────────────────────────────────────

def plot_learning_curves(
    cd_history:  List[Dict],
    qc_history:  List[Dict],
    save_path:   str = "fig5a_learning_curves.png",
) -> None:
    """Reproduce Figure 5a: original-minus-reconstructed free-energy curves."""
    _require_mpl()

    fig, ax = plt.subplots(1, 1, figsize=(7, 5))

    for history, label, color in [
        (cd_history, "CD-learning", "#ef3b2c"),
        (qc_history, "QC-assisted learning", "#8e44ad"),
    ]:
        if not history:
            continue
        epochs    = [h["epoch"] for h in history]
        differences = [h["free_energy_difference"] for h in history]
        ax.plot(epochs, differences, color=color, lw=2, label=label)
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Free energy difference")
    ax.legend()
    ax.grid(alpha=0.2)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[FIG5a] Saved → {save_path}")
