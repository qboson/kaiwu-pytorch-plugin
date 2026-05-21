import os

import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
import seaborn as sns


def setup_plotting():
    sc.set_figure_params(dpi=150, frameon=False, figsize=(6, 6))
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["pdf.fonttype"] = 42
    sns.set_theme(style="whitegrid", palette="muted")


def plot_training_history(history, out_path=None):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    plot_kwargs = {"marker": "o", "markersize": 4, "linewidth": 1.5}

    plots = [
        (axes[0], "train_loss", "val_loss", "Total Loss"),
        (axes[1], "train_recon_loss", "val_recon_loss", "Reconstruction Loss"),
        (axes[2], "train_kl", "val_kl", "KL"),
    ]
    for ax, train_key, val_key, title in plots:
        ax.plot(epochs, history[train_key], label="Train", **plot_kwargs)
        ax.plot(epochs, history[val_key], label="Val", linestyle="--", **plot_kwargs)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(True, alpha=0.25)
        ax.set_xlim((0.5, 1.5) if len(epochs) == 1 else (1, len(epochs)))

    sns.despine()
    plt.tight_layout()
    if out_path is not None:
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    return fig


def save_umap_plots(adata, labels_key, batch_key, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sc.pl.umap(adata, color=labels_key, ax=axes[0], show=False, title="UMAP: Cell Types")
    sc.pl.umap(adata, color=batch_key, ax=axes[1], show=False, title="UMAP: Batches")
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "qvae_umap_labels_batches.png"), dpi=300)
    plt.close(fig)


def save_energy_plots(adata, labels_key, out_dir):
    energy = adata.obs["QVAE_Energy"]
    sc.pl.umap(
        adata,
        color="QVAE_Energy",
        cmap="coolwarm",
        title="UMAP: QVAE Energy",
        vmax=np.percentile(energy, 95),
        vmin=np.percentile(energy, 5),
        show=False,
    )
    plt.savefig(os.path.join(out_dir, "qvae_energy_umap.png"), dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(12, 5))
    order = adata.obs.groupby(labels_key, observed=True)["QVAE_Energy"].median().sort_values().index
    sns.violinplot(
        x=labels_key,
        y="QVAE_Energy",
        data=adata.obs,
        order=order,
        inner="quartile",
        palette="Spectral",
        density_norm="width",
        linewidth=1.2,
    )
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.title("QVAE Energy Distribution across Cell Types", fontsize=14, pad=15)
    plt.ylabel("QVAE Energy")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "qvae_energy_by_celltype.png"), dpi=300)
    plt.close()
