# -*- coding: utf-8 -*-
"""Plotting helpers for QVAE-Anomaly scoring visualisations."""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def metric_text_block(metrics, rho=None, p=None, cm=None):
    """Compose a monospace multi-line metrics string for annotation."""
    lines = [
        f"F1={metrics['f1']:.3f}  PR={metrics['pr_auc']:.3f}  ROC={metrics['roc_auc']:.3f}",
        f"prec={metrics['precision']:.3f}  recall={metrics['recall']:.3f}",
    ]
    if rho is not None:
        lines.append(f"ρ(temp,energy)={rho:.2f}" + (f" (p={p:.1e})" if p is not None else ""))
    if cm is not None:
        lines.append(f"confusion [[TN,FP],[FN,TP]] = {np.asarray(cm).tolist()}")
    return "\n".join(lines)


def plot_score_distribution(energy, y, thr, metrics, cm, out_png,
                            title="Internal test energy by label",
                            label_low="low", label_normal="normal"):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    df = pd.DataFrame({"energy": energy, "label": [label_low if v == 1 else label_normal for v in y]})
    sns.boxplot(x="label", y="energy", data=df, order=[label_low, label_normal],
                hue="label", palette="Set2", width=0.4, showfliers=True,
                fliersize=3, legend=False, ax=ax)
    sns.stripplot(x="label", y="energy", data=df, order=[label_low, label_normal],
                  color="black", size=2, alpha=0.4, jitter=0.2, ax=ax)
    ax.axhline(float(thr), ls="--", c="grey", lw=1.5, label=f"val thr={float(thr):.1f}")
    ax.text(0.02, 0.02, metric_text_block(metrics, cm=cm), transform=ax.transAxes,
            va="bottom", ha="left", family="monospace", fontsize=9,
            bbox=dict(boxstyle="round", fc="white", ec="grey", alpha=0.9))
    ax.set_title(title)
    ax.set_ylabel("RBM Free Energy (higher = low-temp-like)")
    ax.legend(loc="upper right")
    plt.tight_layout(); plt.savefig(out_png, dpi=200); plt.close(fig)


def plot_internal_hist(energy, y, thr, metrics, cm, out_png,
                       label_low="low", label_normal="normal"):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    df = pd.DataFrame({"energy": energy, "label": [label_low if v == 1 else label_normal for v in y]})
    sns.histplot(data=df, x="energy", hue="label", hue_order=[label_low, label_normal],
                 bins=40, stat="density", common_norm=False, palette="Set2", alpha=0.5, ax=ax)
    ax.axvline(float(thr), ls="--", c="grey", lw=1.5)
    ax.text(float(thr), ax.get_ylim()[1] * 0.95, f" thr={float(thr):.1f}",
            color="grey", fontsize=9, va="top")
    ax.text(0.02, 0.98, metric_text_block(metrics, cm=cm), transform=ax.transAxes,
            va="top", ha="left", family="monospace", fontsize=9,
            bbox=dict(boxstyle="round", fc="white", ec="grey", alpha=0.9))
    ax.set_title("Internal test energy density by label")
    ax.set_xlabel("RBM Free Energy (higher = low-temp-like)")
    plt.tight_layout(); plt.savefig(out_png, dpi=200); plt.close(fig)


def plot_score_by_tempbin(scores, tbin, order, thr_val, thr_bench,
                          metrics, rho, p, out_png,
                          title="QVAE-Anomaly energy score by LDH temperature bin"):
    fig, ax = plt.subplots(figsize=(10, 5.5))
    df = pd.DataFrame({"energy": scores, "bin": list(tbin)})
    sns.boxplot(x="bin", y="energy", data=df, order=order, hue="bin",
                palette="coolwarm_r", width=0.5, showfliers=True,
                fliersize=4, legend=False, ax=ax)
    sns.stripplot(x="bin", y="energy", data=df, order=order, color="black",
                  size=3, alpha=0.6, jitter=0.15, ax=ax)
    ax.axhline(float(thr_val), ls="--", c="grey", lw=1.5,
               label=f"val thr={float(thr_val):.1f} (unbiased)")
    if thr_bench is not None:
        ax.axhline(float(thr_bench), ls=":", c="black", lw=1.5,
                   label=f"benchmark F1 thr={float(thr_bench):.1f}")
    ax.text(0.02, 0.02, metric_text_block(metrics, rho=rho, p=p, cm=np.asarray(metrics.get("confusion", []))),
            transform=ax.transAxes, va="bottom", ha="left",
            family="monospace", fontsize=9,
            bbox=dict(boxstyle="round", fc="white", ec="grey", alpha=0.9))
    ax.set_title(title)
    ax.set_ylabel("RBM Free Energy (higher = low-temp-like)")
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout(); plt.savefig(out_png, dpi=200); plt.close(fig)


def plot_pr_roc_curve(y_true, y_score, out_png, title="PR/ROC curve",
                      label="model"):
    """Plot PR and ROC curves side by side."""
    from sklearn.metrics import (precision_recall_curve, roc_curve,
                                 average_precision_score, roc_auc_score)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    p, r, _ = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    ax1.plot(r, p, lw=2, label=f"{label} (AP={ap:.3f})")
    ax1.set_xlabel("Recall"); ax1.set_ylabel("Precision")
    ax1.set_title("Precision-Recall"); ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc = roc_auc_score(y_true, y_score)
    ax2.plot(fpr, tpr, lw=2, label=f"{label} (AUC={roc:.3f})")
    ax2.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax2.set_xlabel("FPR"); ax2.set_ylabel("TPR")
    ax2.set_title("ROC"); ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    fig.suptitle(title)
    plt.tight_layout(); plt.savefig(out_png, dpi=200); plt.close(fig)


def plot_latent_tsne(z, y, out_png, title="Latent space t-SNE",
                     label_anomaly="anomaly", label_normal="normal"):
    """t-SNE 2D scatter of latent codes colored by label."""
    from sklearn.manifold import TSNE
    z_2d = TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(z)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(z_2d[y == 0, 0], z_2d[y == 0, 1], c="steelblue",
               s=5, alpha=0.5, label=label_normal)
    ax.scatter(z_2d[y == 1, 0], z_2d[y == 1, 1], c="crimson",
               s=8, alpha=0.7, label=label_anomaly)
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout(); plt.savefig(out_png, dpi=200); plt.close(fig)


def plot_feature_importance(importances, out_png,
                            feature_names=None, top_k=20,
                            title="Feature importance (F1 drop under permutation)"):
    """Bar plot of feature importances (higher = more important)."""
    imp = np.asarray(importances)
    order = np.argsort(imp)[::-1][:top_k]
    names = feature_names or [f"f{i}" for i in range(len(imp))]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.barh([names[i] for i in order][::-1], imp[order][::-1], color="steelblue")
    ax.set_xlabel("F1 drop when permuted")
    ax.set_title(title)
    plt.tight_layout(); plt.savefig(out_png, dpi=200); plt.close(fig)


def plot_timeseries_anomaly(time, energy, y, thr, out_png,
                            title="Energy score vs time"):
    """Scatter of energy over time, anomalies highlighted."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.scatter(time[y == 0], energy[y == 0], c="steelblue", s=2, alpha=0.3, label="normal")
    ax.scatter(time[y == 1], energy[y == 1], c="crimson", s=20, alpha=0.8, label="anomaly")
    ax.axhline(thr, ls="--", c="grey", lw=1.5, label=f"thr={thr:.1f}")
    ax.set_xlabel("Time (hour)")
    ax.set_ylabel("RBM Free Energy")
    ax.set_title(title)
    ax.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(out_png, dpi=200); plt.close(fig)


def plot_attack_type_box(energy, attack_type, out_png, thr=None,
                         order=None, title="Energy by attack type"):
    """Box plot of RBM free energy grouped by attack type (KDD)."""
    if order is None:
        order = ["normal", "dos", "probe", "r2l", "u2r"]
    df = pd.DataFrame({"energy": energy, "type": attack_type})
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=df, x="type", y="energy", order=order, ax=ax,
                palette="Set2", fliersize=2)
    if thr is not None:
        ax.axhline(thr, ls="--", c="grey", lw=1.5, label=f"thr={thr:.1f}")
        ax.legend(fontsize=8)
    ax.set_xlabel("Attack type")
    ax.set_ylabel("RBM Free Energy")
    ax.set_title(title)
    plt.tight_layout(); plt.savefig(out_png, dpi=200); plt.close(fig)
