import matplotlib as mpl
mpl.use("Agg")
mpl.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "axes.linewidth": 0.8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "xtick.minor.size": 1.5,
    "ytick.minor.size": 1.5,
    "legend.fontsize": 8,
    "legend.frameon": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

import matplotlib.pyplot as plt
from analyze_results import load_results_csv
import numpy as np


def style_ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(which="both", top=False, right=False)
    ax.minorticks_on()
    ax.margins(x=0.02, y=0.05)


def main():
    fastjet_rows = load_results_csv("results_baseline_sa.csv")
    kaiwu_rows = load_results_csv("results_kaiwu_sa.csv")
    energy_fastjet = np.array([r["best_energy"] for r in fastjet_rows])
    energy_kaiwu = np.array([r["best_energy"] for r in kaiwu_rows])
    all_energy = np.concatenate([energy_fastjet, energy_kaiwu])
    fig, ax = plt.subplots(figsize=(3.35, 2.4))
    bins = np.linspace(all_energy.min(), all_energy.max(), 21)
    ax.hist(energy_fastjet, bins=bins, alpha=0.6, label="Classical SA", color="C0", edgecolor="black", linewidth=0.4)
    ax.hist(energy_kaiwu, bins=bins, alpha=0.6, label="Kaiwu-SA", color="C1", edgecolor="black", linewidth=0.4)
    ax.set_xlabel("Best objective / energy")
    ax.set_ylabel("Counts")
    style_ax(ax)
    ax.legend(loc="best")
    fig.tight_layout(pad=0.3)
    fig.savefig("energy_hist_compare_sa_vs_kaiwu.pdf", bbox_inches="tight", pad_inches=0.01, transparent=True)
    fig.savefig("energy_hist_compare_sa_vs_kaiwu.png", dpi=300, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)


if __name__ == "__main__":
    main()


