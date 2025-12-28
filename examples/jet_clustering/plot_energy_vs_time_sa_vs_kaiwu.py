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


def ensure_same_events(a, b):
    da = {r["event_idx"]: r for r in a}
    db = {r["event_idx"]: r for r in b}
    common = sorted(set(da.keys()) & set(db.keys()))
    a2 = [da[i] for i in common]
    b2 = [db[i] for i in common]
    return a2, b2


def main():
    sa = load_results_csv("results_baseline_sa.csv")
    kaiwu = load_results_csv("results_kaiwu_sa.csv")
    sa, kaiwu = ensure_same_events(sa, kaiwu)
    time_sa = np.array([r["avg_time_per_run"] for r in sa])
    energy_sa = np.array([r["best_energy"] for r in sa])
    time_kaiwu = np.array([r["avg_time_per_run"] for r in kaiwu])
    energy_kaiwu = np.array([r["best_energy"] for r in kaiwu])
    fig, ax = plt.subplots(figsize=(3.35, 2.4))
    ax.scatter(time_sa, energy_sa, label="Classical SA", color="C0", s=18, marker="o", alpha=0.8)
    ax.scatter(time_kaiwu, energy_kaiwu, label="Kaiwu-SA", color="C1", s=18, marker="s", alpha=0.8)
    ax.set_xlabel("Running time (s)")
    ax.set_ylabel("Objective / Energy")
    ax.set_yscale("log")
    ax.grid(which="major", linestyle=":", linewidth=0.4, alpha=0.6)
    ax.grid(which="minor", linestyle=":", linewidth=0.2, alpha=0.3)
    style_ax(ax)
    ax.legend(loc="best")
    fig.tight_layout(pad=0.3)
    fig.savefig("energy_vs_time_sa_vs_kaiwu.pdf", bbox_inches="tight", pad_inches=0.01, transparent=True)
    fig.savefig("energy_vs_time_sa_vs_kaiwu.png", dpi=300, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)


if __name__ == "__main__":
    main()


