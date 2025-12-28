import os
import csv
import numpy as np
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

SA_COLOR = "C0"
RND_COLOR = "C1"

def style_ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(which="both", top=False, right=False)
    ax.margins(x=0.02, y=0.05)
    ax.minorticks_on()
    ax.tick_params(which="minor", length=1.5)


def read_csv_results(path, default_solver_name):
    rows = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for r in reader:
            out = {}
            out["event_idx"] = int(float(r.get("event_idx", 0)))
            out["N"] = int(float(r.get("N", 0)))
            out["best_energy"] = float(r.get("best_energy", "nan"))
            out["solve_time"] = float(r.get("solve_time", "nan"))
            out["build_time"] = float(r.get("build_time", 0.0)) if "build_time" in r else 0.0
            out["is_feasible"] = (str(r.get("is_feasible", "true")).lower() == "true")
            out["violation_rate"] = float(r.get("violation_rate", 0.0)) if "violation_rate" in r else 0.0
            n_runs = r.get("n_runs", "")
            if n_runs != "":
                out["n_runs"] = int(float(n_runs))
            else:
                out["n_runs"] = 0
            if out["n_runs"] and out["n_runs"] > 0:
                out["avg_time_per_run"] = out["solve_time"] / out["n_runs"]
            else:
                out["avg_time_per_run"] = out["solve_time"]
            out["solver"] = r.get("solver", default_solver_name)
            rows.append(out)
    rows.sort(key=lambda x: x["event_idx"])
    rows = [r for r in rows if np.isfinite(r["best_energy"]) and np.isfinite(r["solve_time"])]
    return rows


def ensure_same_events(a, b):
    da = {r["event_idx"]: r for r in a}
    db = {r["event_idx"]: r for r in b}
    common = sorted(set(da.keys()) & set(db.keys()))
    a2 = [da[i] for i in common]
    b2 = [db[i] for i in common]
    return a2, b2, common


def save_fig(fig, base):
    pdf = base + ".pdf"
    png = base + ".png"
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.01, transparent=True)
    fig.savefig(png, dpi=300, bbox_inches="tight", pad_inches=0.01)
    print(f"Saved: {pdf}")
    print(f"Saved: {png}")


def plot_energy_vs_event(a, b, outbase):
    a, b, _ = ensure_same_events(a, b)
    x = [r["event_idx"] for r in a]
    ya = [r["best_energy"] for r in a]
    yb = [r["best_energy"] for r in b]
    la = a[0]["solver"] if a else "A"
    lb = b[0]["solver"] if b else "B"
    fig, ax = plt.subplots(figsize=(3.35, 2.4))
    # Plot with different styles and transparency to distinguish overlapping lines
    ax.plot(x, ya, marker="o", markersize=3.5, linewidth=1.4, label=la, color=SA_COLOR, 
            alpha=0.8, linestyle='-', markeredgewidth=0.8, markerfacecolor='none', markeredgecolor=SA_COLOR)
    ax.plot(x, yb, marker="s", markersize=2.8, linewidth=1.2, label=lb, color=RND_COLOR, 
            alpha=0.85, linestyle='--')
    ax.set_xlabel("Event index")
    ax.set_ylabel("Best energy")
    style_ax(ax)
    ax.legend(loc="best")
    fig.tight_layout(pad=0.3)
    save_fig(fig, outbase)
    plt.close(fig)


def plot_energy_boxplot(a, b, outbase):
    a, b, _ = ensure_same_events(a, b)
    la = a[0]["solver"] if a else "A"
    lb = b[0]["solver"] if b else "B"
    ea = [r["best_energy"] for r in a]
    eb = [r["best_energy"] for r in b]
    fig, ax = plt.subplots(figsize=(3.35, 2.4))
    bp = ax.boxplot(
        [ea, eb],
        labels=[la, lb],
        showfliers=False,
        widths=0.55,
        patch_artist=True,
        medianprops={"linewidth": 1.2},
        boxprops={"linewidth": 1.0},
        whiskerprops={"linewidth": 1.0},
        capprops={"linewidth": 1.0},
    )
    for patch, c in zip(bp["boxes"], [SA_COLOR, RND_COLOR]):
        patch.set_facecolor(c)
        patch.set_alpha(0.25)
    ax.set_ylabel("Best energy")
    style_ax(ax)
    fig.tight_layout(pad=0.3)
    save_fig(fig, outbase)
    plt.close(fig)


def plot_energy_vs_N(a, b, outbase):
    a, b, _ = ensure_same_events(a, b)
    la = a[0]["solver"] if a else "A"
    lb = b[0]["solver"] if b else "B"

    def summarize(rows):
        Ns = sorted(set(r["N"] for r in rows))
        mu, sd = [], []
        for n in Ns:
            vals = [r["best_energy"] for r in rows if r["N"] == n]
            mu.append(float(np.mean(vals)))
            sd.append(float(np.std(vals)))
        return Ns, mu, sd

    Na, mua, sda = summarize(a)
    Nb, mub, sdb = summarize(b)
    fig, ax = plt.subplots(figsize=(3.35, 2.4))
    # Plot SA with slightly offset x-coordinates and different style for visibility
    ax.errorbar([n - 0.05 for n in Na], mua, yerr=sda, marker="o", markersize=3.5, 
                linewidth=1.2, capsize=2.5, label=la, color=SA_COLOR, alpha=0.85, linestyle='-')
    ax.errorbar([n + 0.05 for n in Nb], mub, yerr=sdb, marker="s", markersize=3.2, 
                linewidth=1.2, capsize=2.5, label=lb, color=RND_COLOR, alpha=0.85, linestyle='--')
    ax.set_xlabel("Number of particles (N)")
    ax.set_ylabel("Best energy (mean ± std)")
    style_ax(ax)
    ax.legend(loc="best")
    fig.tight_layout(pad=0.3)
    save_fig(fig, outbase)
    plt.close(fig)


def plot_pareto_energy_time(a, b, outbase):
    a, b, _ = ensure_same_events(a, b)
    la = a[0]["solver"] if a else "A"
    lb = b[0]["solver"] if b else "B"
    xa = [r["avg_time_per_run"] for r in a]
    ya = [r["best_energy"] for r in a]
    xb = [r["avg_time_per_run"] for r in b]
    yb = [r["best_energy"] for r in b]

    frontier_xa, frontier_ya = [], []
    frontier_xb, frontier_yb = [], []

    combined_a = sorted(zip(xa, ya), key=lambda t: t[0])
    min_e = float("inf")
    for x, y in combined_a:
        if y < min_e:
            min_e = y
            frontier_xa.append(x)
            frontier_ya.append(y)

    combined_b = sorted(zip(xb, yb), key=lambda t: t[0])
    min_e = float("inf")
    for x, y in combined_b:
        if y < min_e:
            min_e = y
            frontier_xb.append(x)
            frontier_yb.append(y)

    fig, ax = plt.subplots(figsize=(3.35, 2.4))
    ax.scatter(xa, ya, s=18, marker="o", alpha=0.75, label=la, color=SA_COLOR)
    ax.scatter(xb, yb, s=18, marker="s", alpha=0.75, label=lb, color=RND_COLOR)

    if len(frontier_xa) > 1:
        ax.plot(frontier_xa, frontier_ya, linestyle="--", linewidth=1.0, alpha=0.6)
    if len(frontier_xb) > 1:
        ax.plot(frontier_xb, frontier_yb, linestyle="--", linewidth=1.0, alpha=0.6)

    if xa and xb:
        time_min = min(min(xa), min(xb))
        time_max = max(max(xa), max(xb))
        if time_min > 0 and (time_max / time_min) > 10:
            ax.set_xscale("log")
            ax.tick_params(which="minor", length=0.8)

    ax.set_xlabel("Time per run (s)")
    ax.set_ylabel("Best energy")
    style_ax(ax)
    ax.legend(loc="best")
    fig.tight_layout(pad=0.3)
    save_fig(fig, outbase)
    plt.close(fig)


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    random_csv = os.path.join(base_dir, "results_baseline_random.csv")
    sa_csv = os.path.join(base_dir, "results_baseline_sa.csv")
    outdir = base_dir
    os.makedirs(outdir, exist_ok=True)
    random_rows = read_csv_results(random_csv, default_solver_name="Random")
    sa_rows = read_csv_results(sa_csv, default_solver_name="SA")
    print(f"Loaded Random: {len(random_rows)} events")
    print(f"Loaded SA: {len(sa_rows)} events")
    sa_common, random_common, common_ids = ensure_same_events(sa_rows, random_rows)
    print(f"Common events: {len(common_ids)}")
    plot_energy_vs_event(sa_common, random_common, os.path.join(outdir, "energy_vs_event_compare"))
    plot_energy_boxplot(sa_common, random_common, os.path.join(outdir, "energy_boxplot_compare"))
    plot_energy_vs_N(sa_common, random_common, os.path.join(outdir, "energy_vs_N_compare"))
    plot_pareto_energy_time(sa_common, random_common, os.path.join(outdir, "pareto_energy_time"))


