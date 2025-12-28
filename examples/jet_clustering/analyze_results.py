import numpy as np
import csv
import matplotlib as mpl
mpl.use("Agg")

mpl.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
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
from typing import List, Dict

def load_results_csv(csv_file: str) -> List[Dict]:
    results = []
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            row['event_idx'] = int(float(row.get('event_idx', 0)))
            row['N'] = int(float(row.get('N', 0)))
            row['best_energy'] = float(row.get('best_energy', 0.0))
            row['build_time'] = float(row.get('build_time', 0.0))
            row['solve_time'] = float(row.get('solve_time', 0.0))
            row['is_feasible'] = str(row.get('is_feasible', 'true')).lower() == 'true'
            row['violation_rate'] = float(row.get('violation_rate', 0.0))
            
            for k in ["mean_energy", "std_energy", "min_energy"]:
                if k in row and row[k] != "":
                    row[k] = float(row[k])
                else:
                    row[k] = row['best_energy'] if k == "mean_energy" or k == "min_energy" else 0.0
            
            for k in ["n_steps", "n_runs", "n_evals"]:
                if k in row and row[k] != "":
                    row[k] = int(float(row[k]))
                else:
                    row[k] = 0
            
            if 'solver' in row:
                row['solver'] = row['solver']
            
            if row.get('n_runs', 0) > 0:
                row['avg_time_per_run'] = row['solve_time'] / row['n_runs']
            else:
                row['avg_time_per_run'] = row['solve_time']
            
            results.append(row)
    return results

def print_statistics(results: List[Dict], solver_name: str = "Baseline"):
    energies = [r['best_energy'] for r in results]
    solve_times = [r['solve_time'] for r in results]
    feasible_count = sum([r['is_feasible'] for r in results])
    
    print(f"\n{'='*60}")
    print(f"{solver_name} Statistics")
    print(f"{'='*60}")
    print(f"Total events: {len(results)}")
    print(f"\nEnergy:")
    print(f"  Mean: {np.mean(energies):.4f}")
    print(f"  Std:  {np.std(energies):.4f}")
    print(f"  Min:  {np.min(energies):.4f}")
    print(f"  Max:  {np.max(energies):.4f}")
    print(f"\nSolve Time (s):")
    print(f"  Mean: {np.mean(solve_times):.6f}")
    print(f"  Std:  {np.std(solve_times):.6f}")
    print(f"  Min:  {np.min(solve_times):.6f}")
    print(f"  Max:  {np.max(solve_times):.6f}")
    print(f"\nFeasibility:")
    print(f"  Feasible: {feasible_count}/{len(results)} ({100*feasible_count/len(results):.1f}%)")
    violation_rates = [r['violation_rate'] for r in results]
    print(f"  Average violation rate: {np.mean(violation_rates):.6f}")
    print(f"{'='*60}\n")

def sanitize_filename(s: str) -> str:
    return s.replace(' ', '_').replace('/', '_').replace('\\', '_')


def style_ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(which="both", top=False, right=False)

def plot_energy_vs_event(results: List[Dict], output_file: str = None):
    event_indices = [r['event_idx'] for r in results]
    energies = [r['best_energy'] for r in results]
    solver_name = results[0].get('solver', 'Baseline')
    
    if output_file is None:
        solver_safe = sanitize_filename(solver_name)
        output_file = f"energy_vs_event_{solver_safe}.png"

    fig, ax = plt.subplots(figsize=(3.35, 2.4))  
    ax.plot(event_indices, energies, 'o-', markersize=3, linewidth=1.2)
    ax.set_xlabel('Event index')
    ax.set_ylabel('Best energy')
    ax.margins(x=0.02, y=0.05)
    style_ax(ax)
    fig.tight_layout(pad=0.4)

    pdf_file = output_file.replace(".png", ".pdf")
    fig.savefig(pdf_file, bbox_inches="tight", transparent=True)
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Saved plot to {output_file} and {pdf_file}")
    plt.close(fig)

def plot_energy_histogram(results: List[Dict], output_file: str = None):
    energies = [r['best_energy'] for r in results]
    solver_name = results[0].get('solver', 'Baseline')
    
    if output_file is None:
        solver_safe = sanitize_filename(solver_name)
        output_file = f"energy_histogram_{solver_safe}.png"

    fig, ax = plt.subplots(figsize=(3.35, 2.4))
    ax.hist(energies, bins=20, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Best energy')
    ax.set_ylabel('Frequency')
    ax.margins(x=0.02, y=0.05)
    ax.axvline(np.mean(energies), linestyle='--', linewidth=1.0,
               label=f'Mean: {np.mean(energies):.2f}')
    ax.legend()
    style_ax(ax)
    fig.tight_layout(pad=0.4)

    pdf_file = output_file.replace(".png", ".pdf")
    fig.savefig(pdf_file, bbox_inches="tight", transparent=True)
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Saved plot to {output_file} and {pdf_file}")
    plt.close(fig)

def plot_solve_time_histogram(results: List[Dict], output_file: str = None):
    solve_times = [r['solve_time'] for r in results]
    solver_name = results[0].get('solver', 'Baseline')
    
    if output_file is None:
        solver_safe = sanitize_filename(solver_name)
        output_file = f"solve_time_histogram_{solver_safe}.png"

    fig, ax = plt.subplots(figsize=(3.35, 2.4))
    ax.hist(solve_times, bins=20, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Solve time (s)')
    ax.set_ylabel('Frequency')
    ax.margins(x=0.02, y=0.05)
    ax.axvline(np.mean(solve_times), linestyle='--', linewidth=1.0,
               label=f'Mean: {np.mean(solve_times):.4f}s')
    ax.legend()
    style_ax(ax)
    fig.tight_layout(pad=0.4)

    pdf_file = output_file.replace(".png", ".pdf")
    fig.savefig(pdf_file, bbox_inches="tight", transparent=True)
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Saved plot to {output_file} and {pdf_file}")
    plt.close(fig)

def plot_energy_vs_N(results: List[Dict], output_file: str = None):
    N_values = [r['N'] for r in results]
    energies = [r['best_energy'] for r in results]
    solver_name = results[0].get('solver', 'Baseline')
    
    if output_file is None:
        solver_safe = sanitize_filename(solver_name)
        output_file = f"energy_vs_N_{solver_safe}.png"

    unique_N = sorted(set(N_values))
    mean_energies = []
    std_energies = []
    
    for n in unique_N:
        n_energies = [e for i, e in enumerate(energies) if N_values[i] == n]
        mean_energies.append(np.mean(n_energies))
        std_energies.append(np.std(n_energies))
    fig, ax = plt.subplots(figsize=(3.35, 2.4))
    ax.scatter(N_values, energies, alpha=0.4, s=10, label='Individual events')
    ax.errorbar(unique_N, mean_energies, yerr=std_energies,
                fmt='o-', color='red', linewidth=1.5, markersize=4,
                capsize=3, capthick=1.2, label='Mean ± std')
    ax.set_xlabel('Number of particles (N)')
    ax.set_ylabel('Best energy')
    ax.margins(x=0.02, y=0.05)
    style_ax(ax)
    ax.legend()
    fig.tight_layout(pad=0.4)

    pdf_file = output_file.replace(".png", ".pdf")
    fig.savefig(pdf_file, bbox_inches="tight", transparent=True)
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Saved plot to {output_file} and {pdf_file}")
    plt.close(fig)

if __name__ == "__main__":
    import sys
    
    csv_file = "results_baseline.csv"
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
    
    try:
        results = load_results_csv(csv_file)
        solver_name = results[0].get('solver', 'Baseline')
        
        print_statistics(results, solver_name)
        
        plot_energy_vs_event(results)
        plot_energy_histogram(results)
        plot_solve_time_histogram(results)
        plot_energy_vs_N(results)
        
    except FileNotFoundError:
        print(f"Error: {csv_file} not found.")
        print("Please run baseline_solver.py first to generate results.")

