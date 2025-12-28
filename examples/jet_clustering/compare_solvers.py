import numpy as np
import csv
from analyze_results import load_results_csv, print_statistics

def compare_solvers(csv_files: list, solver_names: list = None):
    all_results = {}
    for csv_file in csv_files:
        try:
            results = load_results_csv(csv_file)
            if len(results) == 0:
                print(f"Warning: {csv_file} is empty, skipping")
                continue
            solver_name = results[0].get('solver', csv_file.split('_')[-1].replace('.csv', ''))
            all_results[solver_name] = results
        except FileNotFoundError:
            print(f"Warning: {csv_file} not found, skipping")
    
    print("\n" + "="*80)
    print("Comparison of Solvers")
    print("="*80)
    
    print(f"\n{'Solver':<20} {'Mean Energy':<15} {'Min Energy':<15} {'Mean Time (s)':<15} {'Std Time':<15} {'Feasible %':<12} {'Mean Std Energy':<15}")
    print("-"*95)
    
    comparison_data = []
    for name, results in all_results.items():
        energies = [r['best_energy'] for r in results]
        # Use avg_time_per_run when available for fair comparison; fall back to total solve_time
        solve_times = [r.get("avg_time_per_run", r["solve_time"]) for r in results]
        feasible_count = sum([r['is_feasible'] for r in results])
        
        mean_energy = np.mean(energies)
        min_energy = np.min(energies)
        mean_time = np.mean(solve_times)
        std_time = np.std(solve_times)
        feasible_pct = 100 * feasible_count / len(results)
        
        std_energies = [r.get('std_energy', 0.0) for r in results if 'std_energy' in r]
        mean_std_energy = np.mean(std_energies) if std_energies else 0.0
        
        comparison_data.append({
            'solver': name,
            'mean_energy': mean_energy,
            'min_energy': min_energy,
            'mean_time': mean_time,
            'std_time': std_time,
            'feasible_pct': feasible_pct,
            'mean_std_energy': mean_std_energy
        })
        
        print(f"{name:<20} {mean_energy:>14.2f} {min_energy:>14.2f} {mean_time:>14.4f} {std_time:>14.6f} {feasible_pct:>11.1f}% {mean_std_energy:>14.4f}")
    
    print("\n" + "="*95)
    
    if len(comparison_data) >= 2:
        best_mean = min(d['mean_energy'] for d in comparison_data)
        best_min = min(d['min_energy'] for d in comparison_data)
        fastest = min(d['mean_time'] for d in comparison_data)
        
        stable_vals = [d['mean_std_energy'] for d in comparison_data if d['mean_std_energy'] > 0]
        most_stable = min(stable_vals) if stable_vals else None
        
        print("\nBest performers:")
        for d in comparison_data:
            if abs(d['mean_energy'] - best_mean) < 0.01:
                print(f"  Lowest mean energy: {d['solver']} ({d['mean_energy']:.2f})")
        for d in comparison_data:
            if abs(d['min_energy'] - best_min) < 0.01:
                print(f"  Lowest min energy: {d['solver']} ({d['min_energy']:.2f})")
        for d in comparison_data:
            if abs(d['mean_time'] - fastest) < 0.001:
                print(f"  Fastest: {d['solver']} ({d['mean_time']:.4f}s)")
        if most_stable is not None:
            for d in comparison_data:
                if d['mean_std_energy'] > 0 and abs(d['mean_std_energy'] - most_stable) < 0.001:
                    print(f"  Most stable (lowest std): {d['solver']} ({d['mean_std_energy']:.4f})")
    
    print("\n" + "="*95)
    
    return comparison_data

if __name__ == "__main__":
    import sys
    
    csv_files = [
        "results_baseline_random.csv",
        "results_baseline_sa.csv"
    ]
    
    if len(sys.argv) > 1:
        csv_files = sys.argv[1:]
    
    available_files = [f for f in csv_files]
    compare_solvers(available_files)

