import numpy as np
import csv
import time
from typing import Tuple
from qubo_model import (
    K, build_qubo_matrix_k2, qubo_energy, decode_assignment,
    check_feasibility, load_DR_from_npz
)

def estimate_initial_temperature(Q: np.ndarray, N: int, n_samples: int = 100, seed: int = 0) -> float:
    """
    Estimate a reasonable initial temperature scale by sampling random single-spin flips
    and measuring the typical energy change |ΔE|.
    """
    rng = np.random.default_rng(seed)

    x = np.zeros(N * K, dtype=int)
    labels = rng.integers(0, K, size=N)
    for i in range(N):
        x[i * K + labels[i]] = 1

    E_current = qubo_energy(Q, x)
    delta_E_samples = []

    for _ in range(n_samples):

        i = rng.integers(0, N)
        old_jet = labels[i]
        new_jet = rng.integers(0, K)

        if new_jet == old_jet:
            continue

        x[i * K + old_jet] = 0
        x[i * K + new_jet] = 1
        E_after = qubo_energy(Q, x)
        delta_E_samples.append(abs(E_after - E_current))
        x[i * K + new_jet] = 0
        x[i * K + old_jet] = 1

    if len(delta_E_samples) > 0:
        std_delta = np.std(delta_E_samples)
        return 0.5 * std_delta if std_delta > 0 else 10.0
    return 100.0

def classical_simulated_annealing(
    Q: np.ndarray,
    N: int,
    T_init: float = None,
    T_final: float = 0.01,
    n_steps: int = 10000,
    seed: int = 0,
    adaptive_T: bool = True
) -> Tuple[np.ndarray, float]:
    rng = np.random.default_rng(seed)
    
    x = np.zeros(N * K, dtype=int)
    labels = rng.integers(0, K, size=N)
    for i in range(N):
        x[i * K + labels[i]] = 1
    
    E_current = qubo_energy(Q, x)
    E_best = E_current
    x_best = x.copy()
    
    if adaptive_T:
        T_est = estimate_initial_temperature(Q, N, n_samples=100, seed=seed)
        if T_init is None:
            T_init = T_est
        else:
            T_init = 0.5 * (T_init + T_est)
    elif T_init is None:
        T_init = estimate_initial_temperature(Q, N, n_samples=100, seed=seed)
    
    cooling_rate = (T_final / T_init) ** (1.0 / n_steps)
    T = T_init
    
    for step in range(n_steps):
        i = rng.integers(0, N)
        old_jet = labels[i]
        new_jet = rng.integers(0, K)
        
        if new_jet == old_jet:
            continue
        
        x[i * K + old_jet] = 0
        x[i * K + new_jet] = 1
        
        E_new = qubo_energy(Q, x)
        delta_E = E_new - E_current
        
        if delta_E < 0 or rng.random() < np.exp(-delta_E / T):
            labels[i] = new_jet
            E_current = E_new
            
            if E_current < E_best:
                E_best = E_current
                x_best = x.copy()
        else:
            x[i * K + new_jet] = 0
            x[i * K + old_jet] = 1
        
        T *= cooling_rate
    
    return x_best, E_best

def solve_single_event_sa(
    npz_file: str,
    event_idx: int,
    lambda_constraint: float = 10.0,
    w_mode: str = "dr2",
    T_init: float = None,
    T_final: float = 0.01,
    n_steps: int = 10000,
    n_runs: int = 1,
    seed: int = 0,
    adaptive_T: bool = True
) -> dict:
    DR = load_DR_from_npz(npz_file, event_idx)
    N = DR.shape[0]
    
    t_start = time.time()
    Q = build_qubo_matrix_k2(DR, lambda_constraint=lambda_constraint, w_mode=w_mode)
    build_time = time.time() - t_start
    
    energies = []
    best_x_overall = None
    best_E_overall = 1e100
    
    t_start = time.time()
    for run in range(n_runs):
        best_x, best_E = classical_simulated_annealing(
            Q, N, T_init=T_init, T_final=T_final, n_steps=n_steps, 
            seed=seed + run, adaptive_T=adaptive_T
        )
        energies.append(best_E)
        if best_E < best_E_overall:
            best_E_overall = best_E
            best_x_overall = best_x.copy()
    solve_time = time.time() - t_start
    
    labels = decode_assignment(best_x_overall, N)
    is_feasible, violation_rate = check_feasibility(best_x_overall, N)
    
    result = {
        "event_idx": event_idx,
        "N": N,
        "best_energy": best_E_overall,
        "mean_energy": np.mean(energies),
        "std_energy": np.std(energies),
        "min_energy": np.min(energies),
        "build_time": build_time,
        "solve_time": solve_time,
        "is_feasible": is_feasible,
        "violation_rate": violation_rate,
        "labels": labels,
        "solver": "SA",
        "n_steps": n_steps,
        "n_runs": n_runs,
        "n_evals": n_steps * n_runs,
        "avg_time_per_run": solve_time / n_runs if n_runs > 0 else solve_time
    }
    
    return result

def batch_solve_events_sa(
    npz_file: str,
    n_events: int,
    lambda_constraint: float = 10.0,
    w_mode: str = "dr2",
    T_init: float = None,
    T_final: float = 0.01,
    n_steps: int = 10000,
    n_runs: int = 1,
    seed: int = 0,
    adaptive_T: bool = True,
    output_csv: str = "results_baseline_sa.csv"
) -> list:
    results = []
    
    for event_idx in range(n_events):
        print(f"Solving event {event_idx}...", end=" ")
        result = solve_single_event_sa(
            npz_file, event_idx,
            lambda_constraint=lambda_constraint,
            w_mode=w_mode,
            T_init=T_init,
            T_final=T_final,
            n_steps=n_steps,
            n_runs=n_runs,
            seed=seed + event_idx * 1000,
            adaptive_T=adaptive_T
        )
        results.append(result)
        if n_runs > 1:
            print(f"N={result['N']}, E={result['best_energy']:.2f} (mean={result['mean_energy']:.2f}±{result['std_energy']:.2f}), feasible={result['is_feasible']}")
        else:
            print(f"N={result['N']}, E={result['best_energy']:.2f}, feasible={result['is_feasible']}")
    
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "event_idx", "N", "best_energy", "mean_energy", "std_energy", "min_energy",
            "build_time", "solve_time", "avg_time_per_run", "is_feasible", "violation_rate",
            "solver", "n_steps", "n_runs", "n_evals"
        ])
        for r in results:
            writer.writerow([
                r["event_idx"], r["N"], r["best_energy"],
                r.get("mean_energy", r["best_energy"]),
                r.get("std_energy", 0.0),
                r.get("min_energy", r["best_energy"]),
                r["build_time"], r["solve_time"], r.get("avg_time_per_run", r["solve_time"]),
                r["is_feasible"], r["violation_rate"], r["solver"],
                r.get("n_steps", 0), r.get("n_runs", 1), r.get("n_evals", 0)
            ])
    
    print(f"\nSaved results to {output_csv}")
    return results

if __name__ == "__main__":
    npz_file = "events_data.npz"
    
    try:
        data = np.load(npz_file)
        n_events = data["n_events"][0]
        print(f"Found {n_events} events in {npz_file}\n")
        
        DR0 = load_DR_from_npz(npz_file, 0)
        N0 = DR0.shape[0]
        Q0 = build_qubo_matrix_k2(DR0, lambda_constraint=10.0, w_mode="dr2")
        
        print(f"Testing event 0: N={N0}")
        print(f"QUBO matrix shape: {Q0.shape}\n")
        
        best_x, best_E = classical_simulated_annealing(
            Q0, N0, T_init=None, T_final=0.01, n_steps=10000, seed=123, adaptive_T=True
        )
        labels = decode_assignment(best_x, N0)
        
        print("Best energy:", best_E)
        print("Labels:", labels)
        
        is_feasible, violation_rate = check_feasibility(best_x, N0)
        print(f"\nFeasible: {is_feasible}, Violation rate: {violation_rate}")
        
        print(f"\n{'='*60}")
        print("Running batch solve on all events...")
        print(f"{'='*60}\n")
        
        results = batch_solve_events_sa(
            npz_file,
            n_events=n_events,
            lambda_constraint=10.0,
            w_mode="dr2",
            T_init=None,
            T_final=0.01,
            n_steps=10000,
            n_runs=10,
            seed=0,
            adaptive_T=True,
            output_csv="results_baseline_sa.csv"
        )
        
        print(f"\n{'='*60}")
        print("Summary:")
        print(f"  Total events: {len(results)}")
        print(f"  Average energy: {np.mean([r['best_energy'] for r in results]):.2f}")
        print(f"  Average solve time: {np.mean([r['solve_time'] for r in results]):.4f}s")
        print(f"  Feasible solutions: {sum([r['is_feasible'] for r in results])}/{len(results)}")
        
    except FileNotFoundError:
        print(f"Error: {npz_file} not found.")
        print("Please run generate_data.py first to generate data.")

