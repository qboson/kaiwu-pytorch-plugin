import numpy as np
import csv
import time
from typing import Tuple
from qubo_model import (
    K, build_qubo_matrix_k2, qubo_energy, decode_assignment,
    check_feasibility, load_DR_from_npz
)

def random_feasible_search(
    Q: np.ndarray,
    N: int,
    n_samples: int = 50000,
    seed: int = 0
) -> Tuple[np.ndarray, float]:
    rng = np.random.default_rng(seed)
    best_x = None
    best_E = 1e100

    for _ in range(n_samples):
        labels = rng.integers(0, K, size=N)
        x = np.zeros(N * K, dtype=int)
        for i in range(N):
            x[i * K + labels[i]] = 1

        E = qubo_energy(Q, x)
        if E < best_E:
            best_E = E
            best_x = x.copy()

    return best_x, best_E

def solve_single_event_random(
    npz_file: str,
    event_idx: int,
    lambda_constraint: float = 10.0,
    w_mode: str = "dr2",
    n_samples: int = 50000,
    seed: int = 0
) -> dict:
    DR = load_DR_from_npz(npz_file, event_idx)
    N = DR.shape[0]
    
    t_start = time.time()
    Q = build_qubo_matrix_k2(DR, lambda_constraint=lambda_constraint, w_mode=w_mode)
    build_time = time.time() - t_start
    
    t_start = time.time()
    best_x, best_E = random_feasible_search(Q, N, n_samples=n_samples, seed=seed)
    solve_time = time.time() - t_start
    
    labels = decode_assignment(best_x, N)
    is_feasible, violation_rate = check_feasibility(best_x, N)
    
    return {
        "event_idx": event_idx,
        "N": N,
        "best_energy": best_E,
        "build_time": build_time,
        "solve_time": solve_time,
        "is_feasible": is_feasible,
        "violation_rate": violation_rate,
        "labels": labels,
        "solver": "Random",
        "n_evals": n_samples,
        "avg_time_per_run": solve_time
    }

def batch_solve_events_random(
    npz_file: str,
    n_events: int,
    lambda_constraint: float = 10.0,
    w_mode: str = "dr2",
    n_samples: int = 50000,
    seed: int = 0,
    output_csv: str = "results_baseline_random.csv"
) -> list:
    results = []
    
    for event_idx in range(n_events):
        print(f"Solving event {event_idx}...", end=" ")
        result = solve_single_event_random(
            npz_file, event_idx,
            lambda_constraint=lambda_constraint,
            w_mode=w_mode,
            n_samples=n_samples,
            seed=seed + event_idx
        )
        results.append(result)
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
                r.get("best_energy", 0.0), 0.0, r.get("best_energy", 0.0),
                r["build_time"], r["solve_time"], r.get("avg_time_per_run", r["solve_time"]),
                r["is_feasible"], r["violation_rate"], r["solver"],
                0, 1, r.get("n_evals", 0)
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
        print(f"QUBO matrix shape: {Q0.shape}")
        print(f"Upper-triangular check: {np.allclose(np.tril(Q0, -1), 0.0)}\n")
        
        best_x, best_E = random_feasible_search(Q0, N0, n_samples=50000, seed=123)
        labels = decode_assignment(best_x, N0)
        
        print("Best energy:", best_E)
        print("Labels:", labels)
        print("Feasible check per particle sums:", best_x.reshape(N0, K).sum(axis=1)[:10])
        
        is_feasible, violation_rate = check_feasibility(best_x, N0)
        print(f"\nFeasible: {is_feasible}, Violation rate: {violation_rate}")
        
        print(f"\n{'='*60}")
        print("Running batch solve on all events...")
        print(f"{'='*60}\n")
        
        results = batch_solve_events_random(
            npz_file,
            n_events=n_events,
            lambda_constraint=10.0,
            w_mode="dr2",
            n_samples=50000,
            seed=0,
            output_csv="results_baseline_random.csv"
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

