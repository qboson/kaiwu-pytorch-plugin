import numpy as np
import csv
import time
from typing import Tuple, Dict
from qubo_model import (
    K, build_qubo_matrix_k2, qubo_energy, decode_assignment,
    check_feasibility, load_DR_from_npz
)
from kaiwu_config import init_kaiwu_license

try:
    import kaiwu as kw
    from kaiwu.classical import SimulatedAnnealingOptimizer
    KAIWU_AVAILABLE = True
except ImportError:
    KAIWU_AVAILABLE = False
    print("Warning: Kaiwu SDK is not installed.")


def qubo_to_dict(Q: np.ndarray) -> Dict[Tuple[int, int], float]:
    n = Q.shape[0]
    Q_dict = {}
    for i in range(n):
        for j in range(i, n):
            val = float(Q[i, j])
            if abs(val) > 1e-12:
                Q_dict[(i, j)] = val
    return Q_dict


def repair_onehot_k2(x: np.ndarray, N: int) -> np.ndarray:
    xr = x.copy().astype(int)
    for i in range(N):
        a = i * 2
        b = a + 1
        s = xr[a] + xr[b]
        if s == 1:
            continue
        xr[a] = 1
        xr[b] = 0
    return xr


def repair_onehot_k2_from_spin(s: np.ndarray, N: int) -> np.ndarray:
    s = np.asarray(s, dtype=float).ravel()
    x = np.zeros(2 * N, dtype=int)
    for i in range(N):
        a = 2 * i
        b = a + 1
        if s[a] <= s[b]:
            x[a] = 1
        else:
            x[b] = 1
    return x


def local_improve_k2(Q: np.ndarray, x: np.ndarray, N: int, n_pass: int = 2) -> np.ndarray:
    x = x.copy().astype(int)
    for _ in range(n_pass):
        improved = False
        for i in range(N):
            a = 2 * i
            b = a + 1
            if x[a] == 1:
                x_try = x.copy()
                x_try[a], x_try[b] = 0, 1
            else:
                x_try = x.copy()
                x_try[a], x_try[b] = 1, 0
            if qubo_energy(Q, x_try) < qubo_energy(Q, x):
                x = x_try
                improved = True
        if not improved:
            break
    return x


def solve_qubo_with_kaiwu_sa(
    Q: np.ndarray,
    num_reads: int = 100,
    sweeps: int = 2000,
    seed: int = 0
) -> Tuple[np.ndarray, float]:
    if not KAIWU_AVAILABLE:
        raise ImportError("Kaiwu SDK is not installed.")
    
    init_kaiwu_license()
    
    n_vars = Q.shape[0]
    best_x = None
    best_E = float('inf')
    
    np.random.seed(seed)
    
    try:
        iters_per_t = max(1, sweeps // 1000) if sweeps > 0 else 10
        optimizer = SimulatedAnnealingOptimizer(
            initial_temperature=100.0,
            alpha=0.99,
            cutoff_temperature=0.001,
            iterations_per_t=iters_per_t,
            size_limit=num_reads
        )
        solutions = optimizer.solve(Q)
        solutions = np.array(solutions)
        if solutions.ndim == 1:
            solutions = solutions.reshape(1, -1)
        for s in solutions:
            s = np.array(s, dtype=float).ravel()
            if s.size != n_vars:
                continue
            N = n_vars // K
            x = repair_onehot_k2_from_spin(s, N)
            x = local_improve_k2(Q, x, N, n_pass=2)
            E = qubo_energy(Q, x)
            if E < best_E:
                best_E = E
                best_x = x.copy()
    except Exception as e:
        print(f"Warning: error when using Kaiwu SDK SA, falling back to local SA: {e}")
        for read in range(num_reads):
            x = np.random.randint(0, 2, size=n_vars, dtype=int)
            E_current = qubo_energy(Q, x)
            T_init = 1.0
            T_final = 0.01
            cooling_rate = (T_final / T_init) ** (1.0 / sweeps)
            T = T_init
            for sweep in range(sweeps):
                i = np.random.randint(0, n_vars)
                x_new = x.copy()
                x_new[i] = 1 - x_new[i]
                E_new = qubo_energy(Q, x_new)
                delta_E = E_new - E_current
                if delta_E < 0 or np.random.random() < np.exp(-delta_E / T):
                    x = x_new
                    E_current = E_new
                    if E_current < best_E:
                        best_E = E_current
                        best_x = x.copy()
                T *= cooling_rate
    
    if best_x is None:
        x = np.random.randint(0, 2, size=n_vars, dtype=int)
        best_x = x
        best_E = qubo_energy(Q, best_x)
    
    return best_x, best_E


def solve_single_event_kaiwu_sa(
    npz_file: str,
    event_idx: int,
    lambda_constraint: float = 10.0,
    w_mode: str = "dr2",
    num_reads: int = 10,
    sweeps: int = 10000,
    seed: int = 0
) -> dict:
    DR = load_DR_from_npz(npz_file, event_idx)
    N = DR.shape[0]
    
    t_start = time.time()
    Q = build_qubo_matrix_k2(DR, lambda_constraint=lambda_constraint, w_mode=w_mode)
    build_time = time.time() - t_start
    
    energies = []
    best_x_overall = None
    best_E_overall = 1e100
    
    np.random.seed(seed)
    t_start = time.time()
    
    for run in range(num_reads):
        try:
            best_x, best_E = solve_qubo_with_kaiwu_sa(Q, num_reads=1, sweeps=sweeps, seed=seed + run)
            energies.append(best_E)
            if best_E < best_E_overall:
                best_E_overall = best_E
                best_x_overall = best_x.copy()
        except Exception as e:
            print(f"警告: Event {event_idx}, Run {run} 失败: {e}")
            continue
    
    solve_time = time.time() - t_start
    
    if best_x_overall is None:
        best_x_overall = np.zeros(N * K, dtype=int)
        for i in range(N):
            best_x_overall[i * K + np.random.randint(0, K)] = 1
        best_E_overall = qubo_energy(Q, best_x_overall)
    
    labels = decode_assignment(best_x_overall, N)
    is_feasible, violation_rate = check_feasibility(best_x_overall, N)
    
    result = {
        "event_idx": event_idx,
        "N": N,
        "best_energy": best_E_overall,
        "mean_energy": np.mean(energies) if energies else best_E_overall,
        "std_energy": np.std(energies) if len(energies) > 1 else 0.0,
        "min_energy": np.min(energies) if energies else best_E_overall,
        "build_time": build_time,
        "solve_time": solve_time,
        "is_feasible": is_feasible,
        "violation_rate": violation_rate,
        "labels": labels,
        "solver": "Kaiwu_SA",
        "n_steps": sweeps,
        "n_runs": num_reads,
        "n_evals": sweeps * num_reads,
        "avg_time_per_run": solve_time / num_reads if num_reads > 0 else solve_time
    }
    
    return result


def batch_solve_events_kaiwu_sa(
    npz_file: str,
    n_events: int,
    lambda_constraint: float = 10.0,
    w_mode: str = "dr2",
    num_reads: int = 10,
    sweeps: int = 10000,
    seed: int = 0,
    output_csv: str = "results_kaiwu_sa.csv"
) -> list:
    if not KAIWU_AVAILABLE:
        raise ImportError("Kaiwu SDK is not installed.")
    
    results = []
    
    for event_idx in range(n_events):
        print(f"Solving event {event_idx} with Kaiwu SA...", end=" ")
        try:
            result = solve_single_event_kaiwu_sa(
                npz_file, event_idx,
                lambda_constraint=lambda_constraint,
                w_mode=w_mode,
                num_reads=num_reads,
                sweeps=sweeps,
                seed=seed + event_idx * 1000
            )
            results.append(result)
            if num_reads > 1:
                print(f"N={result['N']}, E={result['best_energy']:.2f} (mean={result['mean_energy']:.2f}±{result['std_energy']:.2f}), feasible={result['is_feasible']}")
            else:
                print(f"N={result['N']}, E={result['best_energy']:.2f}, feasible={result['is_feasible']}")
        except Exception as e:
            print(f"Failed: {e}")
            continue
    
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
        
        if not KAIWU_AVAILABLE:
            print("Error: Kaiwu SDK is not installed.")
            print("Please install Kaiwu SDK first, for example: pip install kaiwu-1.3.0-cp310-none-any.whl")
            exit(1)
        
        init_kaiwu_license()
        
        DR0 = load_DR_from_npz(npz_file, 0)
        N0 = DR0.shape[0]
        Q0 = build_qubo_matrix_k2(DR0, lambda_constraint=10.0, w_mode="dr2")
        
        print(f"Testing event 0: N={N0}")
        print(f"QUBO matrix shape: {Q0.shape}\n")
        
        best_x, best_E = solve_qubo_with_kaiwu_sa(Q0, num_reads=5, sweeps=1000, seed=123)
        labels = decode_assignment(best_x, N0)
        
        print("Best energy:", best_E)
        print("Labels:", labels)
        
        is_feasible, violation_rate = check_feasibility(best_x, N0)
        print(f"\nFeasible: {is_feasible}, Violation rate: {violation_rate}")
        
        print(f"\n{'='*60}")
        print("Running batch solve on all events with Kaiwu SA...")
        print(f"{'='*60}\n")
        
        results = batch_solve_events_kaiwu_sa(
            npz_file,
            n_events=n_events,
            lambda_constraint=10.0,
            w_mode="dr2",
            num_reads=10,
            sweeps=10000,
            seed=0,
            output_csv="results_kaiwu_sa.csv"
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
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

