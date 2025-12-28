import numpy as np
import csv
import time
from typing import Tuple
from qubo_model import (
    K, build_qubo_matrix_k2, qubo_energy, decode_assignment,
    check_feasibility, load_DR_from_npz
)


def _load_eta_phi_from_npz(npz_file: str, event_idx: int) -> Tuple[np.ndarray, np.ndarray]:
    data = np.load(npz_file)
    eta = data[f"event_{event_idx}_eta"]
    phi = data[f"event_{event_idx}_phi"]
    return eta.astype(float), phi.astype(float)


def _kmeans_eta_phi(
    eta: np.ndarray,
    phi: np.ndarray,
    K_local: int = 2,
    max_iters: int = 100,
    rng: np.random.Generator | None = None
) -> Tuple[np.ndarray, float]:
    if rng is None:
        rng = np.random.default_rng()

    N = eta.shape[0]
    X = np.stack([eta, np.cos(phi), np.sin(phi)], axis=1)

    indices = rng.choice(N, size=K_local, replace=False)
    centroids = X[indices]

    labels = np.full(N, -1, dtype=int)

    for _ in range(max_iters):
        diff = X[:, None, :] - centroids[None, :, :]
        d2 = np.sum(diff ** 2, axis=2)
        new_labels = np.argmin(d2, axis=1)

        if np.array_equal(new_labels, labels):
            break

        labels = new_labels

        for k in range(K_local):
            mask = labels == k
            if np.any(mask):
                centroids[k] = X[mask].mean(axis=0)
            else:
                centroids[k] = X[rng.integers(0, N)]

    sse = float(np.sum((X - centroids[labels]) ** 2))
    return labels, sse


def solve_single_event_kmeans(
    npz_file: str,
    event_idx: int,
    lambda_constraint: float = 10.0,
    w_mode: str = "dr2",
    n_runs: int = 10,
    max_iters: int = 100,
    seed: int = 0,
    K_local: int = K
) -> dict:
    DR = load_DR_from_npz(npz_file, event_idx)
    N = DR.shape[0]

    t_start = time.time()
    Q = build_qubo_matrix_k2(DR, lambda_constraint=lambda_constraint, w_mode=w_mode)
    build_time = time.time() - t_start

    if K_local != K:
        raise ValueError(f"K mismatch between KMeans ({K_local}) and QUBO model K ({K})")

    eta, phi = _load_eta_phi_from_npz(npz_file, event_idx)

    energies = []
    sses = []
    best_x_overall = None
    best_E_overall = 1e100
    best_sse_overall = 1e100

    rng = np.random.default_rng(seed)

    t_start = time.time()
    for run in range(n_runs):
        labels, sse = _kmeans_eta_phi(eta, phi, K_local=K_local, max_iters=max_iters, rng=rng)
        x = np.zeros(N * K_local, dtype=int)
        for i in range(N):
            x[i * K_local + labels[i]] = 1

        E = qubo_energy(Q, x)
        energies.append(E)
        sses.append(sse)

        if E < best_E_overall:
            best_E_overall = E
            best_x_overall = x.copy()
        if sse < best_sse_overall:
            best_sse_overall = sse

    solve_time = time.time() - t_start

    labels_best = decode_assignment(best_x_overall, N)
    is_feasible, violation_rate = check_feasibility(best_x_overall, N)

    result = {
        "event_idx": event_idx,
        "N": N,
        "best_energy": best_E_overall,
        "mean_energy": float(np.mean(energies)),
        "std_energy": float(np.std(energies)),
        "min_energy": float(np.min(energies)),
        "best_sse": best_sse_overall,
        "mean_sse": float(np.mean(sses)),
        "std_sse": float(np.std(sses)),
        "build_time": build_time,
        "solve_time": solve_time,
        "is_feasible": is_feasible,
        "violation_rate": violation_rate,
        "labels": labels_best,
        "solver": "KMeans",
        "n_steps": max_iters,
        "n_runs": n_runs,
        "n_evals": N * max_iters * n_runs,
        "avg_time_per_run": solve_time / n_runs if n_runs > 0 else solve_time
    }

    return result


def batch_solve_events_kmeans(
    npz_file: str,
    n_events: int,
    lambda_constraint: float = 10.0,
    w_mode: str = "dr2",
    n_runs: int = 10,
    max_iters: int = 100,
    seed: int = 0,
    output_csv: str = "results_kmeans.csv"
) -> list:
    results = []

    for event_idx in range(n_events):
        print(f"Solving event {event_idx} with KMeans...", end=" ")
        result = solve_single_event_kmeans(
            npz_file,
            event_idx,
            lambda_constraint=lambda_constraint,
            w_mode=w_mode,
            n_runs=n_runs,
            max_iters=max_iters,
            seed=seed + event_idx * 1000
        )
        results.append(result)
        print(
            f"N={result['N']}, E={result['best_energy']:.2f} "
            f"(mean={result['mean_energy']:.2f}±{result['std_energy']:.2f}), "
            f"feasible={result['is_feasible']}"
        )

    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "event_idx", "N", "best_energy", "mean_energy", "std_energy", "min_energy",
            "best_sse", "mean_sse", "std_sse",
            "build_time", "solve_time", "avg_time_per_run", "is_feasible", "violation_rate",
            "solver", "n_steps", "n_runs", "n_evals"
        ])
        for r in results:
            writer.writerow([
                r["event_idx"], r["N"], r["best_energy"],
                r.get("mean_energy", r["best_energy"]),
                r.get("std_energy", 0.0),
                r.get("min_energy", r["best_energy"]),
                r.get("best_sse", 0.0),
                r.get("mean_sse", 0.0),
                r.get("std_sse", 0.0),
                r["build_time"], r["solve_time"], r.get("avg_time_per_run", r["solve_time"]),
                r["is_feasible"], r["violation_rate"], r["solver"],
                r.get("n_steps", 0), r.get("n_runs", 1), r.get("n_evals", 0)
            ])

    print(f"\nSaved KMeans results to {output_csv}")
    return results


if __name__ == "__main__":
    npz_file = "events_data.npz"

    try:
        data = np.load(npz_file)
        n_events = int(data["n_events"][0])
        print(f"Found {n_events} events in {npz_file}\n")

        results = batch_solve_events_kmeans(
            npz_file=npz_file,
            n_events=n_events,
            lambda_constraint=10.0,
            w_mode="dr2",
            n_runs=10,
            max_iters=100,
            seed=0,
            output_csv="results_kmeans.csv"
        )

        print(f"\n{'='*60}")
        print("Summary (KMeans):")
        print(f"  Total events: {len(results)}")
        print(f"  Average energy: {np.mean([r['best_energy'] for r in results]):.2f}")
        print(f"  Average solve time: {np.mean([r['solve_time'] for r in results]):.4f}s")
        print(f"  Feasible solutions: {sum([r['is_feasible'] for r in results])}/{len(results)}")

    except FileNotFoundError:
        print(f"Error: {npz_file} not found.")
        print("Please run generate_data.py first to generate data.")


