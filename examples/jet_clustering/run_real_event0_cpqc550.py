import json
import time
import os

import numpy as np
import kaiwu as kw
import requests
import kaiwu.cim._optimizer_adapter as oa

from real_qubo_model import build_qubo_real


SAVE_DIR = os.path.abspath("./kaiwu_runs")
os.makedirs(SAVE_DIR, exist_ok=True)
kw.common.CheckpointManager.save_dir = SAVE_DIR
print("Checkpoint save_dir =", kw.common.CheckpointManager.save_dir)

_old_put = requests.put


def put_retry(url, data=None, **kwargs):
    kwargs.pop("timeout", None)
    for attempt in range(5):
        try:
            print(f"[PUT attempt {attempt+1}/5]", url)
            return _old_put(url, data=data, timeout=(20, 300), **kwargs)
        except Exception as e:
            print("PUT failed:", repr(e))
            if attempt == 4:
                raise
            time.sleep(2 * (attempt + 1))


requests.put = put_retry
oa.requests.put = put_retry
print("Patched requests.put timeout to (20,300)s with retry")


def get_b(sol_dict: dict, idx: int) -> int:
    return int(round(sol_dict.get(f"b[{idx}]", 0.0)))


def check_constraints(sol_dict: dict, N: int) -> int:
    violations = 0
    for i in range(N):
        x0 = get_b(sol_dict, i * 2 + 0)
        x1 = get_b(sol_dict, i * 2 + 1)
        if x0 + x1 != 1:
            violations += 1
    return violations


def cluster_sizes(sol_dict: dict, N: int) -> tuple[int, int]:
    size0 = sum(get_b(sol_dict, i * 2 + 0) for i in range(N))
    size1 = sum(get_b(sol_dict, i * 2 + 1) for i in range(N))
    return int(size0), int(size1)


def repair_onehot_balanced(sol_dict: dict, N: int, target0: int) -> dict:
    fixed = dict(sol_dict)

    xs = []
    for i in range(N):
        a = f"b[{i * 2}]"
        b = f"b[{i * 2 + 1}]"
        v0 = float(fixed.get(a, 0.0))
        v1 = float(fixed.get(b, 0.0))
        x0 = int(round(v0))
        x1 = int(round(v1))
        xs.append((v0, v1, x0, x1))

    size0 = sum(x0 for (_, _, x0, x1) in xs if x0 + x1 == 1)

    for i, (v0, v1, x0, x1) in enumerate(xs):
        a = f"b[{i * 2}]"
        b = f"b[{i * 2 + 1}]"

        if x0 + x1 == 1:
            fixed[a], fixed[b] = float(x0), float(x1)
            continue

        if size0 < target0:
            fixed[a], fixed[b] = 1.0, 0.0
            size0 += 1
        else:
            fixed[a], fixed[b] = 0.0, 1.0

    return fixed


def qubo_energy_from_Qint(sol_dict: dict, Q_int: np.ndarray) -> float:
    n = Q_int.shape[0]
    x = np.zeros(n, dtype=np.int16)
    for i in range(n):
        x[i] = int(round(sol_dict.get(f"b[{i}]", 0.0)))
    return float(x @ (Q_int @ x))


def main():
    event_idx = 0
    lam = 300.0
    mu = 60.0
    target_max = 100.0

    Q = build_qubo_real(
        npz_file="events_data.npz",
        event_idx=event_idx,
        lambda_constraint=lam,
        w_mode="dr2",
        mu_balance=mu,
    )

    n_vars = max(max(i, j) for (i, j) in Q.keys()) + 1
    N = n_vars // 2
    print(f"[event {event_idx}] N={N}, n_vars={n_vars}, nnz={len(Q)}")

    if n_vars > 550:
        raise ValueError(f"n_vars={n_vars} exceeds CPQC-550 limit 550")

    max_abs_q = max(abs(v) for v in Q.values())
    print("max |Q_ij| =", max_abs_q)

    Q_mat = np.zeros((n_vars, n_vars), dtype=float)
    for (i, j), v in Q.items():
        Q_mat[i, j] = v
        if i != j:
            Q_mat[j, i] = v

    max_abs_mat = float(np.max(np.abs(Q_mat)))
    scale = target_max / max_abs_mat if max_abs_mat > 0 else 1.0
    Q_mat = Q_mat * scale
    print(
        f"Scaled QUBO: max_abs {max_abs_mat:.3f} -> "
        f"{np.max(np.abs(Q_mat)):.3f}, scale={scale:.6f}"
    )

    Q_int = np.rint(Q_mat).astype(np.int16)
    Q_int_base = Q_int.copy()
    max_int = int(np.max(np.abs(Q_int)))
    print("Quantized QUBO: max_abs_int =", max_int)
    if max_int == 0:
        raise ValueError("Quantization made all coefficients zero. Increase target_max.")

    onehot_boost = 1
    max_int_boost = max_int
    print("After onehot_boost, max_abs_int =", max_int_boost)
    if max_int_boost > 500:
        raise ValueError(
            f"Coefficients too large after boost: {max_int_boost}. "
            "Lower onehot_boost or target_max."
        )

    qubo_model = kw.qubo.qubo_matrix_to_qubo_model(Q_int)

    task_prefix = (
        f"WuYueCup_real_event{event_idx}_N{N}"
        f"_lam{int(lam)}_mu{int(mu)}_tmax{int(target_max)}_oh{onehot_boost}"
    )
    optimizer = kw.cim.CIMOptimizer(
        task_name_prefix=task_prefix,
        wait=True,
        interval=1,
    )

    solver = kw.solver.SimpleSolver(optimizer)

    t0 = time.time()
    print(">>> submitting and waiting for real machine result ...")
    sol_dict, energy = solver.solve_qubo(qubo_model)
    print(">>> AFTER solve_qubo")
    t1 = time.time()

    if sol_dict is None:
        print("No solution returned from solver.")
        out = {
            "event_idx": event_idx,
            "N": N,
            "n_vars": n_vars,
            "lambda_constraint": lam,
            "mu_balance": mu,
            "energy": None,
            "elapsed_sec": float(t1 - t0),
            "max_abs_Q": float(max_abs_q),
            "note": "No solution found by CIM solver.",
        }
    else:
        if not any(str(k).startswith("b[") for k in sol_dict.keys()):
            print(
                "Warning: solution keys are not b[i] style:",
                list(sol_dict.keys())[:5],
            )

        target0 = N // 2
        sol_fixed = repair_onehot_balanced(sol_dict, N, target0)

        viol_raw = check_constraints(sol_dict, N)
        viol = check_constraints(sol_fixed, N)
        s0, s1 = cluster_sizes(sol_fixed, N)

        print("Energy:", energy)
        E_raw_int_base = qubo_energy_from_Qint(sol_dict, Q_int_base)
        E_fix_int_base = qubo_energy_from_Qint(sol_fixed, Q_int_base)
        E_raw_int = qubo_energy_from_Qint(sol_dict, Q_int)
        E_fix_int = qubo_energy_from_Qint(sol_fixed, Q_int)
        print("QUBO energy (int, base raw):", E_raw_int_base)
        print("QUBO energy (int, base fixed):", E_fix_int_base)
        print("QUBO energy (int, boost raw):", E_raw_int)
        print("QUBO energy (int, boost fixed):", E_fix_int)
        print("Constraint violations (raw):", viol_raw)
        print("Constraint violations (fixed):", viol)
        print(f"Cluster sizes: c0={s0}, c1={s1}, target0={target0}")

        out = {
            "event_idx": event_idx,
            "N": N,
            "n_vars": n_vars,
            "lambda_constraint": lam,
            "mu_balance": mu,
            "target0": target0,
            "energy": float(energy),
            "elapsed_sec": float(t1 - t0),
            "qubo_energy_int_base_raw": E_raw_int_base,
            "qubo_energy_int_base_fixed": E_fix_int_base,
            "qubo_energy_int_boost_raw": E_raw_int,
            "qubo_energy_int_boost_fixed": E_fix_int,
            "violations_raw": int(viol_raw),
            "violations": int(viol),
            "cluster0_size": s0,
            "cluster1_size": s1,
            "max_abs_Q": float(max_abs_q),
            "solution": {str(k): float(v) for k, v in sol_dict.items()},
            "solution_fixed": {str(k): float(v) for k, v in sol_fixed.items()},
        }
    fname = (
        f"real_event{event_idx}_lam{int(lam)}_mu{int(mu)}"
        f"_tmax{int(target_max)}_oh{onehot_boost}.json"
    )
    with open(fname, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("Saved:", fname)


if __name__ == "__main__":
    main()