import numpy as np

K = 2

def build_qubo_matrix_k2(
    DR: np.ndarray,
    lambda_constraint: float = 10.0,
    w_mode: str = "dr2"
) -> np.ndarray:
    N = DR.shape[0]
    n_vars = N * K
    Q = np.zeros((n_vars, n_vars), dtype=float)

    if w_mode == "dr2":
        W = DR**2
    elif w_mode == "dr":
        W = DR
    else:
        raise ValueError("w_mode must be 'dr2' or 'dr'")

    for i in range(N):
        for j in range(i + 1, N):
            w_ij = W[i, j]
            for k in range(K):
                a = i * K + k
                b = j * K + k
                if a < b:
                    Q[a, b] += w_ij
                else:
                    Q[b, a] += w_ij

    for i in range(N):
        a = i * K + 0
        b = i * K + 1

        Q[a, a] += -lambda_constraint
        Q[b, b] += -lambda_constraint

        Q[min(a, b), max(a, b)] += 2.0 * lambda_constraint

    return Q

def qubo_energy(Q: np.ndarray, x: np.ndarray) -> float:
    x = x.astype(float)
    return float(x @ Q @ x)

def decode_assignment(x: np.ndarray, N: int) -> np.ndarray:
    X = x.reshape(N, K)
    return np.argmax(X, axis=1)

def check_feasibility(x: np.ndarray, N: int) -> tuple:
    X = x.reshape(N, K)
    row_sums = X.sum(axis=1)
    is_feasible = np.allclose(row_sums, 1.0)
    violation_rate = np.mean(np.abs(row_sums - 1.0))
    return is_feasible, violation_rate

def load_DR_from_npz(npz_file: str, event_idx: int) -> np.ndarray:
    data = np.load(npz_file)
    return data[f"event_{event_idx}_DR"]

