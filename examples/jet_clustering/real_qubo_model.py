import numpy as np

K = 2


def build_qubo_matrix_k2(
    DR: np.ndarray,
    lambda_constraint: float = 200.0,
    w_mode: str = "dr2",
    mu_balance: float = 30.0,
    balance_target0: int | None = None,
    balance_on_cluster: int = 0
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
            if w_ij == 0.0:
                continue
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
        if a < b:
            Q[a, b] += 2.0 * lambda_constraint
        else:
            Q[b, a] += 2.0 * lambda_constraint

    if mu_balance > 0.0:
        if balance_target0 is None:
            balance_target0 = N // 2
        t = int(balance_target0)
        c = int(balance_on_cluster)
        idxs = [i * K + c for i in range(N)]
        diag_add = mu_balance * (1 - 2 * t)
        for a in idxs:
            Q[a, a] += diag_add
        off_add = 2.0 * mu_balance
        for p in range(N):
            for q in range(p + 1, N):
                a = idxs[p]
                b = idxs[q]
                if a < b:
                    Q[a, b] += off_add
                else:
                    Q[b, a] += off_add

    return Q


def load_DR_from_npz(npz_file: str, event_idx: int) -> np.ndarray:
    data = np.load(npz_file)
    return data[f"event_{event_idx}_DR"]


def build_qubo_real(
    npz_file: str = "events_data.npz",
    event_idx: int = 0,
    lambda_constraint: float = 200.0,
    w_mode: str = "dr2",
    mu_balance: float = 30.0
) -> dict:
    DR = load_DR_from_npz(npz_file, event_idx)
    Q_mat = build_qubo_matrix_k2(
        DR,
        lambda_constraint=lambda_constraint,
        w_mode=w_mode,
        mu_balance=mu_balance,
        balance_target0=DR.shape[0] // 2,
        balance_on_cluster=0
    )
    n = Q_mat.shape[0]
    Q = {}
    for i in range(n):
        for j in range(i, n):
            v = Q_mat[i, j]
            if abs(v) > 1e-12:
                Q[(i, j)] = float(v)
    return Q


