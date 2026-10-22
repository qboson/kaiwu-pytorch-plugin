"""Pin the local-search QUBO solver to the global optimum on a small instance."""

import itertools

import numpy as np

from kaiwu.torch_plugin.maifs.qubo import QuadraticLinearSolver, solve_qubo

QUBO = np.array(
    [
        [0.1257302210933933, -0.1321048632913019, 0.6404226504432821],
        [0.0, 0.10490011715303971, -0.535669373161111],
        [0.0, 0.0, 0.36159505490948474],
    ]
)
LINEAR = np.array([1.3040000451301372, 0.9470809631292422, -0.7037352358069926])


def _ising_energy(ising: np.ndarray, binary: np.ndarray) -> float:
    spins = np.r_[2 * np.asarray(binary) - 1, 1]
    return float(np.sum(np.triu(ising) * np.outer(spins, spins)))


def test_local_search_reaches_the_global_optimum() -> None:
    ising = QuadraticLinearSolver().solve(QUBO, LINEAR)
    energies = [
        _ising_energy(ising, np.array(state))
        for state in itertools.product([0, 1], repeat=QUBO.shape[0])
    ]
    best = min(energies)

    solution = solve_qubo(
        QUBO, LINEAR, np.zeros(QUBO.shape[0], dtype=int), "local_search", max_iter=100
    )

    assert set(np.unique(solution)).issubset({0, 1})
    assert abs(_ising_energy(ising, solution) - best) < 1e-9
