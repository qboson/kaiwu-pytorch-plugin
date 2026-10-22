"""Compare complete local solvers on CPU; this does not time model training.

Run from the repository root: ``python benchmarks/maifs_local_search.py``.
Times are medians over three problems. Exact random states are checked before
reporting timings; close floating-point gains can differ from full subtraction.
"""

import argparse
import json
from statistics import median
from time import perf_counter

import numpy as np

from kaiwu.torch_plugin.maifs.qubo import _solve_ising_local_search


def full_scan(matrix, initial, max_iter):
    """Legacy solver using full outer-product energies for each candidate."""
    spins = np.r_[2 * initial - 1, 1].astype(int)
    weights = np.triu(matrix)
    value = float(np.sum(weights * np.outer(spins, spins)))
    for _ in range(max_iter):
        best_value, best_index = value, -1
        for index in range(len(spins)):
            candidate = spins.copy()
            candidate[index] *= -1
            candidate_value = float(np.sum(weights * np.outer(candidate, candidate)))
            if candidate_value < best_value:
                best_value, best_index = candidate_value, index
        if best_index < 0:
            break
        spins[best_index] *= -1
        value = best_value
    return spins.reshape(1, -1)


def main():
    """Measure both integer and floating-point problems without timing assertions."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", type=int, default=[64, 128, 256])
    parser.add_argument("--max-iter", type=int, default=100)
    args = parser.parse_args()
    if args.max_iter < 1 or min(args.sizes) < 1:
        parser.error("sizes and max-iter must be positive")
    for integer in (True, False):
        for size in args.sizes:
            old_times, new_times = [], []
            for seed in (111, 112, 113):
                rng = np.random.default_rng(seed)
                matrix = (rng.integers(-10, 11, (size + 1, size + 1)).astype(float)
                          if integer else rng.normal(size=(size + 1, size + 1)))
                initial = rng.integers(0, 2, size)
                start = perf_counter()
                old_state = full_scan(matrix, initial, args.max_iter)
                old_times.append(perf_counter() - start)
                start = perf_counter()
                new_state = _solve_ising_local_search(matrix, initial, args.max_iter)
                new_times.append(perf_counter() - start)
                np.testing.assert_array_equal(old_state, new_state)
            print(json.dumps({
                "features": size, "coefficients": "integer" if integer else "float",
                "full_scan_ms": median(old_times) * 1000,
                "incremental_ms": median(new_times) * 1000,
                "ratio_of_medians": median(old_times) / median(new_times),
            }))


if __name__ == "__main__":
    main()
