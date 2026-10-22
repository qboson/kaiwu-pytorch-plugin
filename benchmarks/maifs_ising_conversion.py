"""Ordered QUBO-to-Ising conversion CPU microbenchmark, no solver or QPU."""
import json
from pathlib import Path
import statistics
import time

import kaiwu
import numpy as np

kaiwu.__path__.insert(0, str(Path(__file__).resolve().parents[1] / "src/kaiwu"))
from kaiwu.torch_plugin.maifs.qubo import QuadraticLinearSolver  # pylint: disable=wrong-import-position


def legacy(matrix):
    """Original scalar conversion, including its original addition order."""
    n = matrix.shape[0]
    result = np.zeros((n + 1, n + 1))
    for row in range(n):
        result[row, n] += 0.5 * float(matrix[row, row])
        for col in range(row + 1, n):
            pair = 0.25 * float(matrix[row, col])
            result[row, col] += pair
            result[row, n] += pair
            result[col, n] += pair
    return result


def measure(operation, matrix):
    """Warm up and return the median of five full conversions."""
    operation(matrix)
    samples = []
    for _ in range(5):
        start = time.perf_counter()
        operation(matrix)
        samples.append(time.perf_counter() - start)
    return statistics.median(samples) * 1000


results = []
for size in (64, 256, 512, 1024):
    original_times = []
    changed_times = []
    for seed in (7, 17, 27):
        matrix = np.random.default_rng(seed).normal(size=(size, size))
        np.testing.assert_array_equal(
            legacy(matrix), QuadraticLinearSolver.qubo_matrix_to_ising_matrix(matrix),
        )
        original_times.append(measure(legacy, matrix))
        changed_times.append(measure(QuadraticLinearSolver.qubo_matrix_to_ising_matrix, matrix))
    results.append({"size": size, "legacy_ms": statistics.median(original_times),
                    "ordered_ms": statistics.median(changed_times),
                    "float64_square_scratch_bytes": size * size * 8})
print(json.dumps(results, indent=2))
