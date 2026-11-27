"""CPU microbenchmark of Ising conversion, excluding SDK sampling and training.

Run from the repository root: ``python benchmarks/gbrbm_ising.py``.
Temporary byte counts describe tensors, not measured peak process memory.
"""

import argparse
import json
from statistics import median
from time import perf_counter

import numpy as np
import torch

from kaiwu.torch_plugin.gbrbm import GaussianBernoulliRestrictedBoltzmannMachine


def dense_reference(model):
    """Reference conversion before Gaussian precision was applied by row scaling."""
    linear = model.quadratic_coef.T @ (model.mu / model.var) + model.linear_bias
    quadratic = model.quadratic_coef.T @ torch.diag(1 / model.var) @ model.quadratic_coef
    matrix = torch.zeros(model.num_bernoulli + 1, model.num_bernoulli + 1)
    matrix[:-1, :-1] = quadratic / 8
    matrix[:-1, -1] = linear / 4 + quadratic.sum(dim=0) / 8
    matrix[-1, :-1] = matrix[:-1, -1]
    return matrix.numpy()


def elapsed_ms(function, repeats):
    """Return the median of repeated complete conversion calls."""
    times = []
    for _ in range(repeats):
        start = perf_counter()
        function()
        times.append((perf_counter() - start) * 1000)
    return median(times)


def main():
    """Report timings and analytical temporary tensor sizes on one CPU thread."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gaussian-sizes", nargs="+", type=int, default=[1024, 4096])
    parser.add_argument("--bernoulli-size", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if args.repeats < 1 or args.bernoulli_size < 1 or min(args.gaussian_sizes) < 1:
        parser.error("sizes and repeats must be positive")
    torch.set_num_threads(1)
    torch.manual_seed(27)
    with torch.no_grad():
        for size in args.gaussian_sizes:
            model = GaussianBernoulliRestrictedBoltzmannMachine(
                size, args.bernoulli_size, device="cpu",
            )
            reference = lambda: dense_reference(model)
            np.testing.assert_allclose(
                model.get_ising_matrix(), reference(), rtol=1e-5, atol=1e-6,
            )
            print(json.dumps({
                "gaussian": size, "bernoulli": args.bernoulli_size,
                "dense_ms": elapsed_ms(reference, args.repeats),
                "row_scaled_ms": elapsed_ms(model.get_ising_matrix, args.repeats),
                "dense_diagonal_bytes": size * size * model.mu.element_size(),
                "row_scaled_bytes": size * args.bernoulli_size * model.mu.element_size(),
            }))


if __name__ == "__main__":
    main()
