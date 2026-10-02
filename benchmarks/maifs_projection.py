"""CPU projection microbenchmark; no training or external solver is timed.

Run from the repository root: python benchmarks/maifs_projection.py
The reference copies the previous complete-candidate-energy implementation.
Near floating-point ties, the new gain-based decisions can differ.
"""

import argparse
import json
from statistics import median
from time import perf_counter

import numpy as np
from torch import nn

from kaiwu.torch_plugin.maifs.plugin import FeatureSelectionWrapper


def legacy_projection(wrapper, candidate, quadratic, linear):
    mask = candidate.copy()

    def objective(index, value):
        changed = mask.copy()
        changed[index] = value
        return float(0.5 * changed @ quadratic @ changed + linear @ changed), int(index)

    count = int(mask.sum())
    while count > wrapper.max_selected_features:
        index = min(np.flatnonzero(mask), key=lambda index: objective(index, 0))
        mask[index] = 0
        count -= 1
    enforce = wrapper._min_selected_features_explicit or count == 0
    while enforce and count < wrapper.min_selected_features:
        index = min(np.flatnonzero(mask == 0), key=lambda index: objective(index, 1))
        mask[index] = 1
        count += 1
    return mask


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", type=int, default=[64, 128, 256])
    args = parser.parse_args()
    if min(args.sizes) < 2:
        parser.error("sizes must be at least two")
    for size in args.sizes:
        before, after = [], []
        for seed in (111, 112, 113):
            rng = np.random.default_rng(seed)
            quadratic, linear = rng.normal(size=(size, size)), rng.normal(size=size)
            wrapper = FeatureSelectionWrapper(nn.Identity(), size, max_selected_features=size // 2)
            candidate = np.ones(size, dtype=int)
            start = perf_counter()
            expected = legacy_projection(wrapper, candidate, quadratic, linear)
            before.append((perf_counter() - start) * 1000)
            start = perf_counter()
            actual = wrapper._project_selected_feature_count(candidate, quadratic, linear)
            after.append((perf_counter() - start) * 1000)
            np.testing.assert_array_equal(actual, expected)
        print(json.dumps({"features": size, "full_energy_ms": median(before),
                          "incremental_ms": median(after), "seeds": [111, 112, 113]}))


if __name__ == "__main__":
    main()
