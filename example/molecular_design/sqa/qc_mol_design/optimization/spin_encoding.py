"""Energy-preserving signed permutations of the physical Ising spins."""
import numpy as np


def encode_prepared(prepared, mode="none", seed=0):
    if mode not in ("none", "gauge", "permute", "gauge_permute"):
        raise ValueError("Unknown spin encoding")
    n = len(prepared["ising"])
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(n) if "permute" in mode else np.arange(n)
    gauge = rng.choice([-1, 1], n) if "gauge" in mode else np.ones(n, dtype=int)
    return {**prepared, "spin_permutation": permutation, "spin_gauge": gauge,
            "metadata": {**prepared["metadata"], "spin_encoding": mode, "spin_encoding_seed": int(seed)}}


def physical_matrix(logical, prepared):
    n = len(logical)
    permutation = prepared.get("spin_permutation", np.arange(n))
    gauge = prepared.get("spin_gauge", np.ones(n, dtype=int))
    return np.asarray(logical)[np.ix_(permutation, permutation)] * np.outer(gauge, gauge)


def logical_spins(physical, prepared):
    physical = np.atleast_2d(physical)
    n = physical.shape[1]
    permutation = prepared.get("spin_permutation", np.arange(n))
    gauge = prepared.get("spin_gauge", np.ones(n, dtype=int))
    logical = np.empty_like(physical)
    logical[:, permutation] = physical * gauge
    return logical
