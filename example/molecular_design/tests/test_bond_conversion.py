"""Bond conversion order, dtype and ownership contracts, without a solver."""
import itertools
import os
from pathlib import Path
import sys

import numpy as np
import pytest

SOURCE = os.environ.get("QC_MODEL_SOURCE", "sqa")
if SOURCE not in ("sqa", "cim_sa"):
    raise ValueError("QC_MODEL_SOURCE must be sqa or cim_sa")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / SOURCE))
from qc_mol_design.optimization.qubo import MoleculeQUBO


def test_exhaustive_four_atom_binary_round_trip_and_energy():
    qubo = MoleculeQUBO(["C"] * 4, [4] * 4)
    pairs = list(itertools.combinations(range(4), 2))
    coefficients = np.arange(36).reshape(6, 6)
    for bits in itertools.product((0, 1), repeat=6):
        sample = dict(enumerate(bits))
        adjacency = qubo.decode_sample(sample)
        np.testing.assert_array_equal(adjacency, adjacency.T)
        np.testing.assert_array_equal(np.diag(adjacency), np.zeros(4))
        assert adjacency.dtype == np.int32
        for pair, bit in zip(pairs, bits):
            assert adjacency[pair] == bit
        encoded = qubo.bond_vector(adjacency)
        np.testing.assert_array_equal(encoded, bits)
        assert encoded.dtype == np.float32
        expected_energy = sum(coefficients[i, j] * bits[i] * bits[j]
                              for i in range(6) for j in range(6))
        assert encoded @ coefficients @ encoded == expected_energy


@pytest.mark.parametrize("n", [0, 1, 2, 5])
def test_noncontiguous_adjacency_and_independent_float32_result(n):
    qubo = MoleculeQUBO(["C"] * n, [4] * n)
    adjacency = np.arange(4*n*n, dtype=np.float64).reshape(2*n, 2*n)[::2, ::2]
    expected = np.array([adjacency[i, j] for i, j in itertools.combinations(range(n), 2)],
                        dtype=np.float32)
    vector = qubo.bond_vector(adjacency)
    np.testing.assert_array_equal(vector, expected)
    assert vector.shape == (n*(n-1)//2,)
    assert vector.dtype == np.float32
    assert not np.shares_memory(adjacency, vector)
    vector[:] = -1
    np.testing.assert_array_equal(qubo.bond_vector(adjacency), expected)


def test_missing_extra_and_integer_convertible_sample_values():
    qubo = MoleculeQUBO(["C"] * 3, [4] * 3)
    actual = qubo.decode_sample({0: "1", 2: 1.9, 99: 1})
    np.testing.assert_array_equal(actual, [[0, 1, 0], [1, 0, 1], [0, 1, 0]])


@pytest.mark.parametrize("n", [0, 1])
def test_no_bond_variables_decode_to_empty_or_single_atom_graph(n):
    qubo = MoleculeQUBO(["C"] * n, [4] * n)
    actual = qubo.decode_sample({})
    np.testing.assert_array_equal(actual, np.zeros((n, n), dtype=np.int32))
