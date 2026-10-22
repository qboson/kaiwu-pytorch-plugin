"""Pinned SDK CPU precision tests for a valid all-zero QUBO."""

import itertools
from pathlib import Path

import kaiwu
import numpy as np
import pytest
import torch
from torch import nn

from kaiwu.torch_plugin.maifs import FeatureSelectionWrapper, qubo


def assert_restores_original_spins(explorer, original_size):
    """Repeat each original spin across its replicas, independently of restoration."""
    counts = np.diff(np.r_[-1, explorer.plan.last_var_idx])
    for original in itertools.product((-1, 1), repeat=original_size):
        replicas = np.repeat(original, counts)
        np.testing.assert_array_equal(explorer.restore_solution(replicas), original)


def test_real_sdk_zero_split_and_restore_are_supported():
    """Only the SDK's precision calculation is degenerate for zero coefficients."""
    adjust, calculate = qubo._get_kaiwu_precision_helpers()
    assert adjust is kaiwu.ising.adjust_ising_matrix_precision
    assert calculate is kaiwu.ising.calculate_ising_matrix_bit_width
    assert qubo._get_kaiwu_split_helper() is kaiwu.preprocess.perform_precision_adaption_split
    split, mapping = kaiwu.preprocess.perform_precision_adaption_split(
        np.zeros((3, 3)), param_bit=14, min_increment=1,
        penalty=None, round_to_increment=True,
    )
    np.testing.assert_array_equal(split, np.zeros((3, 3)))
    np.testing.assert_array_equal(mapping, np.arange(3))
    for spins in itertools.product((-1, 1), repeat=3):
        np.testing.assert_array_equal(
            kaiwu.preprocess.restore_split_solution(np.asarray(spins), mapping), spins
        )


@pytest.mark.parametrize("dtype", [np.int32, np.float32, np.float64])
@pytest.mark.parametrize("precision", [4, 8, 14])
def test_zero_adjustment_preserves_coefficients_and_has_finite_metadata(dtype, precision):
    """Zero needs one signed bit and unit scaling, without aliasing the input."""
    original = np.zeros((3, 3), dtype=dtype)
    adjusted, metadata = qubo.PrecisionSplitExplorer._adjust_to_precision(original, precision)
    np.testing.assert_array_equal(adjusted, np.zeros((3, 3), dtype=int))
    assert np.issubdtype(adjusted.dtype, np.integer)
    assert metadata == {"precision": 1, "multiplier": 1.0}
    assert not np.shares_memory(adjusted, original)
    adjusted[0, 1] = 1
    np.testing.assert_array_equal(original, np.zeros((3, 3)))


@pytest.mark.parametrize("precision", [4, 8, 14])
def test_zero_search_uses_real_split_and_restores_all_original_spin_vectors(precision):
    """Public precision search retains a flat energy landscape at every source width."""
    explorer = qubo.PrecisionSplitExplorer(
        target_precision=precision, max_bits=16, max_precision=precision + 4
    )
    plan = explorer.search(np.zeros((3, 3)))
    assert plan.source_precision == precision + 4
    assert plan.split_size == 3
    assert plan.precision_info == {"precision": 1, "multiplier": 1.0}
    np.testing.assert_array_equal(plan.adjusted_matrix, np.zeros((3, 3)))
    np.testing.assert_array_equal(plan.split_matrix, np.zeros((3, 3)))
    np.testing.assert_array_equal(plan.last_var_idx, np.arange(3))
    assert_restores_original_spins(explorer, 3)
    for spins in itertools.product((-1, 1), repeat=3):
        spins = np.asarray(spins)
        assert spins @ plan.split_matrix @ spins == 0


@pytest.mark.parametrize("kind", ["integer", "decimal", "irrational"])
def test_nonzero_adjustment_and_real_split_restoration_keep_existing_behavior(kind):
    """Verify both finite SDK scaling and the lossy fallback against a numeric oracle."""
    matrices = {
        "integer": np.array([[0., 2., -3.], [2., 0., 1.], [-3., 1., 0.]]),
        "decimal": np.array([[0., .22, .198], [.22, 0., .197], [.198, .197, 0.]]),
        "irrational": np.array([
            [0., np.sqrt(2), np.pi], [np.sqrt(2), 0., np.e], [np.pi, np.e, 0.]
        ]),
    }
    original = matrices[kind]
    adjusted, metadata = qubo.PrecisionSplitExplorer._adjust_to_precision(original, 8)
    if kind == "integer":
        expected = original.astype(int)
        assert metadata == {"precision": 3, "multiplier": 1.0}
    else:
        expected = np.rint(original * 127 / np.abs(original).max()).astype(int)
        assert metadata == {"precision": float("inf"), "multiplier": float("inf")}
    np.testing.assert_array_equal(adjusted, expected)
    explorer = qubo.PrecisionSplitExplorer(target_precision=4, max_bits=32, max_precision=8)
    plan = explorer.search(original)
    assert plan.source_precision == 8
    np.testing.assert_array_equal(plan.adjusted_matrix, expected)
    assert_restores_original_spins(explorer, 3)


@pytest.fixture
def offline_cim(monkeypatch, tmp_path):
    """Enumerate valid spins without constructing an SDK hardware optimizer."""
    calls = []
    # Every public call uses a new disposable base and retains records: no deletion.
    monkeypatch.setattr(kaiwu.common.CheckpointManager, "save_dir", None)

    class EnumeratedCIM:
        def __init__(self, **kwargs):
            path = Path(kaiwu.common.CheckpointManager.save_dir).resolve()
            assert path.is_relative_to(tmp_path.resolve())

        def solve(self, matrix):
            np.testing.assert_array_equal(matrix, np.zeros((3, 3), dtype=int))
            assert np.issubdtype(matrix.dtype, np.integer)
            # Stable tie order selects a negative-gauge solution decoding to [1, 0].
            spins = [(-1, 1, -1)]
            spins += [x for x in itertools.product((-1, 1), repeat=3) if x != spins[0]]
            spins = np.asarray(spins)
            energies = np.einsum("bi,ij,bj->b", spins, matrix, spins)
            calls.append(matrix.copy())
            return spins[np.argsort(energies, kind="stable")]

    monkeypatch.setattr(kaiwu.cim, "CIMOptimizer", EnumeratedCIM)
    return {"save_dir": tmp_path / "checkpoints", "cleanup_records": False}, calls


def test_public_zero_qubo_reaches_offline_optimizer_and_decodes_valid_optimum(offline_cim):
    """A flat QUBO remains solvable through real precision search and public decoding."""
    options, calls = offline_cim
    selected = qubo.solve_qubo(
        np.zeros((2, 2)), np.zeros(2), np.ones(2, dtype=int),
        solver="kaiwu_cim", **options,
    )
    np.testing.assert_array_equal(selected, [1, 0])
    assert len(calls) == 1
    assert 0.5 * selected @ np.zeros((2, 2)) @ selected + np.zeros(2) @ selected == 0


def test_real_zero_linear_model_can_update_its_feature_mask(offline_cim):
    """An ordinary differentiable model produces a legitimate zero feature-selection QUBO."""
    options, calls = offline_cim
    model = nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        model.weight.zero_()
    selector = FeatureSelectionWrapper(
        model, feature_dim=2, solver="kaiwu_cim", solver_kwargs=options
    )
    inputs = torch.tensor([[1., 0.], [0., 1.], [1., 1.]])
    batches = [(inputs, torch.zeros(3, 1))]
    gradient, hessian = selector.compute_mask_derivatives(batches, nn.MSELoss())
    np.testing.assert_array_equal(gradient, np.zeros(2))
    np.testing.assert_array_equal(hessian, np.zeros((2, 2)))
    selected = selector.update_mask(batches, nn.MSELoss())
    np.testing.assert_array_equal(selected, [1, 0])
    np.testing.assert_array_equal(selector.get_support(), [True, False])
    assert len(calls) == 1
    torch.testing.assert_close(selector(inputs), torch.zeros(3, 1))
