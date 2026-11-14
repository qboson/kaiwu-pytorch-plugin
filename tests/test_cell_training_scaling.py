"""Cell validation values must not determine Bernoulli training preprocessing."""

import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch


def unavailable_solver(*args, **kwargs):
    """Fail if data-only checks accidentally reach an external SDK solver."""
    raise AssertionError(f"Unexpected solver call: {args}, {kwargs}")


def load_example(name, path):
    """Load an actual example source file without requiring Scanpy."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(name="trainer_class")
def fixture_trainer_class(monkeypatch):
    """Isolate only optional solver imports, retaining real models and loaders."""
    pytest.importorskip("sklearn")
    pytest.importorskip("tqdm")
    folder = Path(__file__).resolve().parents[1] / "example/qvae_cell"
    for name, export in (
        ("kaiwu.classical", "SimulatedAnnealingOptimizer"),
        ("kaiwu.cim", "CIMOptimizer"),
        ("kaiwu.preprocess", "PrecisionReducer"),
    ):
        boundary = ModuleType(name)
        setattr(boundary, export, unavailable_solver)
        monkeypatch.setitem(sys.modules, name, boundary)
    models = load_example("cell_scaling_models", folder / "models.py")
    monkeypatch.setitem(sys.modules, "models", models)
    return load_example("cell_scaling_trainer", folder / "trainer.py").Trainer


def make_trainer(trainer_class, loss_type="bernoulli"):
    """Use a deterministic split whose held-out row IDs are 1 and 8."""
    return trainer_class(SimpleNamespace(
        loss_type=loss_type, val_percentage=0.2, seed=42, batch_size=3,
    ), device="cpu")


def prepare(trainer, values):
    """Exercise the same conversion/loader sequence as the CLI and notebook."""
    converted = trainer.adata_to_array(SimpleNamespace(X=values))
    return trainer.make_loaders(converted, np.arange(values.shape[0]))


def assert_loaders(loaders, expected):
    """Check real iteration, expression/ID alignment and complete split coverage."""
    row_ids = []
    for loader in loaders:
        seen = []
        for expressions, ids in loader:
            torch.testing.assert_close(expressions, torch.tensor(expected[ids.numpy()]))
            assert expressions.dtype == torch.float32
            seen.extend(ids.tolist())
        row_ids.append(sorted(seen))
    assert row_ids[1] == [1, 8]
    assert row_ids[0] == [0, 2, 3, 4, 5, 6, 7, 9]
    assert row_ids[2] == list(range(10))


@pytest.mark.parametrize("sparse", [False, True])
def test_holdout_extrema_do_not_change_training_inputs_or_bias(trainer_class, sparse):
    """Changing only validation cells leaves the actual training data identical."""
    trainer = make_trainer(trainer_class)
    original = np.arange(2, 32, dtype=np.float32).reshape(10, 3)
    reference = original.copy()
    reference[[1, 8]] = [-100, 0, 100]
    changed = reference.copy()
    changed[[1, 8]] = [-10000, 0, 10000]
    expected = np.clip((reference - 2) / 29, 0, 1)
    if sparse:
        scipy_sparse = pytest.importorskip("scipy.sparse")
        inputs = [scipy_sparse.csr_matrix(values) for values in (reference, changed)]
    else:
        inputs = [reference, changed]
    first, second = [prepare(trainer, values) for values in inputs]
    assert_loaders(first, expected)
    assert_loaders(second, expected)
    torch.testing.assert_close(first[0].dataset.tensors[0], second[0].dataset.tensors[0])
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(17)
        first_mean = trainer.compute_mean_x(first[0])
        torch.manual_seed(17)
        second_mean = trainer.compute_mean_x(second[0])
    np.testing.assert_array_equal(first_mean, second_mean)
    np.testing.assert_array_equal(reference[[0, 2, 3, 4, 5, 6, 7, 9]],
                                  original[[0, 2, 3, 4, 5, 6, 7, 9]])
    np.testing.assert_array_equal(changed[[1, 8]], [[-10000, 0, 10000]] * 2)


@pytest.mark.parametrize("constant", [-2.0, 2.0])
def test_constant_training_range_is_finite_and_holdout_is_clipped(trainer_class, constant):
    """A zero-width training range does not borrow validation extrema."""
    values = np.full((10, 2), constant, dtype=np.float32)
    values[[1, 8]] = [constant - 1, constant + 1]
    expected = np.zeros_like(values)
    expected[[1, 8]] = [0, 1]
    assert_loaders(prepare(make_trainer(trainer_class), values), expected)


def test_probability_training_range_is_not_rescaled_for_holdout(trainer_class):
    """Out-of-range validation cells cannot activate training normalization."""
    values = np.linspace(0.2, 0.8, 20, dtype=np.float32).reshape(10, 2)
    values[[1, 8]] = [-2, 5]
    assert_loaders(prepare(make_trainer(trainer_class), values), np.clip(values, 0, 1))


@pytest.mark.parametrize("loss_type", ["bernoulli", "mse"])
def test_existing_input_domains_and_source_arrays_are_preserved(trainer_class, loss_type):
    """Probability inputs stay unchanged; MSE retains arbitrary expression values."""
    values = np.linspace(0.1, 0.9, 20, dtype=np.float32).reshape(10, 2)
    if loss_type == "mse":
        values = values * 40 - 20
    original = values.copy()
    assert_loaders(prepare(make_trainer(trainer_class, loss_type), values), original)
    np.testing.assert_array_equal(values, original)
