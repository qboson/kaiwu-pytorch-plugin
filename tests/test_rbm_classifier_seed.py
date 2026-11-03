"""Importing the classifier example must not reseed its caller's experiments."""

import importlib.util
from pathlib import Path
import random
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch


def offline_sampler(**kwargs):
    """Supply legal deterministic Ising samples without an external solver."""
    del kwargs

    def solve(matrix):
        states = np.ones((2, len(matrix)), dtype=np.float32)
        states[0, :-1] = -1
        return states

    return SimpleNamespace(solve=solve)


def load_module(name, path):
    """Execute an actual example module from its checked-out source."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(name="classifier_path")
def fixture_classifier_path(monkeypatch):
    """Import optional plotting dependencies before recording caller RNG state."""
    for dependency in ("matplotlib", "seaborn", "sklearn", "scipy"):
        pytest.importorskip(dependency)
    for name, exports in (
        ("kaiwu.classical", ("SimulatedAnnealingOptimizer",)),
        ("kaiwu.cim", ("CIMOptimizer", "PrecisionReducer")),
    ):
        boundary = ModuleType(name)
        for export in exports:
            setattr(boundary, export, offline_sampler)
        monkeypatch.setitem(sys.modules, name, boundary)
    folder = Path(__file__).resolve().parents[1] / "example/rbm_digits"
    runner = load_module("seed_rbm_digits", folder / "rbm_digits.py")
    monkeypatch.setitem(sys.modules, "rbm_digits", runner)
    # Each test restores the caller's streams, independently of the code under test.
    python_state, numpy_state = random.getstate(), np.random.get_state()
    try:
        with torch.random.fork_rng(devices=[]):
            yield folder / "rbm_classifier.py"
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def test_import_preserves_python_numpy_and_torch_streams(classifier_path):
    """Merely discovering train_classifier leaves all three CPU RNGs intact."""
    random.seed(91)
    np.random.seed(92)
    torch.manual_seed(93)
    python_state, numpy_state, torch_state = (
        random.getstate(), np.random.get_state(), torch.random.get_rng_state().clone(),
    )
    module = load_module("seed_import_classifier", classifier_path)
    assert callable(module.train_classifier)
    assert random.getstate() == python_state
    after_numpy = np.random.get_state()
    assert after_numpy[0] == numpy_state[0]
    np.testing.assert_array_equal(after_numpy[1], numpy_state[1])
    assert after_numpy[2:] == numpy_state[2:]
    assert torch.equal(torch.random.get_rng_state(), torch_state)


def test_repeated_training_uses_the_example_seed(classifier_path, monkeypatch):
    """Real RBM/SGD and both logistic fits repeat despite unrelated RNG use."""
    module = load_module("seed_train_classifier", classifier_path)
    features = np.random.default_rng(4).uniform(0, 1, size=(24, 8)).astype(np.float32)
    labels = np.tile([0, 1], 12)

    def small_dataset(self, plot_img=False):
        del self, plot_img
        return features[:16], features[16:], labels[:16], labels[16:]

    monkeypatch.setattr(module.RBMRunner, "load_data", small_dataset)
    first, first_labels, first_predictions = module.train_classifier(n_iter=1)
    first_state = {name: value.detach().clone() for name, value in first.rbm.state_dict().items()}
    first_draws = (random.random(), np.random.random(), torch.rand(4))
    random.seed(101)
    np.random.seed(102)
    torch.manual_seed(103)
    second, second_labels, second_predictions = module.train_classifier(n_iter=1)
    second_draws = (random.random(), np.random.random(), torch.rand(4))

    for name, value in second.rbm.state_dict().items():
        torch.testing.assert_close(value, first_state[name], rtol=0, atol=0)
    np.testing.assert_array_equal(first_labels, second_labels)
    np.testing.assert_array_equal(first_predictions, second_predictions)
    assert first_draws[:2] == second_draws[:2]
    torch.testing.assert_close(first_draws[2], second_draws[2], rtol=0, atol=0)
