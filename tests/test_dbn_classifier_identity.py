"""Classification contracts of the actual supervised DBN example."""

import importlib.util
from itertools import product
from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("sklearn")
pytest.importorskip("scipy")
pytest.importorskip("matplotlib")
pytest.importorskip("seaborn")

from sklearn.base import is_classifier
from sklearn.metrics import get_scorer


class ExhaustiveSampler:
    """Supply valid Ising solutions without hardware or SDK authentication."""

    def __init__(self):
        self.calls = 0

    def solve(self, matrix):
        self.calls += 1
        spins = np.array(list(product((-1.0, 1.0), repeat=len(matrix) - 1)))
        states = np.column_stack((spins, np.ones(len(spins))))
        energies = -np.einsum("bi,ij,bj->b", states, matrix, states)
        return states[np.argsort(energies)[:2]]


@pytest.fixture
def classifier_class(monkeypatch):
    example_dir = Path(__file__).resolve().parents[1] / "example" / "dbn_digits"
    monkeypatch.syspath_prepend(str(example_dir))
    spec = importlib.util.spec_from_file_location(
        "supervised_dbn_classifier_test", example_dir / "supervised_dbn_digits.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert Path(module.__file__).resolve() == example_dir / "supervised_dbn_digits.py"
    return module.SupervisedDBNClassification


@pytest.fixture(params=[False, True], ids=["downstream-classifier", "fine-tuning"])
def fine_tuning(request):
    return request.param


@pytest.fixture(params=[("cold", "warm"), (7, 42)], ids=["strings", "noncontiguous"])
def labels(request):
    return np.repeat(request.param, 4)


@pytest.fixture
def fitted_classifier(classifier_class, fine_tuning, labels):
    torch.manual_seed(7)
    features = np.array(
        [[0, 0], [0, 1], [1, 0], [1, 1],
         [40, 40], [40, 41], [41, 40], [41, 41]],
        dtype=np.float64,
    )
    model = classifier_class(
        hidden_layers_structure=[2],
        fine_tuning=fine_tuning,
        n_epochs_rbm=1,
        n_iter_backprop=1,
        verbose=False,
        random_state=7,
    )
    sampler = ExhaustiveSampler()
    model.unsupervised_dbn._trainer.sampler = sampler
    assert model.fit(features, labels) is model
    assert sampler.calls == 1
    assert model.unsupervised_dbn._dbn._is_trained
    rbm = model.unsupervised_dbn.get_rbm_layer(0)
    assert torch.isfinite(rbm.quadratic_coef.grad).all()
    return model, features, labels


def test_supervised_dbn_has_classifier_identity(fitted_classifier):
    model, _, _ = fitted_classifier
    assert is_classifier(model)


def test_probability_scorer_matches_observed_label_log_probability(fitted_classifier):
    model, features, labels = fitted_classifier
    probabilities = model.predict_proba(features)
    label_columns = np.array([np.flatnonzero(model.classes_ == label)[0] for label in labels])
    expected = np.log(probabilities[np.arange(len(labels)), label_columns]).mean()

    actual = get_scorer("neg_log_loss")(model, features, labels)

    assert np.isfinite(actual)
    assert actual == pytest.approx(expected, rel=1e-6, abs=1e-7)


def test_original_labels_and_accuracy_remain_valid(fitted_classifier):
    model, features, labels = fitted_classifier
    predictions = model.predict(features)
    probabilities = model.predict_proba(features)
    np.testing.assert_array_equal(model.classes_, np.unique(labels))
    assert set(predictions) <= set(labels)
    assert probabilities.shape == (len(labels), len(model.classes_))
    assert np.isfinite(probabilities).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1, rtol=1e-6)
    assert model.score(features, labels) == np.mean(predictions == labels)
