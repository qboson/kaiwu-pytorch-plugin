"""Tiny real MLP training, without optional plotting/FID/GIF dependencies."""
import importlib.util
from pathlib import Path
import sys
import types

import numpy as np
import pytest
import torch

pytest.importorskip("sklearn")
pytest.importorskip("tqdm")


@pytest.fixture(scope="module", autouse=True)
def single_thread_training():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def classifier_module(monkeypatch):
    helpers = types.ModuleType("utils.helpers")
    helpers.plot_training_curves = lambda **kwargs: None
    monkeypatch.setitem(sys.modules, "utils", types.ModuleType("utils"))
    monkeypatch.setitem(sys.modules, "utils.helpers", helpers)
    path = Path(__file__).resolve().parents[1] / "example/qvae_mnist/downstream/classifier.py"
    spec = importlib.util.spec_from_file_location("mlp_label_example", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "tqdm", lambda values, **kwargs: values)
    return module


def make_classifier(module, output_dim=2):
    model = module.MLPClassifier(
        input_dim=4, hidden_dims=[], output_dim=output_dim,
        epochs_mlp=1, batch_size_mlp=8, device="cpu", random_state=7,
    )
    original = model._create_model
    def initialize():
        network = original()
        # Controlled initialization avoids the separate zero-best-state bug #227.
        with torch.no_grad():
            for parameter in network.parameters():
                parameter.zero_()
        return network
    model._create_model = initialize
    return model


def data(labels):
    labels = np.asarray(labels)
    encoded = np.tile(np.arange(len(labels)), 20)
    x = np.random.default_rng(17).normal(size=(len(encoded), 4)).astype(np.float32)
    x[:, 0] = 2 * encoded - (len(labels) - 1)
    return x, labels[encoded]


@pytest.mark.parametrize("labels", [
    [0, 1], [2, 7], [-3, 5], [10, 20], ["cat", "dog"],
    [False, True], [2.0, 7.0], ["zebra", "apple"], [7, 2],
])
@pytest.mark.parametrize("output_dim", [2, None, np.int64(2)])
def test_real_training_and_probability_column_labels(classifier_module, labels, output_dim):
    x, y = data(labels)
    saved_x, saved_y = x.copy(), y.copy()
    model = make_classifier(classifier_module, output_dim)
    assert model.fit(x, y) is model
    np.testing.assert_array_equal(model.classes_, np.unique(y))
    assert model.output_dim_ == 2
    assert model.output_dim == output_dim
    predicted, probability = model.predict(x), model.predict_proba(x)
    assert probability.shape == (len(x), 2)
    assert predicted.dtype == model.classes_.dtype
    assert np.isin(predicted, model.classes_).all()
    np.testing.assert_array_equal(model.classes_[probability.argmax(1)], predicted)
    np.testing.assert_allclose(probability.sum(1), 1., rtol=1e-6, atol=1e-6)
    assert model.score(x, y) == np.mean(predicted == y)
    np.testing.assert_array_equal(x, saved_x)
    np.testing.assert_array_equal(y, saved_y)


@pytest.mark.parametrize("labels", [[2, 7], [-3, 5], ["cat", "dog"]])
def test_encoded_reference_has_identical_training(classifier_module, labels):
    x, y = data(labels)
    classes, encoded = np.unique(y, return_inverse=True)
    model = make_classifier(classifier_module).fit(x, y)
    reference = make_classifier(classifier_module).fit(x, encoded)
    np.testing.assert_array_equal(model.predict_proba(x), reference.predict_proba(x))
    np.testing.assert_array_equal(model.predict(x), classes[reference.predict(x)])
    for key, value in model.model.state_dict().items():
        torch.testing.assert_close(value, reference.model.state_dict()[key], rtol=0, atol=0)


@pytest.mark.parametrize("output_dim", [None, 2])
def test_refit_replaces_class_mapping(classifier_module, output_dim):
    x, y = data([2, 7])
    model = make_classifier(classifier_module, output_dim).fit(x, y)
    old_network = model.model
    x, y = data([10, 20])
    model.fit(x, y)
    np.testing.assert_array_equal(model.classes_, [10, 20])
    assert model.model is not old_network
    assert np.isin(model.predict(x), model.classes_).all()


def test_inferred_output_width_can_change_on_refit(classifier_module):
    model = make_classifier(classifier_module, None)
    for labels in [[2, 7], [2, 7, 11]]:
        x, y = data(labels)
        model.fit(x, y)
        assert model.output_dim_ == len(labels)
        assert model.predict_proba(x).shape[1] == len(labels)
        assert np.isin(model.predict(x), model.classes_).all()
    assert model.output_dim is None


@pytest.mark.parametrize("dimension", [0, 1, 3, 10, -1, 2.0, True, np.bool_(True)])
def test_bad_dimension_fails_before_training(classifier_module, monkeypatch, dimension):
    x, y = data([2, 7])
    model = make_classifier(classifier_module, dimension)
    calls = []
    monkeypatch.setattr(model, "_create_model", lambda: calls.append(True))
    with pytest.raises(ValueError, match="output_dim"):
        model.fit(x, y)
    assert calls == []
    assert model.classes_ is None
    assert model.model is None


def test_invalid_refit_preserves_previous_fitted_model(classifier_module):
    x, y = data([2, 7])
    model = make_classifier(classifier_module).fit(x, y)
    network, classes = model.model, model.classes_.copy()
    old_prediction = model.predict(x)
    bad_x, bad_y = data([1, 2, 3])
    with pytest.raises(ValueError, match="number of classes"):
        model.fit(bad_x, bad_y)
    assert model.model is network
    np.testing.assert_array_equal(model.classes_, classes)
    np.testing.assert_array_equal(model.predict(x), old_prediction)


def test_default_ten_class_mnist_layout_still_supported(classifier_module):
    x, y = data(range(10))
    model = make_classifier(classifier_module, 10).fit(x, y)
    np.testing.assert_array_equal(model.classes_, np.arange(10))
    assert model.predict_proba(x).shape == (200, 10)


@pytest.mark.parametrize("target", [np.linspace(0, 1, 40), np.full(40, np.nan)])
def test_nonclassification_targets_rejected(classifier_module, target):
    model = make_classifier(classifier_module, None)
    with pytest.raises(ValueError):
        model.fit(np.zeros((40, 4)), target)
    assert model.model is None
