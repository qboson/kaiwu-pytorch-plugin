"""Actual DBN estimator configuration, cloning, refitting and model search."""

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

from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline


class ExhaustiveSampler:
    """Return legal Ising states at the solver boundary without credentials."""

    def __init__(self, **kwargs):
        self.calls = 0

    def solve(self, matrix):
        self.calls += 1
        spins = np.array(list(product((-1.0, 1.0), repeat=len(matrix) - 1)))
        states = np.column_stack((spins, np.ones(len(spins))))
        energies = -np.einsum("bi,ij,bj->b", states, matrix, states)
        return states[np.argsort(energies)[:2]]


@pytest.fixture
def examples(monkeypatch):
    example_dir = Path(__file__).resolve().parents[1] / "example" / "dbn_digits"
    monkeypatch.syspath_prepend(str(example_dir))
    import dbn_trainer

    assert Path(dbn_trainer.__file__).resolve() == example_dir / "dbn_trainer.py"
    monkeypatch.setattr(dbn_trainer, "SimulatedAnnealingOptimizer", ExhaustiveSampler)
    spec = importlib.util.spec_from_file_location(
        "dbn_parameters_test", example_dir / "supervised_dbn_digits.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return dbn_trainer.DBNPretrainer, module.SupervisedDBNClassification


@pytest.fixture
def data():
    features = np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.],
                         [40., 40.], [40., 41.], [41., 40.], [41., 41.]])
    return features, np.array([7] * 4 + [42] * 4)


def pretrainer_parameters():
    return dict(hidden_layers_structure=[2], learning_rate_rbm=.037, n_epochs_rbm=1,
                batch_size=2, verbose=False, shuffle=False, drop_last=False,
                plot_img=True, random_state=7, use_cim=False)


def supervised_parameters(fine_tuning):
    return dict(hidden_layers_structure=[2], learning_rate_rbm=.037, n_epochs_rbm=1,
                batch_size=2, verbose=False, plot_img=True, random_state=7,
                fine_tuning=fine_tuning, learning_rate=.023, n_iter_backprop=1,
                l2_regularization=.005, activation_function="relu", dropout_p=.25,
                use_cim=False, classifier_type="svm", clf_C=3., clf_iter=17)


def assert_configuration(model, expected):
    assert model.get_params(deep=False) == expected
    assert model.get_params() == expected
    for name, value in expected.items():
        assert getattr(model, name) == value


def test_pretrainer_exposes_all_constructor_parameters(examples):
    pretrainer_class, _ = examples
    parameters = pretrainer_parameters()
    model = pretrainer_class(**parameters)
    assert_configuration(model, parameters)
    assert model.hidden_layers_structure is parameters["hidden_layers_structure"]


def test_pretrainer_parent_initialization_supports_subclass_parameters(examples, data):
    pretrainer_class, _ = examples
    features, _ = data

    class NamedPretrainer(pretrainer_class):
        def __init__(self, name="default"):
            super().__init__(hidden_layers_structure=[2], n_epochs_rbm=1, verbose=False)
            self.name = name

    model = NamedPretrainer(name="custom")
    model.fit(features)
    copied = clone(model)
    assert copied.name == "custom"
    assert copied._dbn.rbm_layers is None


@pytest.mark.parametrize("fine_tuning", [False, True])
def test_supervised_exposes_all_constructor_parameters(examples, fine_tuning):
    _, supervised_class = examples
    parameters = supervised_parameters(fine_tuning)
    model = supervised_class(**parameters)
    assert_configuration(model, parameters)
    assert model.hidden_layers_structure is parameters["hidden_layers_structure"]


@pytest.mark.parametrize("estimator", ["pretrainer", "classifier", "fine-tuning"])
def test_clone_keeps_configuration_without_learned_data(examples, data, estimator):
    pretrainer_class, supervised_class = examples
    features, labels = data
    if estimator == "pretrainer":
        parameters = pretrainer_parameters()
        model = pretrainer_class(**parameters)
        model.fit(features)
        original_dbn = model._dbn
    else:
        parameters = supervised_parameters(estimator == "fine-tuning")
        model = supervised_class(**parameters)
        model.fit(features, labels)
        original_dbn = model.unsupervised_dbn._dbn

    trainer = model._trainer if estimator == "pretrainer" else model.unsupervised_dbn._trainer
    assert trainer.sampler.calls == 8 // parameters["batch_size"]
    copied = clone(model)

    assert_configuration(copied, parameters)
    assert copied.hidden_layers_structure is not model.hidden_layers_structure
    copied_dbn = copied._dbn if estimator == "pretrainer" else copied.unsupervised_dbn._dbn
    assert copied_dbn is not original_dbn
    assert copied_dbn.rbm_layers is None
    assert not copied_dbn._is_trained
    if estimator != "pretrainer":
        assert not hasattr(copied, "classes_")
        assert not hasattr(copied.label_encoder, "classes_")
        assert copied.classifier is None
        assert copied.fine_tune_network is None


def test_pretrainer_sampler_route_follows_updated_use_cim(examples, data, monkeypatch):
    pretrainer_class, _ = examples
    features, _ = data
    import dbn_trainer

    class CIMSampler(ExhaustiveSampler):
        pass

    monkeypatch.setattr(dbn_trainer, "CIMOptimizer", CIMSampler)
    monkeypatch.setattr(dbn_trainer, "PrecisionReducer", lambda sampler, **kwargs: sampler)
    checkpoint = dbn_trainer.kw.common.CheckpointManager
    monkeypatch.setattr(checkpoint, "save_dir", checkpoint.save_dir)
    model = pretrainer_class(hidden_layers_structure=[2], n_epochs_rbm=1, verbose=False)
    model.fit(features)
    original_sampler = model._trainer.sampler
    model.set_params(use_cim=True).fit(features)
    assert isinstance(model._trainer.sampler, CIMSampler)
    assert model._trainer.sampler.calls == 1
    copied = clone(model)
    assert copied.get_params()["use_cim"] is True
    assert isinstance(copied._trainer.sampler, CIMSampler)
    assert copied._dbn.rbm_layers is None
    model.set_params(use_cim=False).fit(features)
    assert type(model._trainer.sampler) is ExhaustiveSampler
    assert model._trainer.sampler is not original_sampler
    assert model._trainer.sampler.calls == 1


def test_pretrainer_set_params_refit_matches_fresh_configuration(examples, data):
    pretrainer_class, _ = examples
    features, _ = data
    model = pretrainer_class(hidden_layers_structure=[2], n_epochs_rbm=1, verbose=False)
    model.fit(features)
    old_rbm = model.get_rbm_layer(0)
    parameters = pretrainer_parameters()
    parameters.update(hidden_layers_structure=[3], n_epochs_rbm=2,
                      batch_size=3, drop_last=True, random_state=13)
    assert model.set_params(**parameters) is model
    torch.manual_seed(13)
    model.fit(features)

    fresh = pretrainer_class(**parameters)
    torch.manual_seed(13)
    fresh.fit(features)

    assert model.get_rbm_layer(0) is not old_rbm
    assert model.transform(features).shape == (8, 3)
    assert model._trainer.sampler.calls == 1 + 2 * (8 // 3)
    np.testing.assert_allclose(model.transform(features), fresh.transform(features), atol=0, rtol=0)
    torch.testing.assert_close(model.get_rbm_layer(0).quadratic_coef,
                               fresh.get_rbm_layer(0).quadratic_coef, atol=0, rtol=0)


@pytest.mark.parametrize("fine_tuning", [False, True])
def test_supervised_set_params_and_batching_reach_actual_fit(examples, data, fine_tuning):
    _, supervised_class = examples
    features, labels = data
    model = supervised_class(hidden_layers_structure=[2], n_epochs_rbm=1,
                             n_iter_backprop=1, verbose=False, fine_tuning=not fine_tuning)
    model.fit(features, labels)
    old_rbm = model.unsupervised_dbn.get_rbm_layer(0)
    previous_calls = model.unsupervised_dbn._trainer.sampler.calls
    parameters = supervised_parameters(fine_tuning)
    parameters.update(hidden_layers_structure=[3], n_epochs_rbm=2,
                      batch_size=2, random_state=13, dropout_p=0.)
    assert model.set_params(**parameters) is model
    torch.manual_seed(13)
    model.fit(features, labels)

    fresh = supervised_class(**parameters)
    torch.manual_seed(13)
    fresh.fit(features, labels)

    assert model.unsupervised_dbn.get_rbm_layer(0) is not old_rbm
    assert model.transform(features).shape == (8, 3)
    assert model.unsupervised_dbn._trainer.sampler.calls == previous_calls + 2 * (8 // 2)
    torch.testing.assert_close(model.unsupervised_dbn.get_rbm_layer(0).quadratic_coef,
                               fresh.unsupervised_dbn.get_rbm_layer(0).quadratic_coef,
                               atol=0, rtol=0)
    np.testing.assert_allclose(model.predict_proba(features), fresh.predict_proba(features),
                               atol=0, rtol=0)


def test_pre_train_false_reuses_prepared_rbm_after_classifier_parameter_update(examples, data):
    _, supervised_class = examples
    features, labels = data
    model = supervised_class(hidden_layers_structure=[2], n_epochs_rbm=1,
                             fine_tuning=False, verbose=False)
    model.pre_train(features)
    prepared_rbm = model.unsupervised_dbn.get_rbm_layer(0)
    weights = prepared_rbm.quadratic_coef.detach().clone()
    calls = model.unsupervised_dbn._trainer.sampler.calls
    model.set_params(clf_C=3.).fit(features, labels, pre_train=False)
    assert model.unsupervised_dbn.get_rbm_layer(0) is prepared_rbm
    assert model.unsupervised_dbn._trainer.sampler.calls == calls
    torch.testing.assert_close(prepared_rbm.quadratic_coef, weights, atol=0, rtol=0)
    assert model.classifier.C == 3.


def test_pretrainer_pipeline_grid_search_runs_actual_candidates(examples, data):
    pretrainer_class, _ = examples
    features, labels = data
    pipeline = Pipeline([
        ("dbn", pretrainer_class(hidden_layers_structure=[2], n_epochs_rbm=1, verbose=False)),
        ("classifier", LogisticRegression()),
    ])
    search = GridSearchCV(pipeline, {"dbn__hidden_layers_structure": [[1], [3]]},
                          cv=StratifiedKFold(2), scoring="accuracy", error_score="raise")
    search.fit(features, labels)
    assert len(search.cv_results_["params"]) == 2
    assert np.isfinite(search.cv_results_["mean_test_score"]).all()
    selected = search.best_estimator_.named_steps["dbn"]
    assert selected.hidden_layers_structure == search.best_params_["dbn__hidden_layers_structure"]
    assert selected.transform(features).shape[1] == selected.hidden_layers_structure[-1]


@pytest.mark.parametrize("fine_tuning", [False, True])
def test_supervised_grid_search_preserves_configuration_and_trains(examples, data, fine_tuning):
    _, supervised_class = examples
    features, labels = data
    parameters = supervised_parameters(fine_tuning)
    parameters.update(dropout_p=0., plot_img=False, classifier_type="logistic")
    model = supervised_class(**parameters)
    parameter = "learning_rate" if fine_tuning else "clf_C"
    grid = {parameter: [.01, .2] if fine_tuning else [.5, 3.]}
    search = GridSearchCV(model, grid, cv=StratifiedKFold(2), scoring="accuracy",
                          error_score="raise").fit(features, labels)
    expected = dict(parameters, **search.best_params_)
    assert_configuration(search.best_estimator_, expected)
    assert len(search.cv_results_["params"]) == 2
    assert np.isfinite(search.cv_results_["mean_test_score"]).all()
    predictions = search.best_estimator_.predict(features)
    assert set(predictions) <= set(labels)
    assert search.best_estimator_.score(features, labels) == np.mean(predictions == labels)
