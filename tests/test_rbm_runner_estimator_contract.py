"""Real RBMRunner cloning, Pipeline search, and current-parameter refitting."""
from itertools import product
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import numpy as np
import pytest
import torch
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline


class LegalSpinAdapter:
    """Supplied legal states at the external solver boundary, not a licensed solve."""

    def __init__(self, **kwargs):
        self.options = kwargs
        self.matrices = []

    def solve(self, matrix):
        assert np.isfinite(matrix).all()
        self.matrices.append(matrix.copy())
        return np.array([list(spins) + [1]
                         for spins in product((-1, 1), repeat=len(matrix) - 1)])


@pytest.fixture
def example(monkeypatch):
    folder = Path(__file__).resolve().parents[1] / 'example/rbm_digits'
    monkeypatch.syspath_prepend(str(folder))
    previous = sys.modules.pop('rbm_digits', None)
    try:
        import rbm_digits as module
        import kaiwu.torch_plugin.restricted_boltzmann_machine as core
        assert Path(module.__file__).resolve() == folder / 'rbm_digits.py'
        assert Path(core.__file__).resolve().is_relative_to(folder.parents[1] / 'src')
        factories = []

        class OfflineSA(LegalSpinAdapter):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                factories.append(self)

        class OfflineCIM(LegalSpinAdapter):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                factories.append(self)

        class OfflinePrecisionReducer:
            def __init__(self, sampler, **kwargs):
                self.sampler = sampler
                self.options = kwargs

            def solve(self, matrix):
                return self.sampler.solve(matrix)

        monkeypatch.setattr(module, 'SimulatedAnnealingOptimizer', OfflineSA)
        monkeypatch.setattr(module, 'CIMOptimizer', OfflineCIM)
        monkeypatch.setattr(module, 'PrecisionReducer', OfflinePrecisionReducer)
        # Restore the process-local SDK global even when the existing CIM setup
        # changes it during fit. Offline adapters never create checkpoint files.
        checkpoint = module.kw.common.CheckpointManager
        monkeypatch.setattr(checkpoint, 'save_dir', checkpoint.save_dir)
        yield module, factories, OfflineSA, OfflineCIM, OfflinePrecisionReducer
    finally:
        sys.modules.pop('rbm_digits', None)
        if previous is not None:
            sys.modules['rbm_digits'] = previous


def _data():
    return np.tile([[.05, .1, .15], [.85, .9, .95]], (6, 1)), np.tile([0, 1], 6)


def _state(runner):
    return {name: value.detach().clone() for name, value in runner.rbm.state_dict().items()}


@pytest.mark.parametrize('use_cim', [False, True])
def test_constructor_parameters_clone_without_creating_a_backend(example, use_cim):
    module, factories, _, _, _ = example
    options = dict(n_components=3, learning_rate=.03, batch_size=4, n_iter=2,
                   verbose=False, plot_img=False, random_state=13, use_cim=use_cim)
    before_directory = module.kw.common.CheckpointManager.save_dir
    torch.manual_seed(7)
    before_rng = torch.random.get_rng_state().clone()

    runner = module.RBMRunner(**options)
    params = runner.get_params()
    copied = clone(runner)

    assert params == copied.get_params() == options
    assert runner.rbm is copied.rbm is None
    assert runner.sampler is copied.sampler is None
    assert not factories
    assert module.kw.common.CheckpointManager.save_dir == before_directory
    assert torch.equal(before_rng, torch.random.get_rng_state())


def test_clone_of_trained_runner_drops_model_and_backend_state(example):
    module, factories, _, _, _ = example
    X, y = _data()
    runner = module.RBMRunner(n_components=2, batch_size=4, n_iter=2, random_state=13)
    runner.fit(X, y)
    count = len(factories)

    copied = clone(runner)

    assert copied.get_params() == runner.get_params()
    assert copied.rbm is copied.sampler is None
    assert len(factories) == count
    assert runner.rbm is not None and len(runner.sampler.matrices) == 6


def test_set_params_selects_fresh_backend_and_model_on_each_fit(example):
    module, factories, sa_class, cim_class, precision_class = example
    X, y = _data()
    runner = module.RBMRunner(n_components=2, batch_size=4, n_iter=1, random_state=13)
    assert runner.fit(X, y) is runner
    first_model, first_sampler = runner.rbm, runner.sampler
    assert isinstance(first_sampler, sa_class)
    assert first_sampler.options['rand_seed'] == 13

    runner.set_params(use_cim=True, n_components=3, n_iter=2, batch_size=5, random_state=17)
    assert runner.fit(X, y) is runner

    assert runner.rbm is not first_model
    assert runner.rbm.num_visible == 3 and runner.rbm.num_hidden == 3
    assert isinstance(runner.sampler, precision_class)
    assert isinstance(runner.sampler.sampler, cim_class)
    assert runner.sampler.options == dict(precision=8, truncated_precision=10,
                                        target_bits=550, only_feasible_solution=False)
    assert runner.sampler.sampler.options == dict(task_name='test_kpp', wait=True)
    assert len(runner.sampler.sampler.matrices) == 6
    assert all(matrix.shape == (7, 7) for matrix in runner.sampler.sampler.matrices)
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               for parameter in runner.rbm.parameters())

    runner.set_params(use_cim=False, random_state=19, n_iter=1)
    runner.fit(X, y)
    assert isinstance(runner.sampler, sa_class) and runner.sampler is not first_sampler
    assert runner.sampler.options['rand_seed'] == 19
    assert len(runner.sampler.matrices) == 3 and len(factories) == 3


@pytest.mark.parametrize('use_cim', [False, True])
def test_current_integer_seed_reproduces_real_refitting_with_controlled_external_spins(
        example, use_cim):
    module, factories, _, _, _ = example
    X, y = _data()
    options = dict(n_components=2, batch_size=4, n_iter=2, random_state=13, use_cim=use_cim)
    runner = module.RBMRunner(**options)
    runner.fit(X, y)
    first = _state(runner)
    first_features = runner.transform(X)
    first_matrices = [matrix.copy() for matrix in factories[-1].matrices]
    torch.rand(41)

    runner.fit(X, y)

    for name, value in _state(runner).items():
        torch.testing.assert_close(value, first[name], rtol=0, atol=0)
    np.testing.assert_array_equal(runner.transform(X), first_features)
    for expected, actual in zip(first_matrices, factories[-1].matrices):
        np.testing.assert_array_equal(expected, actual)

    runner.set_params(random_state=19)
    runner.fit(X, y)
    changed = _state(runner)
    changed_features = runner.transform(X)
    assert any(not torch.equal(value, first[name]) for name, value in changed.items())
    fresh = module.RBMRunner(**{**options, 'random_state': 19})
    fresh.fit(X, y)
    for name, value in _state(fresh).items():
        torch.testing.assert_close(value, changed[name], rtol=0, atol=0)
    np.testing.assert_array_equal(fresh.transform(X), changed_features)
    if not use_cim:
        assert [sampler.options['rand_seed'] for sampler in factories] == [13, 13, 19, 19]


def test_none_seed_uses_callers_torch_rng_and_omits_sa_seed_option(example):
    module, factories, _, _, _ = example
    X, y = _data()
    torch.manual_seed(5)
    first = module.RBMRunner(n_components=2, batch_size=4, n_iter=2).fit(X, y)
    first_state = _state(first)
    torch.manual_seed(999)
    second = module.RBMRunner(n_components=2, batch_size=4, n_iter=2).fit(X, y)

    assert any(not torch.equal(value, first_state[name])
               for name, value in _state(second).items())
    assert all('rand_seed' not in sampler.options for sampler in factories)


def test_real_pipeline_grid_search_cv_and_refit_use_current_parameters(example):
    module, factories, _, _, _ = example
    X, y = _data()
    pipeline = Pipeline([
        ('rbm', module.RBMRunner(batch_size=4, n_iter=2, random_state=13)),
        ('logistic', LogisticRegression(max_iter=100, random_state=7)),
    ])
    search = GridSearchCV(pipeline, {'rbm__n_components': [2, 3],
                                    'rbm__use_cim': [False, True]},
                          cv=StratifiedKFold(2, shuffle=True, random_state=7),
                          n_jobs=1, error_score='raise')

    search.fit(X, y)

    assert len(search.cv_results_['params']) == 4
    assert np.isfinite(search.cv_results_['mean_test_score']).all()
    fitted = search.best_estimator_.named_steps['rbm']
    assert fitted.n_components == search.best_params_['rbm__n_components']
    assert fitted.use_cim == search.best_params_['rbm__use_cim']
    assert fitted.rbm.num_hidden == fitted.n_components
    assert search.predict(X).shape == y.shape
    assert 0 <= search.score(X, y) <= 1
    assert len(factories) == 9
    assert all(len(sampler.matrices) == (6 if sampler is factories[-1] else 4)
               for sampler in factories)


@pytest.mark.parametrize('seed', [None, 0, 13])
def test_actual_sdk_sa_constructor_receives_current_seed_and_executes_offline_boundary(
        example, monkeypatch, seed):
    module, _, _, _, _ = example
    from kaiwu.classical import SimulatedAnnealingOptimizer

    class OfflineActualSA(SimulatedAnnealingOptimizer):
        """Keep the actual SDK constructor and substitute only external solve."""

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.constructor_options = kwargs
            self.offline = LegalSpinAdapter()

        def solve(self, matrix):
            return self.offline.solve(matrix)

    monkeypatch.setattr(module, 'SimulatedAnnealingOptimizer', OfflineActualSA)
    X, y = _data()
    runner = module.RBMRunner(n_components=2, batch_size=4, n_iter=1, random_state=seed)

    runner.fit(X, y)

    assert isinstance(runner.sampler, SimulatedAnnealingOptimizer)
    if seed is None:
        assert 'rand_seed' not in runner.sampler.constructor_options
        assert isinstance(runner.sampler._rand_seed, int)
    else:
        assert runner.sampler.constructor_options['rand_seed'] == seed
        assert runner.sampler._rand_seed == seed
    assert runner.sampler._alpha == .999 and runner.sampler._size_limit == 100
    assert len(runner.sampler.offline.matrices) == 3
    assert np.isfinite(runner.transform(X)).all()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               for parameter in runner.rbm.parameters())
