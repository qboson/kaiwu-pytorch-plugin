"""Public MNIST sklearn pipeline configuration, cloning, refitting and search."""
from itertools import product
from pathlib import Path
import sys

import numpy as np
import pytest
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.base import clone, is_classifier
from sklearn.metrics import get_scorer
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline


class OfflineSampler:
    """Return legal Ising states solely at the external solver boundary."""
    instances = []

    def __init__(self, **kwargs):
        self.calls = 0
        self.instances.append(self)

    def solve(self, matrix):
        self.calls += 1
        spins = np.array(list(product((-1., 1.), repeat=len(matrix) - 1)))
        states = np.column_stack((spins, np.ones(len(spins))))
        return states[np.argsort(-np.einsum('bi,ij,bj->b', states, matrix, states))[:2]]


@pytest.fixture
def example(monkeypatch):
    folder = Path(__file__).resolve().parents[1] / 'example/qvae_mnist'
    monkeypatch.syspath_prepend(str(folder))
    roots = {'model', 'trainer', 'utils', 'downstream'}
    saved = {name: module for name, module in sys.modules.items()
             if name.split('.')[0] in roots}
    for name in saved:
        del sys.modules[name]
    try:
        import model.model as model_module
        import downstream.classifier as classifier_module
        import downstream.pipeline as pipeline_module
        import trainer.trainer as trainer_module
        import trainer.model_tuner as tuner_module
        import kaiwu.torch_plugin.qvae as qvae_module
        from model import Config
        for module, path in [(model_module, 'model/model.py'),
                             (classifier_module, 'downstream/classifier.py'),
                             (pipeline_module, 'downstream/pipeline.py'),
                             (trainer_module, 'trainer/trainer.py'),
                             (tuner_module, 'trainer/model_tuner.py')]:
            assert Path(module.__file__).resolve() == folder / path
        assert Path(qvae_module.__file__).resolve().is_relative_to(folder.parents[1] / 'src')
        monkeypatch.setattr(model_module, 'SimulatedAnnealingOptimizer', OfflineSampler)
        monkeypatch.setattr(OfflineSampler, 'instances', [])
        monkeypatch.setattr(plt, 'show', lambda: None)
        actual_plot = classifier_module.plot_training_curves
        def plot_curves(**kwargs):
            actual_plot(**kwargs)
            plt.close('all')
        monkeypatch.setattr(classifier_module, 'plot_training_curves', plot_curves)
        actual_reconstruction = trainer_module.Trainer._save_reconstruction
        def save_reconstruction(trainer, *args):
            actual_reconstruction(trainer, *args)
            plt.close('all')
        monkeypatch.setattr(trainer_module.Trainer, '_save_reconstruction', save_reconstruction)
        yield Config, classifier_module.MLPClassifier, pipeline_module
    finally:
        for name in list(sys.modules):
            if name.split('.')[0] in roots:
                del sys.modules[name]
        sys.modules.update(saved)
        plt.close('all')


def data(count=25, width=784):
    labels = np.arange(count) % 2
    if width == 784:
        pixels = np.repeat((.15 + .7 * labels)[:, None], width, axis=1)
        pixels += np.linspace(0, .01, width)[None, :]
    else:
        pixels = np.full((count, width), .5)
    return pixels, labels


def config_for(config_class, output, latent=2):
    return config_class(num_epochs=10, output_dir=str(output), batch_size=32,
                        encoder_hidden_nodes=[3], decoder_hidden_nodes=[3],
                        num_latent_units=latent, dist_beta=.5, kl_beta=.01,
                        classifier_kwargs={'hidden_dims': [], 'output_dim': 2,
                                           'lr_mlp': .02, 'batch_size_mlp': 8, 'epochs_mlp': 2})


def classifier_parameters(output, device=None):
    return dict(input_dim=None, hidden_dims=[], output_dim=2, weight_decay=.003,
                lr_mlp=.02, batch_size_mlp=8, epochs_mlp=2, device=device,
                save_path=str(output), random_state=42)


@pytest.mark.parametrize('device', [None, 'cpu', torch.device('cpu')])
def test_constructor_parameters_keep_identity_and_survive_actual_fit(example, tmp_path, device):
    _, classifier_class, _ = example
    parameters = classifier_parameters(tmp_path, device)
    model = classifier_class(**parameters)
    assert model.get_params(deep=False) == parameters
    assert model.device is parameters['device']
    assert model.hidden_dims is parameters['hidden_dims']
    model.fit(*data(width=2))
    assert model.get_params(deep=False) == parameters
    assert model.input_dim is None
    assert model.input_dim_ == 2
    assert model.device_ == torch.device('cpu')
    copied = clone(model)
    assert copied.get_params(deep=False) == parameters
    assert copied.model is None
    assert not hasattr(copied, 'input_dim_')
    assert not hasattr(copied, 'device_')


def test_transformer_config_identity_and_replacement_are_public_parameters(example, tmp_path):
    config_class, _, pipeline_module = example
    config = config_for(config_class, tmp_path)
    transformer = pipeline_module.PipelineTransformer(config)
    assert transformer.get_params() == {'config': config}
    assert transformer.config is config
    copied = clone(transformer)
    assert copied.config is not config
    assert copied.config.num_latent_units == config.num_latent_units
    assert copied.trainer is copied.model is copied.extractor is None
    replacement = config_for(config_class, tmp_path, latent=3)
    assert transformer.set_params(config=replacement) is transformer
    assert transformer.config is replacement


def test_public_pipeline_clone_discards_actual_learned_state(example, tmp_path):
    config_class, _, pipeline_module = example
    config = config_for(config_class, tmp_path)
    pipeline = pipeline_module.get_full_pipeline(config)
    assert isinstance(pipeline, Pipeline)
    assert pipeline.get_params()['qvae__config'] is config
    pipeline.fit(*data())
    trainer = pipeline.named_steps['qvae'].trainer
    assert len(trainer.train_losses) == len(trainer.test_losses) == 10
    assert len(trainer.train_loader.dataset) == 20
    assert len(trainer.test_loader.dataset) == 5
    assert len(trainer.tuner._vae_optimiser.state) == 8
    assert len(trainer.tuner._bm_optimiser.state) == 2
    assert trainer.model.sampler.calls == 30
    copied = clone(pipeline)
    assert copied.named_steps['qvae'].trainer is None
    assert copied.named_steps['qvae'].model is None
    assert copied.named_steps['classifier'].model is None
    assert copied.named_steps['classifier'].classes_ is None
    assert copied.named_steps['classifier'].input_dim is None
    assert copied.named_steps['classifier'].device is None
    assert copied.get_params()['classifier__lr_mlp'] == .02
    assert not hasattr(copied.named_steps['classifier'], 'input_dim_')
    assert (tmp_path / 'model_final_QVAE.pt').is_file()
    assert (tmp_path / 'qvae_training_curve.png').is_file()
    assert (tmp_path / 'mlp_training_curves_epochs_2.png').is_file()


def test_actual_probability_scoring_uses_classifier_identity(example, tmp_path):
    _, classifier_class, _ = example
    model = classifier_class(**classifier_parameters(tmp_path))
    features, labels = data(width=2)
    model.fit(features, labels)
    probabilities = model.predict_proba(features)
    expected = np.log(probabilities[np.arange(len(labels)), labels]).mean()
    score = get_scorer('neg_log_loss')(model, features, labels)
    assert is_classifier(model)
    assert score == pytest.approx(expected, abs=1e-6)


def test_current_parameters_and_new_feature_width_reach_real_refit(example, tmp_path, monkeypatch):
    _, classifier_class, _ = example
    model = classifier_class(**classifier_parameters(tmp_path))
    model.fit(*data(width=2))
    previous_model = model.model
    optimisers = []
    adam = torch.optim.Adam
    def record_adam(*args, **kwargs):
        optimiser = adam(*args, **kwargs)
        optimisers.append(optimiser)
        return optimiser
    monkeypatch.setattr(torch.optim, 'Adam', record_adam)
    updated = dict(hidden_dims=[3], lr_mlp=.031, batch_size_mlp=5, epochs_mlp=3,
                   weight_decay=.007, device='cpu')
    assert model.set_params(**updated) is model
    model.fit(*data(width=3))
    assert model.model is not previous_model
    assert model.input_dim is None
    assert model.input_dim_ == 3
    assert model.model[0].in_features == 3
    assert model.model[0].out_features == 3
    assert model.lr == updated['lr_mlp']
    assert model.batch_size == updated['batch_size_mlp']
    assert model.epochs == updated['epochs_mlp']
    optimiser = optimisers[-1]
    assert optimiser.param_groups[0]['lr'] == .031
    assert optimiser.param_groups[0]['weight_decay'] == .007
    assert all(state['step'].item() == 3 * (20 // 5) for state in optimiser.state.values())


def test_legacy_alias_updates_keep_public_constructor_parameters_in_sync(example, tmp_path):
    _, classifier_class, _ = example
    model = classifier_class(**classifier_parameters(tmp_path))
    model.lr, model.batch_size, model.epochs = .041, 5, 3
    assert model.get_params()['lr_mlp'] == .041
    assert model.get_params()['batch_size_mlp'] == 5
    assert model.get_params()['epochs_mlp'] == 3
    model.set_params(lr_mlp=.021, batch_size_mlp=8, epochs_mlp=2)
    assert (model.lr, model.batch_size, model.epochs) == (.021, 8, 2)
    model.fit(*data(width=2))
    assert (tmp_path / 'mlp_training_curves_epochs_2.png').is_file()


def test_public_pipeline_config_replacement_refits_current_latent_width(example, tmp_path):
    config_class, _, pipeline_module = example
    pipeline = pipeline_module.get_full_pipeline(config_for(config_class, tmp_path / 'first'))
    features, labels = data()
    pipeline.fit(features, labels)
    old_trainer = pipeline.named_steps['qvae'].trainer
    old_classifier_model = pipeline.named_steps['classifier'].model
    replacement = config_for(config_class, tmp_path / 'refit', latent=3)
    pipeline.set_params(qvae__config=replacement, classifier__lr_mlp=.03,
                        classifier__batch_size_mlp=5, classifier__epochs_mlp=1)
    assert pipeline.get_params()['qvae__config'] is replacement
    pipeline.fit(features, labels)
    transformer = pipeline.named_steps['qvae']
    classifier = pipeline.named_steps['classifier']
    assert transformer.config is replacement
    assert transformer.trainer is not old_trainer
    assert transformer.trainer.model is transformer.model
    assert transformer.model is not old_trainer.model
    assert transformer.trainer.config is replacement
    assert transformer.transform(features).shape == (25, 3)
    assert classifier.model is not old_classifier_model
    assert classifier.input_dim is None and classifier.input_dim_ == 3
    assert classifier.model[0].in_features == 3
    assert pipeline.predict_proba(features).shape == (25, 2)


def test_public_grid_search_trains_and_selects_actual_pipeline_candidates(example, tmp_path):
    config_class, _, pipeline_module = example
    pipeline = pipeline_module.get_full_pipeline(config_for(config_class, tmp_path))
    features, labels = data(50)
    search = GridSearchCV(pipeline, {'classifier__lr_mlp': [.01, .03]},
                          cv=StratifiedKFold(2), scoring='neg_log_loss',
                          error_score='raise').fit(features, labels)
    assert np.isfinite(search.cv_results_['mean_test_score']).all()
    assert len(search.cv_results_['params']) == 2
    selected = search.best_estimator_
    assert selected.get_params()['classifier__lr_mlp'] == search.best_params_['classifier__lr_mlp']
    assert selected.named_steps['classifier'].input_dim is None
    assert selected.named_steps['classifier'].input_dim_ == 2
    assert is_classifier(selected)
    assert selected.predict(features).shape == labels.shape
    assert len(OfflineSampler.instances) == 5
    assert [sampler.calls for sampler in OfflineSampler.instances] == [30, 30, 30, 30, 50]


def test_normal_classifier_subclass_clones_and_fits(example, tmp_path):
    _, classifier_class, _ = example
    class NamedClassifier(classifier_class):
        def __init__(self, name='child', epochs_mlp=2):
            super().__init__(hidden_dims=[], output_dim=2, epochs_mlp=epochs_mlp,
                             save_path=str(tmp_path))
            self.name = name
    model = NamedClassifier(name='custom').fit(*data(width=2))
    copied = clone(model)
    assert copied.get_params() == {'name': 'custom', 'epochs_mlp': 2}
    assert copied.model is None
    copied.fit(*data(width=3))
    assert copied.input_dim_ == 3
    assert copied.predict(data(width=3)[0]).shape == (25,)
