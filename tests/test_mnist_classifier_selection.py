"""Real MLP training must restore the selected validation checkpoint."""
import importlib.util
from pathlib import Path
import sys
from types import ModuleType

import numpy as np
import pytest
import torch


@pytest.fixture
def classifier_module(monkeypatch):
    """Isolate the unrelated optional plotting helper while loading the real file."""
    plots = []
    helpers = ModuleType("utils.helpers")
    helpers.plot_training_curves = lambda **kwargs: plots.append(kwargs)
    utilities = ModuleType("utils")
    utilities.__path__ = []
    monkeypatch.setitem(sys.modules, "utils", utilities)
    monkeypatch.setitem(sys.modules, "utils.helpers", helpers)
    filename = Path(__file__).resolve().parents[1] / "example/qvae_mnist/downstream/classifier.py"
    specification = importlib.util.spec_from_file_location("mnist_selection_classifier", filename)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    assert Path(module.__file__).resolve() == filename.resolve()
    return module, plots


@pytest.mark.parametrize("persist", [False, True])
def test_all_zero_validation_accuracy_retains_a_valid_checkpoint(
    tmp_path, classifier_module, persist
):
    """A genuinely wrong classifier still completes real training and validation."""
    module, plots = classifier_module
    directory = str(tmp_path) if persist else None
    features = np.array([[-1., 0.], [1., 0.]] * 10, dtype=np.float32)
    labels = np.array([0, 1] * 10)
    estimator = module.MLPClassifier(
        hidden_dims=[], output_dim=2, lr_mlp=0., epochs_mlp=2,
        device="cpu", save_path=directory,
    )
    create = estimator._create_model
    def always_wrong_model():
        model = create()
        with torch.no_grad():
            model[0].weight.copy_(torch.tensor([[1., 0.], [-1., 0.]]))
            model[0].bias.zero_()
        return model
    estimator._create_model = always_wrong_model

    assert estimator.fit(features, labels) is estimator

    assert estimator.score(features, labels) == 0.
    assert np.all(np.isfinite(estimator.predict_proba(features)))
    assert plots[0]["val_acc_history"] == [0., 0.]
    if persist:
        checkpoint = torch.load(tmp_path / "best_mlp_classifier.pth", weights_only=True)
        for name, value in estimator.model.state_dict().items():
            torch.testing.assert_close(value, checkpoint[name])


@pytest.mark.parametrize("persist", [False, True])
@pytest.mark.parametrize("ranks,selected", [
    ([80., 40., 20.], 0),
    ([20., 80., 40.], 1),
    ([80., 80., 80.], 0),
])
def test_fit_restores_the_selected_epoch_snapshot(
    tmp_path, classifier_module, persist, ranks, selected
):
    """Ranking controls isolate model selection from real evolving optimizer weights."""
    module, plots = classifier_module
    generator = np.random.default_rng(7)
    features = generator.normal(size=(40, 2)).astype(np.float32)
    labels = (features[:, 0] > 0).astype(np.int64)
    estimator = module.MLPClassifier(
        hidden_dims=[], output_dim=2, lr_mlp=0.1, epochs_mlp=3,
        device="cpu", save_path=str(tmp_path) if persist else None,
        weight_decay=0.,
    )
    evaluate = estimator._eval_mlp_epoch
    snapshots = []
    actual_scores = []
    def rank_real_evaluation(*args, **kwargs):
        accuracy, loss = evaluate(*args, **kwargs)
        actual_scores.append(accuracy)
        snapshots.append({
            key: value.detach().clone()
            for key, value in estimator.model.state_dict().items()
        })
        # Only validation ranking is controlled. Forward, cross entropy,
        # backpropagation, optimizer steps and actual validation all run.
        return ranks[len(snapshots) - 1], loss
    estimator._eval_mlp_epoch = rank_real_evaluation

    estimator.fit(features, labels)

    assert len(snapshots) == 3
    assert all(0. <= score <= 100. for score in actual_scores)
    assert any(not torch.equal(snapshots[0][key], snapshots[-1][key])
               for key in snapshots[0])
    for name, value in estimator.model.state_dict().items():
        torch.testing.assert_close(value, snapshots[selected][name])
    assert plots[0]["val_acc_history"] == ranks
    assert all(np.isfinite(plots[0]["train_loss_history"]))
    if persist:
        checkpoint = torch.load(tmp_path / "best_mlp_classifier.pth", weights_only=True)
        for name, value in checkpoint.items():
            torch.testing.assert_close(value, snapshots[selected][name])
