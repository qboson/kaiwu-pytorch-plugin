"""The Digits classifier must preserve its unsupervised RBM snapshot."""

from pathlib import Path

import numpy as np
import pytest
import torch


def test_fine_tuning_preserves_pretrained_features_and_exports(monkeypatch, tmp_path):
    """Run both training stages and check the separately exported weights."""
    pytest.importorskip("scipy")
    pytest.importorskip("sklearn")
    pytest.importorskip("matplotlib")
    pytest.importorskip("seaborn")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "example/dbn_digits"))
    from supervised_dbn_digits import SupervisedDBNClassification

    class LocalSampler:
        """Return legal spins without using a licensed backend."""

        def solve(self, matrix):
            return -np.ones((1, matrix.shape[0]))

    model = SupervisedDBNClassification(
        hidden_layers_structure=[4, 3], n_epochs_rbm=1, n_iter_backprop=3,
        batch_size=4, verbose=False, random_state=7,
    )
    monkeypatch.setattr(model.unsupervised_dbn._trainer, "sampler", LocalSampler())
    features = np.random.default_rng(12).random((12, 3), dtype=np.float32)
    labels = np.arange(len(features)) % 2
    model.pre_train(features)
    rbms = [model.unsupervised_dbn.get_rbm_layer(i) for i in range(2)]
    weights = [rbm.quadratic_coef.detach().clone() for rbm in rbms]
    biases = [rbm.hidden_bias.detach().clone() for rbm in rbms]
    before = model.transform(features)

    model.fit(features, labels, pre_train=False)
    assert not torch.equal(model.fine_tune_network[0].weight.detach(), weights[0].T)
    monkeypatch.chdir(tmp_path)
    model.save_parameters()
    for index, rbm in enumerate(rbms):
        torch.testing.assert_close(rbm.quadratic_coef, weights[index], atol=0, rtol=0)
        torch.testing.assert_close(rbm.hidden_bias, biases[index], atol=0, rtol=0)
        np.testing.assert_array_equal(
            np.load(f"data/dbn_model_pretrain_layer{index}_weights.npy"), weights[index].numpy(),
        )
        np.testing.assert_array_equal(
            np.load(f"data/dbn_model_pretrain_layer{index}_bias.npy"), biases[index].numpy(),
        )
    np.testing.assert_array_equal(model.transform(features), before)
