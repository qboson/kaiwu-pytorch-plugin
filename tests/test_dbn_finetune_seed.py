"""Reproducibility of supervised Digits fine-tuning with saved pretraining."""

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch


@pytest.mark.parametrize("random_state", [13, None])
def test_fine_tuning_from_same_pretraining_respects_random_state(monkeypatch, random_state):
    """Vary the caller RNG while fitting twice from the same pre-trained RBMs."""
    pytest.importorskip("scipy")
    pytest.importorskip("sklearn")
    pytest.importorskip("matplotlib")
    pytest.importorskip("seaborn")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "example/dbn_digits"))
    from supervised_dbn_digits import SupervisedDBNClassification

    class LocalSampler:
        """Provide deterministic negative states at the solver boundary."""

        def solve(self, matrix):
            return -np.ones((1, matrix.shape[0]))

    model = SupervisedDBNClassification(
        hidden_layers_structure=[4], n_epochs_rbm=1, n_iter_backprop=3,
        batch_size=4, dropout_p=0.3, verbose=False, random_state=random_state,
    )
    monkeypatch.setattr(model.unsupervised_dbn._trainer, "sampler", LocalSampler())
    features = np.random.default_rng(5).random((16, 3), dtype=np.float32)
    labels = np.arange(len(features)) % 2
    model.pre_train(features)
    first, second = deepcopy(model), deepcopy(model)

    rng_unchanged = []
    for caller_seed, estimator in ((4, first), (8, second)):
        torch.manual_seed(caller_seed)
        before = torch.get_rng_state()
        estimator.fit(features, labels, pre_train=False)
        rng_unchanged.append(torch.equal(torch.get_rng_state(), before))

    first_probas = first.predict_proba(features)
    second_probas = second.predict_proba(features)
    if random_state is None:
        assert not np.array_equal(first_probas, second_probas)
        assert rng_unchanged == [False, False]
    else:
        np.testing.assert_array_equal(first_probas, second_probas)
        assert rng_unchanged == [True, True]
