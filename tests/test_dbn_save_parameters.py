"""Parameter-dump contract for the supervised DBN example."""

import os
import sys

import numpy as np
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "example", "dbn_digits"))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from kaiwu.torch_plugin import RestrictedBoltzmannMachine  # noqa: E402
from supervised_dbn_digits import SupervisedDBNClassification  # noqa: E402


def build_rbm():
    """Two visible and three hidden units with known values."""
    rbm = RestrictedBoltzmannMachine(num_visible=2, num_hidden=3)
    with torch.no_grad():
        rbm.quadratic_coef.copy_(
            torch.arange(6, dtype=rbm.quadratic_coef.dtype).view(2, 3)
        )
        rbm.linear_bias.copy_(torch.arange(5, dtype=rbm.linear_bias.dtype))
    return rbm


def build_model(rbm, fine_tune_network=None):
    """Builds only the attributes ``save_parameters`` reads."""
    model = SupervisedDBNClassification.__new__(SupervisedDBNClassification)
    from types import SimpleNamespace

    model.unsupervised_dbn = SimpleNamespace(
        _n_layers=1 if rbm is not None else 0,
        get_rbm_layer=lambda index: rbm,
    )
    model.fine_tuning = fine_tune_network is not None
    if fine_tune_network is not None:
        model.fine_tune_network = fine_tune_network
    return model


def test_pretrained_layer_coefficients_and_hidden_bias_are_saved(tmp_path, monkeypatch):
    """One weights file and one hidden-bias file are written per RBM layer."""
    monkeypatch.chdir(tmp_path)
    model = build_model(build_rbm())
    model.save_parameters("demo")
    weights = np.load(tmp_path / "data" / "demo_pretrain_layer0_weights.npy")
    bias = np.load(tmp_path / "data" / "demo_pretrain_layer0_bias.npy")
    assert weights.shape == (2, 3)
    assert weights.tolist() == [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]
    assert bias.tolist() == [2.0, 3.0, 4.0]


def test_fine_tune_linear_layers_are_saved(tmp_path, monkeypatch):
    """Only linear fine-tune layers contribute their weights and bias."""
    monkeypatch.chdir(tmp_path)
    linear = torch.nn.Linear(3, 4)
    model = build_model(None, fine_tune_network=[linear, torch.nn.ReLU()])
    model.save_parameters("demo")
    weights = np.load(tmp_path / "data" / "demo_finetune_layer0_weights.npy")
    bias = np.load(tmp_path / "data" / "demo_finetune_layer0_bias.npy")
    assert weights.shape == (4, 3)
    assert bias.shape == (4,)
    assert np.array_equal(weights, linear.weight.detach().numpy())
