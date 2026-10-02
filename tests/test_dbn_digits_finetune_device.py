"""Regression test for the device of the fine-tuning network.

`_fine_tuning` / `_predict_with_fine_tuning` move their inputs to
``self.unsupervised_dbn.device``, which is CUDA whenever a GPU is available
(``UnsupervisedDBN.__init__``), but ``_build_fine_tune_network`` created plain
``nn.Linear`` layers that stayed on CPU.  On a CUDA host every fine-tuning run
and prediction therefore failed with a device-mismatch ``RuntimeError``.

This environment has no CUDA, so the two-device split is reproduced by pointing
``unsupervised_dbn.device`` at the meta device: the network must live on the
same device as the DBN, exactly as it would on a GPU host.
"""

import os
import sys

import numpy as np
import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "example", "dbn_digits"))
sys.path.insert(0, os.path.abspath(os.path.join(REPO_ROOT, "src")))

import torch  # noqa: E402

from kaiwu.torch_plugin import RestrictedBoltzmannMachine  # noqa: E402
from supervised_dbn_digits import SupervisedDBNClassification  # noqa: E402


class _StubDBN:
    """Minimal stand-in for ``UnsupervisedDBN`` exposing the used interface."""

    def __init__(self, rbm, device):
        self._rbm = rbm
        self.device = device
        self._n_layers = 1

    def get_rbm_layer(self, index):
        return self._rbm if index == 0 else None


def _classifier_on(device):
    torch.manual_seed(0)
    rbm = RestrictedBoltzmannMachine(num_visible=6, num_hidden=4)
    classifier = SupervisedDBNClassification(
        hidden_layers_structure=[4],
        fine_tuning=True,
        verbose=False,
    )
    classifier.unsupervised_dbn = _StubDBN(rbm, device)
    classifier.classes_ = np.array([0, 1, 2])
    classifier._build_fine_tune_network()
    return classifier


def test_fine_tune_network_follows_the_dbn_device():
    device = torch.device("meta")
    classifier = _classifier_on(device)

    parameter_devices = {
        parameter.device for parameter in classifier.fine_tune_network.parameters()
    }
    assert parameter_devices == {device}, (
        "fine-tuning network must be moved to the DBN's device; on a CUDA host "
        f"it stayed on CPU. Got {parameter_devices}."
    )


def test_fine_tune_network_on_cpu():
    classifier = _classifier_on(torch.device("cpu"))

    parameter_devices = {
        parameter.device for parameter in classifier.fine_tune_network.parameters()
    }
    assert parameter_devices == {torch.device("cpu")}
