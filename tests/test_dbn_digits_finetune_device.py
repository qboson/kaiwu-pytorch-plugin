"""Regression tests for the device of the dbn_digits fine-tuning network.

``_fine_tuning`` and both prediction paths move their inputs to
``self.unsupervised_dbn.device`` (CUDA whenever a GPU is available), so the
network built by ``_build_fine_tune_network`` must live on that device too.
Without the move it stayed on CPU and every fine-tuning run failed on a CUDA
host.

This environment may not have CUDA, so the two-device split is reproduced with
the ``meta`` device: the network must follow whatever device the DBN reports.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("sklearn")
pytest.importorskip("scipy")
pytest.importorskip("matplotlib")
pytest.importorskip("seaborn")

from kaiwu.torch_plugin import RestrictedBoltzmannMachine

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "example" / "dbn_digits"


def _load_example(monkeypatch):
    """Import the example module without training or sampling."""
    monkeypatch.syspath_prepend(str(EXAMPLE))
    spec = importlib.util.spec_from_file_location(
        "digits_supervised_device", EXAMPLE / "supervised_dbn_digits.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _StubDBN:
    """Minimal stand-in exposing the interface used by the network builder."""

    def __init__(self, rbm, device):
        self._rbm = rbm
        self.device = device
        self._n_layers = 1

    def get_rbm_layer(self, index):
        """Return the single stubbed RBM layer."""
        return self._rbm if index == 0 else None


def _classifier_on(module, device):
    torch.manual_seed(0)
    rbm = RestrictedBoltzmannMachine(num_visible=6, num_hidden=4)
    classifier = module.SupervisedDBNClassification(
        hidden_layers_structure=[4],
        fine_tuning=True,
        verbose=False,
    )
    classifier.unsupervised_dbn = _StubDBN(rbm, device)
    classifier.classes_ = np.array([0, 1, 2])
    classifier._build_fine_tune_network()
    return classifier


def test_fine_tune_network_follows_a_non_default_device(monkeypatch):
    """The network must be built on the DBN device, not the default CPU one."""
    module = _load_example(monkeypatch)
    device = torch.device("meta")
    network = _classifier_on(module, device).fine_tune_network

    devices = {parameter.device for parameter in network.parameters()}
    assert devices == {device}, (
        "fine-tuning network must follow the DBN device; on a CUDA host it "
        f"would stay on CPU. Got {devices}."
    )


def test_fine_tune_network_stays_on_cpu_for_cpu_dbns(monkeypatch):
    """A CPU DBN keeps the previous behaviour."""
    module = _load_example(monkeypatch)
    network = _classifier_on(module, torch.device("cpu")).fine_tune_network

    devices = {parameter.device for parameter in network.parameters()}
    assert devices == {torch.device("cpu")}

