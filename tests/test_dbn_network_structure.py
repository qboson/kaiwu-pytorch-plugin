"""Network-structure report contract for the supervised DBN classifier."""

import os
import sys
from types import SimpleNamespace

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "example", "dbn_digits"))

from supervised_dbn_digits import SupervisedDBNClassification  # noqa: E402


def build_model(fine_tuning=False, fine_tune_network=None):
    """Builds only the attributes ``get_network_structure`` reads."""
    model = SupervisedDBNClassification.__new__(SupervisedDBNClassification)
    model.unsupervised_dbn = SimpleNamespace(_n_layers=3)
    model.hidden_layers_structure = [128, 64]
    model.classes_ = [0, 1, 2]
    model.fine_tuning = fine_tuning
    if fine_tune_network is not None:
        model.fine_tune_network = fine_tune_network
    return model


def test_classifier_mode_structure():
    """Without fine-tuning the layer count falls back to zero."""
    structure = build_model().get_network_structure()
    assert structure == {
        "pretrain_layers": 3,
        "fine_tune_layers": 0,
        "hidden_units": [128, 64],
        "num_classes": 3,
        "mode": "classifier",
    }


def test_fine_tuning_mode_counts_the_network_modules():
    """A fine-tune network contributes its module count and the mode name."""
    model = build_model(fine_tuning=True, fine_tune_network=[1, 2, 3, 4])
    structure = model.get_network_structure()
    assert structure["fine_tune_layers"] == 4
    assert structure["mode"] == "fine_tuning"
