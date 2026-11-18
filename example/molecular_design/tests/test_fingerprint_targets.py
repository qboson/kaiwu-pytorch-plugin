"""Single-property baseline targets must pair elementwise with predictions."""
import os
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import pytest
import torch

SOURCE = os.environ.get("QC_MODEL_SOURCE", "sqa")
if SOURCE not in ("sqa", "cim_sa"):
    raise ValueError("QC_MODEL_SOURCE must be sqa or cim_sa")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / SOURCE))
from qc_mol_design.baselines.fingerprint_models import (
    FingerprintDataset, evaluate_fingerprint_predictor, train_fingerprint_predictor,
)
from qc_mol_design.config import CFG


@pytest.fixture(autouse=True)
def cpu_only():
    with patch("torch.cuda.is_available", return_value=False):
        yield


class FirstFeature(torch.nn.Module):
    def forward(self, features):
        return features[:, 0]


@pytest.mark.parametrize("column", [False, True])
def test_exact_predictions_have_zero_mae(column):
    features = np.array([[1., 0.], [3., 0.]], dtype=np.float32)
    targets = features[:, :1] if column else features[:, 0]
    assert evaluate_fingerprint_predictor(FirstFeature(), features, targets) == 0.


def test_column_targets_produce_scalar_dataset_items():
    dataset = FingerprintDataset(np.ones((3, 2)), np.arange(3).reshape(3, 1))
    assert dataset[0][1].shape == ()
    torch.testing.assert_close(dataset.props, torch.arange(3, dtype=torch.float32))


def test_column_and_vector_targets_train_identically():
    features = np.arange(12, dtype=np.float32).reshape(4, 3) / 12
    targets = np.array([0., 1., 2., 3.], dtype=np.float32)
    models = []
    with patch.object(CFG, "batch_size", 4):
        for shape in ((4,), (4, 1)):
            torch.manual_seed(19)
            models.append(train_fingerprint_predictor(features, targets.reshape(shape), 3, epochs=2))
    for vector, column in zip(models[0].parameters(), models[1].parameters()):
        torch.testing.assert_close(vector, column, rtol=0, atol=0)


@pytest.mark.parametrize("targets", [np.zeros((3, 2)), np.zeros((1, 3)), np.zeros(2)])
def test_invalid_target_shapes_fail_before_broadcasting(targets):
    features = np.zeros((3, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="propert"):
        FingerprintDataset(features, targets)
    with pytest.raises(ValueError, match="propert"):
        evaluate_fingerprint_predictor(FirstFeature(), features, targets)
