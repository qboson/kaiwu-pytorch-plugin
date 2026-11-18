"""Exact marginal values/gradients and projection scaling, without solver access.

Run in separate processes with QC_MODEL_SOURCE=sqa or cim_sa.
"""
import itertools
import os
from pathlib import Path
import sys
from unittest.mock import patch

import pytest
import torch

SOURCE = os.environ.get("QC_MODEL_SOURCE", "sqa")
if SOURCE not in ("sqa", "cim_sa"):
    raise ValueError("QC_MODEL_SOURCE must be sqa or cim_sa")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / SOURCE))

from qc_mol_design.config import CFG, N_BINS_PER_PROP
from qc_mol_design.models.crbm import CRBM


@pytest.fixture
def model():
    with patch.dict(N_BINS_PER_PROP, {"qed": 2, "logp": 3, "sas": 4}), \
            patch.object(CFG, "n_hidden", 5), patch.object(CFG, "fp_dim", 7):
        torch.manual_seed(47)
        result = CRBM().double()
        with torch.no_grad():
            for parameter in result.parameters():
                parameter.normal_(0, 0.3)
        yield result


def explicit_marginal(model, prop, target_bin, fingerprints):
    """Independent enumeration using the public paired free-energy method."""
    group = ("qed", "logp", "sas").index(prop)
    energies = []
    for bins in itertools.product(*(range(n) for n in CFG.prop_bin_counts)):
        if bins[group] != target_bin:
            continue
        label = fingerprints.new_zeros(CFG.n_vis)
        for offset, index in zip(CFG.prop_offsets, bins):
            label[offset + index] = 1
        energies.append(model.free_energy(label.expand(len(fingerprints), -1), fingerprints))
    return -torch.logsumexp(-torch.stack(energies, dim=1), dim=1)


@pytest.mark.parametrize("prop,target_bin", [("qed", 1), ("logp", 2), ("sas", 3)])
@pytest.mark.parametrize("batch", [1, 4])
def test_values_and_all_gradients_match_enumeration(model, prop, target_bin, batch):
    fingerprints = torch.randn(batch, 7, dtype=torch.float64, requires_grad=True)
    actual = model.target_free_energy(prop, target_bin, fingerprints)
    expected = explicit_marginal(model, prop, target_bin, fingerprints)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    inputs = (fingerprints, *model.parameters())
    actual_grads = torch.autograd.grad(actual.sum(), inputs)
    expected_grads = torch.autograd.grad(expected.sum(), inputs)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-11, atol=1e-11)


def test_fingerprint_projection_is_not_repeated_for_label_combinations(model):
    fingerprints = torch.randn(4, 7, dtype=torch.float64)
    with patch.object(model, "effective_biases", wraps=model.effective_biases) as project:
        model.target_free_energy("qed", 1, fingerprints)
    project.assert_called_once()
    assert project.call_args.args[0].shape == fingerprints.shape
