"""Degree-conditioned convolution values and gradients against per-node math."""
import os
from pathlib import Path
import sys

import pytest
import torch

SOURCE = os.environ.get("QC_MODEL_SOURCE", "sqa")
if SOURCE not in ("sqa", "cim_sa"):
    raise ValueError("QC_MODEL_SOURCE must be sqa or cim_sa")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / SOURCE))
from qc_mol_design.models.graph_conv import DuvenaudConvLayer


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("mask_kind", ["all", "partial", "none"])
def test_per_node_reference_and_gradients(dtype, mask_kind):
    torch.manual_seed(29)
    layer = DuvenaudConvLayer(4, 5, 3).to(dtype=dtype)
    # Check gradients as well as the normal frozen-weight inference contract.
    layer.requires_grad_(True)
    with torch.no_grad():
        layer.bias.copy_(torch.linspace(-.2, .2, 5, dtype=dtype))
    features = torch.randn(2, 5, 4, dtype=dtype, requires_grad=True)
    adjacency = torch.randn(2, 5, 5, dtype=dtype, requires_grad=True)
    degrees = torch.tensor([[0, 2, 2, 3, 4], [3, 3, 2, 0, -1]])
    # Degree 1 is absent; out-of-range degrees retain the historical zero projection.
    mask = torch.ones(2, 5, dtype=torch.bool)
    if mask_kind == "partial":
        mask[:, -2:] = False
    elif mask_kind == "none":
        mask[:] = False
    actual, actual_pooled = layer(features, adjacency, degrees, mask)
    rows = []
    for batch in range(2):
        nodes = []
        for node in range(5):
            degree = int(degrees[batch, node])
            if mask[batch, node] and 0 <= degree <= layer.max_degree:
                aggregated = features[batch, node] + (
                    adjacency[batch, node, :, None] * features[batch]
                ).sum(0)
                projected = aggregated @ layer.weights[degree]
            else:
                projected = features.new_zeros(5)
            nodes.append(torch.relu(projected + layer.bias))
        rows.append(torch.stack(nodes))
    expected = torch.stack(rows)
    expected_pooled = (expected.softmax(-1) * mask.unsqueeze(-1)).sum(1)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual_pooled, expected_pooled)
    inputs = (features, adjacency, *layer.parameters())
    for output, reference in ((actual, expected), (actual_pooled, expected_pooled)):
        actual_grads = torch.autograd.grad(output.sum(), inputs, retain_graph=True, allow_unused=True)
        expected_grads = torch.autograd.grad(reference.sum(), inputs, retain_graph=True, allow_unused=True)
        for parameter, left, right in zip(inputs, actual_grads, expected_grads):
            torch.testing.assert_close(
                torch.zeros_like(parameter) if left is None else left,
                torch.zeros_like(parameter) if right is None else right,
            )


def test_omitted_mask_matches_all_real_nodes():
    layer = DuvenaudConvLayer(2, 3, 2)
    features = torch.randn(2, 4, 2)
    adjacency = torch.zeros(2, 4, 4)
    degrees = torch.zeros(2, 4, dtype=torch.long)
    for left, right in zip(layer(features, adjacency, degrees),
                           layer(features, adjacency, degrees, torch.ones_like(degrees, dtype=torch.bool))):
        torch.testing.assert_close(left, right)
