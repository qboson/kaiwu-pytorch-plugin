"""Optimizer and loader factory contract for the supervised DBN example."""

import os
import sys

import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "example", "dbn_digits"))

from supervised_dbn_digits import SupervisedDBNClassification  # noqa: E402


def build_model(batch_size=2):
    """Builds only the attributes the two factories read."""
    model = SupervisedDBNClassification.__new__(SupervisedDBNClassification)
    model.learning_rate = 0.25
    model.l2_regularization = 1e-3
    model.batch_size = batch_size
    return model


def test_optimizer_carries_the_learning_rate_and_weight_decay():
    """The SGD optimizer uses the configured lr and L2 regularisation."""
    model = build_model()
    parameters = [torch.nn.Parameter(torch.zeros(2))]
    optimizer = model._create_optimizer(parameters)
    assert isinstance(optimizer, torch.optim.SGD)
    group = optimizer.param_groups[0]
    assert group["lr"] == 0.25
    assert group["weight_decay"] == 1e-3
    assert group["params"] == parameters


def test_data_loader_uses_the_configured_batch_size_and_order():
    """The loader wraps the tensors with the model batch size."""
    model = build_model(batch_size=2)
    X = torch.arange(6, dtype=torch.float32).view(3, 2)
    y = torch.tensor([0, 1, 0])
    loader = model._create_data_loader(X, y, shuffle=False)
    batches = list(loader)
    assert len(batches) == 2
    assert batches[0][0].shape == (2, 2)
    assert batches[0][1].tolist() == [0, 1]
    assert batches[1][1].tolist() == [0]
