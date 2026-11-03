"""Epoch losses match per-observation cross entropy for uneven batches."""
import importlib.util
from pathlib import Path
import sys
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, SubsetRandomSampler


@pytest.fixture
def classifier(monkeypatch):
    root = Path(__file__).resolve().parents[1] / "example/qvae_mnist"
    # The example uses a top-level utils package; do not retain it between tests.
    for name in list(sys.modules):
        if name == "utils" or name.startswith("utils."):
            monkeypatch.delitem(sys.modules, name)
    monkeypatch.syspath_prepend(str(root))
    spec = importlib.util.spec_from_file_location("mnist_mlp_epoch_test", root / "downstream/classifier.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    yield module.MLPClassifier(input_dim=2, output_dim=2, device=torch.device("cpu"))
    for name in list(sys.modules):
        if name == "utils" or name.startswith("utils."):
            del sys.modules[name]


@pytest.mark.parametrize("training", [False, True])
@pytest.mark.parametrize("batch_size", [1, 2, 4, 5])
def test_epoch_loss_is_sample_mean(classifier, training, batch_size):
    x = torch.tensor([[3., -3.], [2., -2.], [1., -1.], [0., 0.], [-5., 5.]])
    y = torch.zeros(5, dtype=torch.long)
    model = nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.eye(2))
    loader = DataLoader(TensorDataset(x, y), batch_size=batch_size)
    criterion = nn.CrossEntropyLoss()
    expected = criterion(model(x), y).item()
    if training:
        optimizer = torch.optim.SGD(model.parameters(), lr=0.)
        accuracy, loss = classifier._train_mlp_epoch(model, loader, optimizer, criterion, "cpu")
        assert model.weight.grad is not None
    else:
        accuracy, loss = classifier._eval_mlp_epoch(model, loader, criterion, "cpu")
    assert loss == pytest.approx(expected, rel=1e-6)
    assert accuracy == 80.


def test_validation_denominator_counts_processed_subset(classifier):
    x = torch.tensor([[4., -4.], [2., -2.], [-3., 3.], [-20., 20.]])
    y = torch.zeros(4, dtype=torch.long)
    loader = DataLoader(TensorDataset(x, y), batch_size=2,
                        sampler=SubsetRandomSampler([0, 1, 2]))
    criterion = nn.CrossEntropyLoss()
    accuracy, loss = classifier._eval_mlp_epoch(nn.Identity(), loader, criterion, "cpu")
    assert loss == pytest.approx(criterion(x[:3], y[:3]).item(), rel=1e-6)
    assert accuracy == pytest.approx(200. / 3.)
