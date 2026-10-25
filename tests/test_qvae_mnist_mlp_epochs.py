"""MLP epoch-loop contract for the QVAE-MNIST downstream classifier."""

import os
import sys

import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ORIGINAL_PATH = list(sys.path)
sys.path[:] = [
    entry
    for entry in ORIGINAL_PATH
    if not entry.endswith("qdiffusion") and not entry.endswith("dplm")
]
for _name in [m for m in list(sys.modules) if m == "utils" or m.startswith("utils.")]:
    del sys.modules[_name]
sys.path.insert(0, os.path.join(REPO_ROOT, "example", "qvae_mnist"))

from downstream.classifier import MLPClassifier  # noqa: E402

sys.path[:] = ORIGINAL_PATH


class FixedLogits(torch.nn.Module):
    """Returns one fixed logits batch per call."""

    def __init__(self, batches):
        super().__init__()
        for index, batch in enumerate(batches):
            self.register_buffer(f"batch{index}", batch)
        self.batches = batches
        self.calls = 0

    def forward(self, x):
        batch = getattr(self, f"batch{self.calls}")
        self.calls += 1
        return batch


class CountingSGD(torch.optim.SGD):
    """Counts optimizer steps."""

    steps = 0

    def step(self, *args, **kwargs):
        CountingSGD.steps += 1
        return super().step(*args, **kwargs)


def test_evaluation_reports_accuracy_and_mean_batch_loss():
    """Both batches are scored with row-wise argmax and per-batch loss mean."""
    model = FixedLogits(
        [
            torch.tensor([[10.0, 0.0], [0.0, 10.0]]),
            torch.tensor([[10.0, 0.0]]),
        ]
    )
    loader = [
        (torch.zeros(2, 2), torch.tensor([0, 1])),
        (torch.zeros(1, 2), torch.tensor([0])),
    ]
    classifier = MLPClassifier.__new__(MLPClassifier)
    accuracy, loss = classifier._eval_mlp_epoch(
        model, loader, torch.nn.CrossEntropyLoss(), torch.device("cpu")
    )
    assert accuracy == 100.0
    assert loss < 1e-2
    assert model.training is False


def test_training_steps_once_per_batch_and_reports_accuracy():
    """Every batch zeroes/steps the optimizer and contributes to accuracy."""
    model = torch.nn.Linear(2, 2)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[10.0, 0.0], [0.0, 10.0]]))
        model.bias.zero_()
    loader = [
        (torch.tensor([[1.0, 0.0], [0.0, 1.0]]), torch.tensor([0, 1])),
        (torch.tensor([[1.0, 0.0]]), torch.tensor([0])),
    ]
    optimizer = CountingSGD(model.parameters(), lr=0.0)
    CountingSGD.steps = 0
    classifier = MLPClassifier.__new__(MLPClassifier)
    accuracy, loss = classifier._train_mlp_epoch(
        model, loader, optimizer, torch.nn.CrossEntropyLoss(), torch.device("cpu")
    )
    assert CountingSGD.steps == 2
    assert accuracy == 100.0
    assert loss > 0.0
    assert model.training is True
