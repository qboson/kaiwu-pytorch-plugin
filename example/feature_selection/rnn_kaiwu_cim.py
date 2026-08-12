from __future__ import annotations

import sys
from pathlib import Path

import torch
from kaiwu.cim._optimizer_adapter import TaskMode
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from kaiwu.torch_plugin import FeatureSelectionWrapper

KAIWU_PROJECT_NAME = "Your Project Name"
KAIWU_PROJECT_NO = "Your Project No"

sample_count = 20
sequence_length = 6
feature_dim = 8
signal_features = [1, 6]

x = torch.randn(sample_count, sequence_length, feature_dim)
score = x[:, :, signal_features[0]].mean(dim=1) - x[:, :, signal_features[1]].mean(dim=1)
y = (score > 0).long()

loader = DataLoader(TensorDataset(x, y), batch_size=10, shuffle=False)


class SimpleRNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.rnn = nn.RNN(feature_dim, hidden_size=8, batch_first=True)
        self.head = nn.Linear(8, 2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        _sequence, hidden = self.rnn(inputs)
        return self.head(hidden[-1])


selector = FeatureSelectionWrapper(
    SimpleRNN(),
    feature_dim=feature_dim,
    cardinality_k=len(signal_features),
    solver="kaiwu_cim",
    solver_kwargs={
        "target_precision": 8,
        "max_bits": 1000,
        "max_precision": 8,
        "precision_step": 4,
        "sample_number": 10,
        "task_mode": TaskMode.SAMPLE,
        "project_no": KAIWU_PROJECT_NO,
    },
    mask_update_epochs=5,
)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(selector.model.parameters(), lr=0.01)

selector.fit_weights(loader, loss_fn, optimizer, epochs=20)
logits = selector(x)
accuracy = (logits.argmax(dim=1) == y).float().mean().item()

print(f"Kaiwu project: {KAIWU_PROJECT_NAME} ({KAIWU_PROJECT_NO})")
print("model: rnn")
print("solver: kaiwu_cim")
print("loss after:", float(loss_fn(logits, y).detach()))
print("accuracy:", accuracy)
print("true signal features:", signal_features)
print("selected features:", selector.selected_indices().tolist())
print("selected mask:", selector.get_support().astype(int).tolist())
