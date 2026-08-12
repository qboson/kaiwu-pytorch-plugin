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
sample_count = 10
feature_dim = 10
signal_indices = [2, 7]

x = torch.randn(sample_count, feature_dim)
true_weight = torch.zeros(feature_dim, 1)
true_weight[signal_indices] = torch.tensor([[6.0], [-4.0]])
y = x @ true_weight + 0.001 * torch.randn(sample_count, 1)

loader = DataLoader(TensorDataset(x, y), batch_size=10, shuffle=False)
model = nn.Linear(feature_dim, 1, bias=False)

selector = FeatureSelectionWrapper(
    model,
    feature_dim=feature_dim,
    cardinality_k=len(signal_indices),
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

loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(selector.model.parameters(), lr=0.05)

selector.fit_weights(loader, loss_fn, optimizer, epochs=10)

print(f"Kaiwu project: {KAIWU_PROJECT_NAME} ({KAIWU_PROJECT_NO})")
print("solver: kaiwu_cim")
print("loss after:", float(loss_fn(selector(x), y).detach()))
print("true signal indices:", signal_indices)
print("selected indices:", selector.selected_indices().tolist())
print("selected mask:", selector.get_support().astype(int).tolist())
