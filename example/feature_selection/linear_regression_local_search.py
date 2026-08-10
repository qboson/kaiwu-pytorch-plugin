from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from kaiwu.torch_plugin import FeatureSelectionWrapper

sample_count = 1024
feature_dim = 100
signal_count = 5

rng = np.random.default_rng()
signal_indices = rng.choice(feature_dim, size=signal_count, replace=False)
signal_weights = rng.uniform(5.0, 10.0, size=signal_count)
signal_weights *= rng.choice([-1.0, 1.0], size=signal_count)

x = torch.randn(sample_count, feature_dim)
true_weight = torch.zeros(feature_dim, 1)
true_weight[torch.tensor(signal_indices, dtype=torch.long)] = torch.tensor(
    signal_weights,
    dtype=true_weight.dtype,
).view(-1, 1)
y = x @ true_weight + 0.001 * torch.randn(sample_count, 1)

loader = DataLoader(TensorDataset(x, y), batch_size=128, shuffle=False)
model = nn.Linear(feature_dim, 1, bias=False)
selector = FeatureSelectionWrapper(
    model,
    feature_dim=feature_dim,
    cardinality_k=signal_count,
    solver="local_search",
    solver_kwargs={"max_iter": 1000},
    mask_update_epochs=20,
)

loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(selector.model.parameters(), lr=0.05)

loss_before = float(loss_fn(selector(x), y).detach())
selector.fit_weights(loader, loss_fn, optimizer, epochs=20)
loss_after = float(loss_fn(selector(x), y).detach())

print("solver: local_search")
print("loss before:", loss_before)
print("loss after:", loss_after)
print("true signal indices:", sorted(signal_indices.tolist()))
print("selected indices:", selector.selected_indices().tolist())
print("selected mask:", selector.get_support().astype(int).tolist())
