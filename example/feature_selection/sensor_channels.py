"""Select two sensor channels using the change in each signal over a time window."""

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from kaiwu.torch_plugin import FeatureSelectionWrapper


def sensor_windows(count: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate six independent sensor traces and a target driven by two slopes."""
    generator = torch.Generator().manual_seed(seed)
    slopes = torch.randn(count, 6, generator=generator)
    offsets = torch.randn(count, 6, generator=generator)
    time = torch.linspace(-1, 1, steps=12)
    noise = torch.randn(count, 6, len(time), generator=generator)
    windows = offsets.unsqueeze(-1) + slopes.unsqueeze(-1) * time + 0.1 * noise
    target = 1.8 * slopes[:, 1] - 1.3 * slopes[:, 4]
    return windows, target.unsqueeze(-1)


class TrendRegressor(nn.Module):
    """Predict from the difference between the late and early channel means."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.readout = nn.Linear(channels, 1, bias=False)
        nn.init.zeros_(self.readout.weight)

    def forward(self, windows: torch.Tensor) -> torch.Tensor:
        """Predict one target per window from its per-channel trends."""
        early, late = windows.chunk(2, dim=-1)
        return self.readout(late.mean(dim=-1) - early.mean(dim=-1))


def run_example(
    train_seed: int, test_seed: int
) -> tuple[list[int], float, float, float]:
    """Return the learned channels and held-out full, selected, and baseline MSE."""
    train_x, train_y = sensor_windows(192, seed=train_seed)
    test_x, test_y = sensor_windows(96, seed=test_seed)
    train = DataLoader(TensorDataset(train_x, train_y), batch_size=64)
    channels = train_x.shape[1]
    selector = FeatureSelectionWrapper(
        TrendRegressor(channels),
        feature_dim=channels,
        input_feature_axis=1,  # One mask entry per sensor, shared over time.
        cardinality_k=2,
        solver="local_search",
    )
    loss = nn.MSELoss()
    optimizer = torch.optim.SGD(selector.model.parameters(), lr=0.1)
    selector.fit_weights(train, loss, optimizer, train_epochs=20)

    with torch.no_grad():
        full_mse = loss(selector(test_x), test_y).item()
        baseline_mse = loss(train_y.mean().expand_as(test_y), test_y).item()

    selector.update_mask(train, loss)
    with torch.no_grad():
        selected_mse = loss(selector(test_x), test_y).item()

    return selector.selected_indices().tolist(), full_mse, selected_mse, baseline_mse


if __name__ == "__main__":
    selected, full_error, selected_error, baseline_error = run_example(42, 43)
    print(f"Selected channels (zero-based): {selected}")
    print(
        f"Held-out MSE: all channels={full_error:.4f}, "
        f"selected={selected_error:.4f}, training mean={baseline_error:.4f}"
    )
