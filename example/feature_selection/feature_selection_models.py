from __future__ import annotations

from numbers import Integral

import torch
from torch import nn


def _positive_int(name: str, value: object) -> int:
    """Returns ``value`` as a positive integer, rejecting bools and floats."""
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


class TinyCNN(nn.Module):
    """Small CNN used by the feature-selection classification example."""

    def __init__(self, image_size: int) -> None:
        super().__init__()
        self.image_size = _positive_int("image_size", image_size)
        self.net = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(8 * self.image_size * self.image_size, 2),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return class logits for flattened image inputs."""
        images = inputs.view(-1, 1, self.image_size, self.image_size)
        return self.net(images)


class SimpleRNN(nn.Module):
    """Small RNN classifier used by the feature-selection example."""

    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.feature_dim = _positive_int("feature_dim", feature_dim)
        self.rnn = nn.RNN(self.feature_dim, hidden_size=8, batch_first=True)
        self.head = nn.Linear(8, 2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return class logits for sequence inputs."""
        _sequence, hidden = self.rnn(inputs)
        return self.head(hidden[-1])


class SimpleLSTM(nn.Module):
    """Small LSTM classifier used by the feature-selection example."""

    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.feature_dim = _positive_int("feature_dim", feature_dim)
        self.lstm = nn.LSTM(self.feature_dim, hidden_size=8, batch_first=True)
        self.head = nn.Linear(8, 2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return class logits for sequence inputs."""
        _sequence, (hidden, _cell) = self.lstm(inputs)
        return self.head(hidden[-1])
