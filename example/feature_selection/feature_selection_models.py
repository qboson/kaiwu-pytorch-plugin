from __future__ import annotations

# Modified to validate model dimensions before constructing Torch layers.

from numbers import Integral

import torch
from torch import nn


def _positive_dimension(value: int, name: str) -> int:
    """Require a positive integral dimension, excluding booleans and floats."""
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


class TinyCNN(nn.Module):
    """Small CNN used by the feature-selection classification example."""

    def __init__(self, image_size: int) -> None:
        super().__init__()
        self.image_size = _positive_dimension(image_size, "image_size")
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
        self.rnn = nn.RNN(
            _positive_dimension(feature_dim, "feature_dim"), hidden_size=8, batch_first=True
        )
        self.head = nn.Linear(8, 2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return class logits for sequence inputs."""
        _sequence, hidden = self.rnn(inputs)
        return self.head(hidden[-1])


class SimpleLSTM(nn.Module):
    """Small LSTM classifier used by the feature-selection example."""

    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            _positive_dimension(feature_dim, "feature_dim"), hidden_size=8, batch_first=True
        )
        self.head = nn.Linear(8, 2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return class logits for sequence inputs."""
        _sequence, (hidden, _cell) = self.lstm(inputs)
        return self.head(hidden[-1])
