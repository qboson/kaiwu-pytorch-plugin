"""Small classifiers used by the feature-selection examples."""

from __future__ import annotations

import torch
from torch import nn


class TinyCNN(nn.Module):
    """Small CNN used by the feature-selection classification example."""

    def __init__(self, image_size: int) -> None:
        super().__init__()
        self.image_size = int(image_size)
        self.net = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(8 * self.image_size * self.image_size, 2),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return class logits for flattened image inputs.

        ``inputs`` is aligned to the live parameter device before the reshape.
        Conv activations stay on that device, so the linear head needs no
        second move. Dtype is unchanged. The caller tensor is not modified in
        place, and a same-device batch keeps its storage.
        """
        images = _on_parameter_device(self, inputs).view(
            -1, 1, self.image_size, self.image_size
        )
        return self.net(images)


class SimpleRNN(nn.Module):
    """Small RNN classifier used by the feature-selection example."""

    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.rnn = nn.RNN(feature_dim, hidden_size=8, batch_first=True)
        self.head = nn.Linear(8, 2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return class logits for sequence inputs.

        ``inputs`` is aligned to the live parameter device. The recurrent
        state stays on that device, so the linear head needs no second move.
        Dtype is unchanged. The caller tensor is not modified in place, and a
        same-device batch keeps its storage.
        """
        _sequence, hidden = self.rnn(_on_parameter_device(self, inputs))
        return self.head(hidden[-1])


class SimpleLSTM(nn.Module):
    """Small LSTM classifier used by the feature-selection example."""

    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.lstm = nn.LSTM(feature_dim, hidden_size=8, batch_first=True)
        self.head = nn.Linear(8, 2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return class logits for sequence inputs.

        ``inputs`` is aligned to the live parameter device. The recurrent
        state stays on that device, so the linear head needs no second move.
        Dtype is unchanged. The caller tensor is not modified in place, and a
        same-device batch keeps its storage.
        """
        _sequence, (hidden, _cell) = self.lstm(_on_parameter_device(self, inputs))
        return self.head(hidden[-1])


def _parameter_device(module: nn.Module, fallback: torch.Tensor) -> torch.device:
    """Return the device ``module`` parameters currently live on.

    A parent ``_apply`` moves conv, recurrent, and linear weights together
    and does not call the child ``.to``. Forwards follow the first parameter,
    which stays on the same device as the linear head after that move. An
    empty module falls back to ``fallback.device``.
    """
    parameter = next(module.parameters(), None)
    if parameter is None:
        return fallback.device
    return parameter.device


def _on_parameter_device(module: nn.Module, inputs: torch.Tensor) -> torch.Tensor:
    """Return ``inputs`` on the parameter device without changing dtype.

    ``Tensor.to`` returns the same object when the device already matches, so
    same-device callers keep their storage. A different device is a copy.
    The caller tensor is not written in place. These classifiers do not take
    an embedding index, an attention mask, or a dropout index.
    """
    return inputs.to(device=_parameter_device(module, inputs))
