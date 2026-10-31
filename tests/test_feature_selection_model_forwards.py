"""Contract test for the feature-selection model forward passes."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
MODELS = REPO / "example" / "feature_selection" / "feature_selection_models.py"


def _load_models():
    spec = importlib.util.spec_from_file_location(
        "feature_selection_models_uut", MODELS
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_tiny_cnn_flattens_images_and_returns_two_logits():
    """Accept flattened images and preserve the two-logit output width."""
    models = _load_models()
    model = models.TinyCNN(image_size=8)

    logits = model(torch.zeros(3, 8 * 8))

    assert logits.shape == (3, 2)
    assert torch.isfinite(logits).all()


def test_simple_rnn_pools_the_last_hidden_state():
    """Keep the recurrent classifier output shape for a real CPU batch."""
    models = _load_models()
    model = models.SimpleRNN(feature_dim=4)

    logits = model(torch.zeros(2, 5, 4))

    assert logits.shape == (2, 2)


def test_simple_lstm_pools_the_last_hidden_state():
    """Keep the LSTM classifier output shape for a real CPU batch."""
    models = _load_models()
    model = models.SimpleLSTM(feature_dim=3)

    logits = model(torch.zeros(4, 6, 3))

    assert logits.shape == (4, 2)


class _FaithfulRecurrent(torch.nn.Module):
    """Single-layer, one-direction fixture: last output equals final hidden.

    Low-amplitude values can be realized for suitable input/gate weights.
    In particular h=0.3 and c=0.7 admit output gate h/tanh(c) in (0, 1).
    Equivalent final-output and final-hidden implementations must both pass.
    """

    def __init__(self, *, lstm=False):
        super().__init__()
        self.lstm = lstm

    def forward(self, inputs):
        """Return mutually consistent recurrent sequence and state values."""
        assert inputs.shape[1] == 3
        values = inputs.new_tensor([-0.2, 0.1, 0.3])
        sequence = values.view(1, 3, 1).expand(inputs.shape[0], 3, 8)
        hidden = sequence[:, -1, :].unsqueeze(0)
        if self.lstm:
            cell = torch.full_like(hidden, 0.7)
            return sequence, (hidden, cell)
        return sequence, hidden


class _RecordingHead(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.inputs = []

    def forward(self, inputs):
        """Record the chosen state and return two synthetic logits."""
        self.inputs.append(inputs.detach().clone())
        return torch.stack((inputs[:, 0], -inputs[:, 0]), dim=1)


def test_rnn_selects_final_recurrent_values_not_first_or_mean():
    """Select final recurrent values while allowing equivalent final output."""
    models = _load_models()
    model = models.SimpleRNN(feature_dim=3)
    model.rnn = _FaithfulRecurrent()
    model.head = _RecordingHead()
    # Independent one-hot inputs leave the fixture's recurrent states realizable.
    inputs = torch.eye(3).unsqueeze(0).expand(2, 3, 3)

    logits = model(inputs)

    assert len(model.head.inputs) == 1
    torch.testing.assert_close(model.head.inputs[0], torch.full((2, 8), 0.3))
    torch.testing.assert_close(logits, torch.tensor([[0.3, -0.3], [0.3, -0.3]]))


def test_lstm_selects_final_hidden_values_not_first_mean_or_cell():
    """Select hidden values instead of history averages or cell state."""
    models = _load_models()
    model = models.SimpleLSTM(feature_dim=3)
    model.lstm = _FaithfulRecurrent(lstm=True)
    model.head = _RecordingHead()
    inputs = torch.eye(3).unsqueeze(0).expand(2, 3, 3)

    logits = model(inputs)

    assert len(model.head.inputs) == 1
    torch.testing.assert_close(model.head.inputs[0], torch.full((2, 8), 0.3))
    torch.testing.assert_close(logits, torch.tensor([[0.3, -0.3], [0.3, -0.3]]))
