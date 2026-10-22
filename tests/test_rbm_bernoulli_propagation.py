"""bernoulli=True must sample binary states and keep the given layer."""

import torch

from kaiwu.torch_plugin import RestrictedBoltzmannMachine


def _model():
    torch.manual_seed(0)
    return RestrictedBoltzmannMachine(2, 3, device=torch.device("cpu"))


def test_get_hidden_samples_binary_hidden_states():
    model = _model()
    visible = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])

    states = model.get_hidden(visible, bernoulli=True)

    hidden = states[:, model.num_visible :]
    assert torch.isin(hidden, torch.tensor([0.0, 1.0])).all()
    torch.testing.assert_close(states[:, : model.num_visible], visible)


def test_get_visible_samples_binary_visible_states():
    model = _model()
    hidden = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])

    states = model.get_visible(hidden, bernoulli=True)

    visible = states[:, : model.num_visible]
    assert torch.isin(visible, torch.tensor([0.0, 1.0])).all()
    torch.testing.assert_close(states[:, model.num_visible :], hidden)
