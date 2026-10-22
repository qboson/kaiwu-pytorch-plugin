"""Pin the binarize option of RBM.get_hidden/get_visible."""

import torch

from kaiwu.torch_plugin import RestrictedBoltzmannMachine


def _model() -> RestrictedBoltzmannMachine:
    torch.manual_seed(0)
    model = RestrictedBoltzmannMachine(num_visible=3, num_hidden=2)
    with torch.no_grad():
        model.quadratic_coef.copy_(torch.randn_like(model.quadratic_coef))
        model.linear_bias.copy_(torch.randn_like(model.linear_bias))
    return model


def test_get_hidden_bernoulli_returns_binary_hidden_units() -> None:
    model = _model()
    visible = torch.rand(4, 3)

    full = model.get_hidden(visible, bernoulli=True)
    torch.testing.assert_close(full[:, :3], visible)
    hidden = full[:, 3:]
    assert torch.all((hidden == 0) | (hidden == 1))

    probs = model.get_hidden(visible, bernoulli=False)[:, 3:]
    expected = torch.sigmoid(
        visible @ model.quadratic_coef + model.linear_bias[3:]
    )
    torch.testing.assert_close(probs, expected)


def test_get_visible_bernoulli_returns_binary_visible_units() -> None:
    model = _model()
    hidden = torch.rand(4, 2)

    full = model.get_visible(hidden, bernoulli=True)
    torch.testing.assert_close(full[:, 3:], hidden)
    visible = full[:, :3]
    assert torch.all((visible == 0) | (visible == 1))
