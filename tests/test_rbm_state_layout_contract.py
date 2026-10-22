"""Pin the state layout of RBM.get_hidden/get_visible."""

import torch

from kaiwu.torch_plugin import RestrictedBoltzmannMachine


def _model() -> RestrictedBoltzmannMachine:
    torch.manual_seed(0)
    model = RestrictedBoltzmannMachine(num_visible=3, num_hidden=2)
    with torch.no_grad():
        model.quadratic_coef.copy_(torch.randn_like(model.quadratic_coef))
        model.linear_bias.copy_(torch.randn_like(model.linear_bias))
    return model


def test_get_hidden_keeps_visible_prefix_and_uses_hidden_bias() -> None:
    model = _model()
    visible = torch.rand(4, 3)

    full = model.get_hidden(visible, bernoulli=False)

    torch.testing.assert_close(full[:, :3], visible)
    expected_hidden = torch.sigmoid(
        visible @ model.quadratic_coef + model.linear_bias[3:]
    )
    torch.testing.assert_close(full[:, 3:], expected_hidden)


def test_get_visible_keeps_hidden_suffix_and_uses_visible_bias() -> None:
    model = _model()
    hidden = torch.rand(4, 2)

    full = model.get_visible(hidden, bernoulli=False)

    torch.testing.assert_close(full[:, 3:], hidden)
    expected_visible = torch.sigmoid(
        hidden @ model.quadratic_coef.t() + model.linear_bias[:3]
    )
    torch.testing.assert_close(full[:, :3], expected_visible)
