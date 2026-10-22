"""RBM propagation must match the activation formulas exactly."""

import torch

from kaiwu.torch_plugin import RestrictedBoltzmannMachine


def _model():
    torch.manual_seed(0)
    model = RestrictedBoltzmannMachine(2, 3, device=torch.device("cpu"))
    with torch.no_grad():
        model.quadratic_coef.normal_(0.0, 0.5)
        model.linear_bias.normal_(0.0, 0.5)
    return model


def test_get_hidden_matches_the_activation_formula():
    model = _model()
    visible = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

    states = model.get_hidden(visible)

    expected = torch.sigmoid(
        visible @ model.quadratic_coef.detach() + model.linear_bias.detach()[model.num_visible :]
    )
    torch.testing.assert_close(states[:, : model.num_visible], visible)
    torch.testing.assert_close(states[:, model.num_visible :], expected)


def test_get_visible_matches_the_activation_formula():
    model = _model()
    hidden = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])

    states = model.get_visible(hidden)

    expected = torch.sigmoid(
        hidden @ model.quadratic_coef.detach().t() + model.linear_bias.detach()[: model.num_visible]
    )
    torch.testing.assert_close(states[:, : model.num_visible], expected)
    torch.testing.assert_close(states[:, model.num_visible :], hidden)
