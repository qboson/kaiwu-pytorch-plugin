"""Clipping must not hide parameter mutations from PyTorch autograd."""

import pytest
import torch

from kaiwu.torch_plugin import BoltzmannMachine, RestrictedBoltzmannMachine


@pytest.fixture(params=[BoltzmannMachine, RestrictedBoltzmannMachine])
def model(request):
    if request.param is BoltzmannMachine:
        return BoltzmannMachine(
            2, quadratic_coef=torch.tensor([[0., 2.], [0., 0.]]),
            linear_bias=torch.tensor([2., -3.]), device="cpu",
        )
    return RestrictedBoltzmannMachine(
        1, 1, quadratic_coef=torch.tensor([[2.]]),
        linear_bias=torch.tensor([2., -3.]), device="cpu",
    )


def test_clipping_between_forward_and_backward_is_detected(model):
    states = torch.tensor([[0.2, 0.7]], requires_grad=True)
    loss = model(states).sum()
    model.clip_parameters(h_range=(-0.5, 0.5), j_range=(-0.25, 0.25))
    # Previously .data bypassed version tracking and backward silently used
    # clipped weights to differentiate a forward computed with old weights.
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        loss.backward()


def test_clipping_preserves_parameters_gradients_and_optimizer_references(model):
    parameters = list(model.parameters())
    optimizer = torch.optim.SGD(parameters, lr=0.1)
    loss = model(torch.tensor([[0.2, 0.7]])).sum()
    loss.backward()
    gradients = [parameter.grad.clone() for parameter in parameters]
    optimizer.step()
    expected_bias = model.linear_bias.detach().clamp(-0.5, 0.5)
    expected_quadratic = model.quadratic_coef.detach().clamp(-0.25, 0.25)

    assert model.clip_parameters((-0.5, 0.5), (-0.25, 0.25)) is None
    for before, after, gradient, optimized in zip(
        parameters, model.parameters(), gradients, optimizer.param_groups[0]["params"]
    ):
        assert before is after is optimized
        assert after.is_leaf and after.requires_grad
        torch.testing.assert_close(after.grad, gradient)
    torch.testing.assert_close(model.linear_bias, expected_bias)
    torch.testing.assert_close(model.quadratic_coef, expected_quadratic)

    optimizer.zero_grad()
    model(torch.tensor([[0.6, 0.1]])).sum().backward()
    assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters())
