"""The Jacobian regulariser must be per-forward and train the layer parameters."""

import torch

from kaiwu.torch_plugin import QDEQ, QDEQConfig


class ScaledInjection(torch.nn.Module):
    """Elementwise scale plus injection, with a learnable scale."""

    def __init__(self, dim=4):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.full((dim,), 0.5))

    def forward(self, z, x):
        """Returns ``z * weight + x``."""
        return z * self.weight + x


class ShortCircuitQDEQ(QDEQ):
    """Skips the fixed-point solve so the layer and regulariser can be tested alone."""

    def solve_fixed_point(self, func, z0):
        """Returns the initial guess without iterating."""
        del func
        return z0, {"nstep": 0}


def _model(vecs):
    layer = ScaledInjection()
    config = QDEQConfig(jac_loss_vecs=vecs)
    return layer, ShortCircuitQDEQ(layer, None, config)


def test_training_mode_stores_a_finite_regulariser():
    _, deq = _model(vecs=2)

    deq(torch.ones(2, 4))

    assert deq.last_jac_loss is not None
    assert torch.isfinite(deq.last_jac_loss)
    assert deq.last_jac_loss.item() >= 0.0


def test_evaluation_clears_the_previous_regulariser():
    _, deq = _model(vecs=2)
    deq(torch.ones(2, 4))
    assert deq.last_jac_loss is not None

    deq.eval()
    with torch.no_grad():
        deq(torch.ones(2, 4))

    assert deq.last_jac_loss is None


def test_disabled_regulariser_stays_none():
    _, deq = _model(vecs=0)

    deq(torch.ones(2, 4))

    assert deq.last_jac_loss is None


def test_regulariser_gradient_reaches_the_layer_parameters():
    layer, deq = _model(vecs=2)
    deq(torch.ones(2, 4))

    deq.last_jac_loss.backward()

    assert layer.weight.grad is not None
    assert torch.isfinite(layer.weight.grad).all()
    assert torch.count_nonzero(layer.weight.grad) > 0
