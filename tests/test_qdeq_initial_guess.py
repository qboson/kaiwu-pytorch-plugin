"""QDEQ must hand the caller's initial guess to solve_fixed_point without grad."""

import torch

from kaiwu.torch_plugin import QDEQ, QDEQConfig


class Contraction(torch.nn.Module):
    """Plain contraction ``z / 2 + x`` with a unique fixed point."""

    def forward(self, z, x):
        """Returns ``z / 2 + x``."""
        return 0.5 * z + x


class RecordingQDEQ(QDEQ):
    """Records what ``forward`` passes to the overridable solver."""

    def __init__(self, layer, config):
        super().__init__(layer, None, config)
        self.received_z0 = None
        self.received_grad_enabled = None

    def solve_fixed_point(self, func, z0):
        """Stores the initial guess and returns it unchanged."""
        self.received_z0 = z0.clone()
        self.received_grad_enabled = torch.is_grad_enabled()
        return z0, {"nstep": 0, "residual": 0.0}


def test_the_caller_initial_guess_reaches_the_solver():
    x = torch.ones(2, 4)
    z0 = torch.full((2, 4), 0.25)
    deq = RecordingQDEQ(Contraction(), QDEQConfig()).eval()

    deq(x, z0)

    torch.testing.assert_close(deq.received_z0, z0)


def test_the_default_initial_guess_is_zeros_shaped_like_the_input():
    x = torch.randn(3, 5)
    deq = RecordingQDEQ(Contraction(), QDEQConfig()).eval()

    deq(x)

    assert deq.received_z0.shape == x.shape
    assert torch.count_nonzero(deq.received_z0) == 0


def test_the_solver_runs_with_grad_disabled():
    deq = RecordingQDEQ(Contraction(), QDEQConfig()).eval()

    deq(torch.ones(2, 4, requires_grad=True))

    assert deq.received_grad_enabled is False
