"""QDEQ.get_last_stats must return a copy of the stats of the latest pass."""

import torch

from kaiwu.torch_plugin import QDEQ, QDEQConfig


class Contraction(torch.nn.Module):
    """Plain contraction ``z / 2 + x`` with a unique fixed point."""

    def forward(self, z, x):
        """Returns ``z / 2 + x``."""
        return 0.5 * z + x


class NoIterationQDEQ(QDEQ):
    """Reports a fixed, easily identified set of solver statistics."""

    def __init__(self):
        super().__init__(Contraction(), None, QDEQConfig())

    def solve_fixed_point(self, func, z0):
        """Returns the initial guess and a one-key statistics dict."""
        del func
        return z0, {"nstep": 3, "residual": 0.5}


def test_returned_mapping_is_a_copy():
    deq = NoIterationQDEQ().eval()
    deq(torch.ones(2, 3))

    stats = deq.get_last_stats()
    stats.clear()

    assert deq.get_last_stats()["forward_nstep"] == 3


def test_each_forward_drops_the_previous_backward_statistics():
    deq = NoIterationQDEQ().eval()
    deq._last_stats["backward_nstep"] = 7

    deq(torch.ones(2, 3))

    assert "backward_nstep" not in deq.get_last_stats()
