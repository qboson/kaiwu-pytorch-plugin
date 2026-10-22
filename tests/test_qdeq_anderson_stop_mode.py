"""The Anderson solver must choose its stopping norm from stop_mode."""

import torch

from kaiwu.torch_plugin import QDEQConfig
from kaiwu.torch_plugin.qdeq import _anderson_solver


def _stats(stop_mode):
    constant = torch.ones(3, dtype=torch.float64)

    def func(value):
        return value + constant

    _, stats = _anderson_solver(
        func,
        torch.zeros(3, dtype=torch.float64),
        QDEQConfig(stop_mode=stop_mode),
        threshold=60,
        eps=0.1,
    )
    return stats


def test_relative_stopping_is_much_looser_than_absolute():
    relative = _stats("rel")
    absolute = _stats("abs")

    assert relative["nstep"] < 40
    assert absolute["nstep"] > 40
    assert absolute["residual"] > relative["residual"]
