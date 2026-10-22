"""A fresh QDEQ has no solver statistics and no regulariser value."""

import torch

from kaiwu.torch_plugin import QDEQ, QDEQConfig


class Contraction(torch.nn.Module):
    """Plain contraction with no sampler (solver is never called)."""

    def forward(self, z, x):
        """Returns ``z / 2 + x``."""
        return 0.5 * z + x


def test_fresh_model_reports_no_statistics():
    model = QDEQ(Contraction(), None, QDEQConfig())

    assert model.get_last_stats() == {}
    assert model.last_jac_loss is None
