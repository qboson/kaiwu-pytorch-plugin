"""discretize_visible_state must keep the input dtype."""

import torch

from kaiwu.torch_plugin.qdiffusion import EnergyModel


def test_float64_logits_stay_float64():
    logits = torch.randn(2, 3, dtype=torch.float64)

    conditioned = EnergyModel().discretize_visible_state(logits)

    assert conditioned.dtype == torch.float64
    torch.testing.assert_close(conditioned, torch.sigmoid(logits))
