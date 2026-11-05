"""QDEQ.jacobian_loss must validate the probe count."""

import pytest
import torch

from kaiwu.torch_plugin import QDEQ


@pytest.mark.parametrize("vecs", [0, -1, 1.5, True])
def test_invalid_probe_count_is_rejected(vecs) -> None:
    z = torch.randn(2, 3, dtype=torch.float64, requires_grad=True)
    with pytest.raises(ValueError, match="vecs must be a positive integer"):
        QDEQ.jacobian_loss(2.0 * z, z, vecs=vecs)


def test_valid_probe_count_returns_a_finite_estimate() -> None:
    z = torch.randn(2, 3, dtype=torch.float64, requires_grad=True)
    loss = QDEQ.jacobian_loss(2.0 * z, z, vecs=2)
    assert torch.isfinite(loss)
    assert loss.item() >= 0.0
