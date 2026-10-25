"""Contract test for the abstract bases of the QVAE stack."""

from __future__ import annotations

import pytest
import torch

from kaiwu.torch_plugin.qvae import AutoEncoderBase
from kaiwu.torch_plugin.qvae_dist_util import DistUtil, SmoothingDist


@pytest.mark.parametrize("method", ["pdf", "cdf", "sample", "log_pdf"])
def test_smoothing_dist_methods_are_abstract(method):
    distribution = SmoothingDist()

    with pytest.raises(NotImplementedError):
        if method == "sample":
            distribution.sample((2, 3))
        else:
            getattr(distribution, method)(torch.zeros(3))


def test_dist_util_sampling_and_entropy_are_abstract():
    utility = DistUtil()

    with pytest.raises(NotImplementedError):
        utility.reparameterize(True)
    with pytest.raises(NotImplementedError):
        utility.entropy()


def test_auto_encoder_base_forward_is_abstract():
    module = object.__new__(AutoEncoderBase)

    with pytest.raises(NotImplementedError):
        AutoEncoderBase.forward(module, torch.zeros(2, 2))
