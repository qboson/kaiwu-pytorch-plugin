"""Pin the truncated support of Exponential.sample."""

import torch

from kaiwu.torch_plugin.qvae_dist_util import Exponential


def test_samples_stay_inside_the_unit_interval() -> None:
    torch.manual_seed(0)
    samples = Exponential(1.0).sample((10000,))

    assert samples.min() >= 0.0
    assert samples.max() <= 1.0


def test_smaller_beta_still_respects_the_support() -> None:
    torch.manual_seed(1)
    samples = Exponential(0.1).sample((10000,))

    assert samples.min() >= 0.0
    assert samples.max() <= 1.0
