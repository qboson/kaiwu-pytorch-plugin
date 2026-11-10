from __future__ import annotations

import os
import sys

import torch

src_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../src"))
sys.path.insert(0, src_root)

from kaiwu.torch_plugin.full_boltzmann_machine import BoltzmannMachine


def test_double_model_gibbs_sample_keeps_dtype() -> None:
    """A .double() BM must Gibbs-sample without a dtype crash."""
    bm = BoltzmannMachine(num_nodes=6).double()

    samples = bm.gibbs_sample(num_steps=1, num_sample=2)

    assert samples.dtype == torch.float64
    assert samples.shape == (2, 6)


def test_double_model_gibbs_sample_accepts_visible_states() -> None:
    """Both conditioning branches must run on a float64 model."""
    bm = BoltzmannMachine(num_nodes=6).double()
    visible = torch.bernoulli(torch.full((2, 3), 0.5, dtype=torch.float64))

    samples = bm.gibbs_sample(num_steps=1, s_visible=visible)

    assert samples.dtype == torch.float64
    torch.testing.assert_close(samples[:, :3], visible)


def test_float32_gibbs_sample_unchanged() -> None:
    """The default float32 pipeline keeps its dtype."""
    bm = BoltzmannMachine(num_nodes=6)

    samples = bm.gibbs_sample(num_steps=1, num_sample=2)

    assert samples.dtype == torch.float32
