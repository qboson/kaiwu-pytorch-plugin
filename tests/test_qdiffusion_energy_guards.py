"""EnergyModel must fail loudly when its BM configuration is incomplete."""

import pytest
import torch

from kaiwu.torch_plugin.qdiffusion import EnergyModel


def test_sampling_without_configuration_is_rejected():
    with pytest.raises(RuntimeError, match="sampling requires"):
        EnergyModel().sample_hidden_state(torch.zeros(1, 2))


def test_sampling_without_a_sampler_is_rejected():
    model = EnergyModel(bm_num_visible=2, bm_num_hidden=3)

    with pytest.raises(RuntimeError, match="sampling requires"):
        model.sample_hidden_state(torch.zeros(1, 2))


def test_scoring_without_bm_num_visible_is_rejected():
    with pytest.raises(RuntimeError, match="visible-logit scoring"):
        EnergyModel().score_visible_logits(torch.zeros(1, 2))
