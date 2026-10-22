"""Nucleus filtering must keep the smallest prefix above top_p."""

import torch

from kaiwu.torch_plugin._qdiffusion_sampling import top_k_top_p_filtering


def test_nucleus_filter_keeps_the_minimal_prefix():
    uniform = torch.zeros(1, 4)
    peaked = torch.tensor([[2.0, 1.0, 0.0, -1.0]])

    uniform_out = top_k_top_p_filtering(uniform, top_k=0, top_p=0.5)
    peaked_out = top_k_top_p_filtering(peaked, top_k=0, top_p=0.9)

    assert uniform_out.tolist() == [[0.0, 0.0, 0.0, float("-inf")]]
    assert peaked_out.tolist() == [[2.0, 1.0, 0.0, float("-inf")]]


def test_top_p_one_keeps_every_token():
    logits = torch.tensor([[2.0, 1.0, 0.0, -1.0]])

    assert top_k_top_p_filtering(logits, top_k=0, top_p=1.0).tolist() == logits.tolist()
