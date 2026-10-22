"""A top_k larger than the vocabulary must not filter anything."""

import torch

from kaiwu.torch_plugin._qdiffusion_sampling import top_k_top_p_filtering


def test_top_k_larger_than_vocab_keeps_everything():
    logits = torch.tensor([[1.0, -1.0, 0.5]])

    filtered = top_k_top_p_filtering(logits, top_k=99, top_p=1.0)

    torch.testing.assert_close(filtered, logits)
