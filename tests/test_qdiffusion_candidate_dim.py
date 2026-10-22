"""The categorical sampler must keep the candidate set as the leading dimension."""

import torch

from kaiwu.torch_plugin._qdiffusion_sampling import stochastic_sample_from_categorical_n


def test_multiple_candidate_sets_lead_the_output():
    torch.manual_seed(0)
    logits = torch.randn(2, 3, 5)

    tokens, scores = stochastic_sample_from_categorical_n(logits, n=4)

    assert tokens.shape == (4, 2, 3)
    assert scores.shape == (4, 2, 3)
    assert tokens.dtype == torch.int64
    assert bool(((tokens >= 0) & (tokens < 5)).all())


def test_a_single_candidate_set_keeps_the_leading_dimension():
    torch.manual_seed(0)
    logits = torch.randn(2, 3, 5)

    tokens, scores = stochastic_sample_from_categorical_n(logits, n=1)

    assert tokens.shape == (1, 2, 3)
    assert scores.shape == (1, 2, 3)
