"""sample_from_categorical must be greedy at temperature zero."""

import torch

from kaiwu.torch_plugin._qdiffusion_sampling import sample_from_categorical


def test_zero_temperature_returns_the_argmax_and_its_log_probability():
    logits = torch.tensor([[1.0, 3.0, 2.0]])

    tokens, scores = sample_from_categorical(logits, temperature=0.0)

    assert tokens.tolist() == [1]
    torch.testing.assert_close(scores, logits.log_softmax(dim=-1)[0, 1].unsqueeze(0))


def test_sampling_keeps_the_leading_shape():
    torch.manual_seed(0)
    logits = torch.randn(2, 3, 5)

    tokens, scores = sample_from_categorical(logits, temperature=1.0)

    assert tokens.shape == (2, 3)
    assert scores.shape == (2, 3)
    assert tokens.dtype == torch.int64
    assert bool(((tokens >= 0) & (tokens < 5)).all())
