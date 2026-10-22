"""Pin the greedy path of sample_from_categorical."""

import torch

from kaiwu.torch_plugin._qdiffusion_sampling import sample_from_categorical

LOGITS = torch.tensor(
    [
        [0.0, 2.0, 1.0, -1.0, 0.5],
        [3.0, 0.0, 0.0, 0.0, 0.0],
    ]
)


def test_zero_temperature_returns_the_argmax() -> None:
    tokens, scores = sample_from_categorical(LOGITS, temperature=0.0)

    expected_scores, expected_tokens = LOGITS.log_softmax(dim=-1).max(dim=-1)
    torch.testing.assert_close(tokens, expected_tokens)
    torch.testing.assert_close(scores, expected_scores)


def test_scores_are_log_probabilities() -> None:
    tokens, scores = sample_from_categorical(LOGITS, temperature=1.0)
    log_probs = LOGITS.log_softmax(dim=-1)
    torch.testing.assert_close(scores, log_probs.gather(1, tokens.unsqueeze(1)).squeeze(1))
