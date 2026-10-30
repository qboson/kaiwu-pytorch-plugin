"""Categorical sampling helpers must validate temperature and candidate count."""

import pytest
import torch

from kaiwu.torch_plugin._qdiffusion_sampling import (
    sample_from_categorical,
    stochastic_sample_from_categorical_n,
)

LOGITS = torch.randn(2, 5)


@pytest.mark.parametrize("temperature", [-1.0, float("nan"), float("inf")])
def test_invalid_temperature_is_rejected(temperature) -> None:
    with pytest.raises(ValueError, match="temperature must be a finite non-negative"):
        sample_from_categorical(LOGITS, temperature)


def test_greedy_and_sampling_paths_still_work() -> None:
    greedy_tokens, greedy_scores = sample_from_categorical(LOGITS, 0.0)
    assert greedy_tokens.shape == (2,)
    assert greedy_scores.shape == (2,)

    tokens, _ = sample_from_categorical(LOGITS, 0.7)
    assert tokens.shape == (2,)


@pytest.mark.parametrize("n", [0, -2])
def test_non_positive_candidate_count_is_rejected(n) -> None:
    with pytest.raises(ValueError, match="n must be a positive integer"):
        stochastic_sample_from_categorical_n(LOGITS, 1.0, 1.0, n)


@pytest.mark.parametrize("n", [1.5, True])
def test_non_integer_candidate_count_is_rejected(n) -> None:
    with pytest.raises(TypeError, match="n must be a positive integer"):
        stochastic_sample_from_categorical_n(LOGITS, 1.0, 1.0, n)


def test_valid_candidate_count_still_samples() -> None:
    tokens, scores = stochastic_sample_from_categorical_n(LOGITS, 1.0, 1.0, 3)
    assert tokens.shape == (3, 2)
    assert scores.shape == (3, 2)
