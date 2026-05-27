# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Private sampling helpers shared by the public QDiffusion model."""

from __future__ import annotations

import torch


def topk_masking(
    scores: torch.Tensor,
    cutoff_len: torch.Tensor,
    stochastic: bool = False,
    temp: float = 1.0,
) -> torch.Tensor:
    """Selects the lowest-score positions used by skeptical remasking."""
    if stochastic:
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(scores) + 1e-8) + 1e-8)
        ranked_scores = scores + temp * gumbel_noise
    else:
        ranked_scores = scores
    sorted_scores = ranked_scores.sort(-1)[0]
    cutoff = sorted_scores.gather(dim=-1, index=cutoff_len)
    return ranked_scores < cutoff


def sample_from_categorical(
    logits: torch.Tensor,
    temperature: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Samples tokens from categorical logits."""
    if temperature:
        dist = torch.distributions.Categorical(logits=logits.div(temperature))
        tokens = dist.sample()
        scores = dist.log_prob(tokens)
    else:
        scores, tokens = logits.log_softmax(dim=-1).max(dim=-1)
    return tokens, scores


def stochastic_sample_from_categorical(
    logits: torch.Tensor,
    temperature: float = 1.0,
    noise_scale: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Applies Gumbel noise before categorical sampling."""
    gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-8) + 1e-8)
    noisy_logits = logits + noise_scale * gumbel_noise
    return sample_from_categorical(noisy_logits, temperature)


def stochastic_sample_from_categorical_n(
    logits: torch.Tensor,
    temperature: float = 1.0,
    noise_scale: float = 1.0,
    n: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Samples multiple noisy categorical candidates."""
    expanded_logits = logits.unsqueeze(0).expand(n, *logits.shape)
    gumbel_noise = -torch.log(
        -torch.log(torch.rand_like(expanded_logits) + 1e-8) + 1e-8
    )
    noisy_logits = expanded_logits + noise_scale * gumbel_noise
    return sample_from_categorical(noisy_logits, temperature)


def top_k_top_p_filtering(
    logits: torch.Tensor,
    top_k: int = 0,
    top_p: float = 0.95,
    filter_value: float = -float("Inf"),
) -> torch.Tensor:
    """Applies top-k and/or nucleus filtering to logits."""
    original_shape = logits.shape
    flat_logits = logits.reshape(-1, original_shape[-1])
    assert flat_logits.dim() == 2

    top_k = min(top_k, flat_logits.size(-1))
    if top_k > 0:
        indices_to_remove = (
            flat_logits < torch.topk(flat_logits, top_k, dim=1)[0][..., -1, None]
        )
        flat_logits[indices_to_remove] = filter_value

    sorted_logits, sorted_indices = torch.sort(flat_logits, descending=True)
    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    sorted_logits[sorted_indices_to_remove] = filter_value
    restored_logits = torch.gather(sorted_logits, 1, sorted_indices.argsort(-1))
    return restored_logits.reshape(original_shape)
