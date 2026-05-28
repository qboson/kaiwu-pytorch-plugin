# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Private sampling helpers shared by the public QDiffusion model."""

from __future__ import annotations

import torch


# Skeptical-remasking helpers.

def topk_masking(
    scores: torch.Tensor,
    cutoff_len: torch.Tensor,
    stochastic: bool = False,
    temp: float = 1.0,
) -> torch.Tensor:
    """Selects the lowest-score positions used by skeptical remasking.

    Args:
        scores: Per-position scores used to rank editable token positions.
        cutoff_len: Per-sample cutoff lengths that determine how many
            positions remain masked.
        stochastic: Whether to perturb the ranking with Gumbel noise.
        temp: Noise temperature applied when ``stochastic`` is enabled.

    Returns:
        torch.Tensor: A boolean mask where ``True`` marks positions below the per-sample
        cutoff.
    """
    if stochastic:
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(scores) + 1e-8) + 1e-8)
        ranked_scores = scores + temp * gumbel_noise
    else:
        ranked_scores = scores
    sorted_scores = ranked_scores.sort(-1)[0]
    cutoff = sorted_scores.gather(dim=-1, index=cutoff_len)
    return ranked_scores < cutoff


# Categorical sampling helpers.

def sample_from_categorical(
    logits: torch.Tensor,
    temperature: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Samples tokens from categorical logits.

    Args:
        logits: Unnormalized categorical logits.
        temperature: Sampling temperature. A falsy value switches to greedy
            argmax decoding.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: A tuple ``(tokens, scores)`` where ``tokens`` contains sampled token
        ids and ``scores`` contains the associated log-probability-style
        scores.
    """
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
    """Applies Gumbel noise before categorical sampling.

    Args:
        logits: Unnormalized categorical logits.
        temperature: Sampling temperature forwarded to
            :func:`sample_from_categorical`.
        noise_scale: Multiplicative scale for the sampled Gumbel noise.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: A tuple ``(tokens, scores)`` sampled from the perturbed categorical
        distribution.
    """
    gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-8) + 1e-8)
    noisy_logits = logits + noise_scale * gumbel_noise
    return sample_from_categorical(noisy_logits, temperature)


def stochastic_sample_from_categorical_n(
    logits: torch.Tensor,
    temperature: float = 1.0,
    noise_scale: float = 1.0,
    n: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Samples multiple noisy categorical candidates.

    Args:
        logits: Unnormalized categorical logits shaped as
            ``[batch, seq_len, vocab]`` or similar.
        temperature: Sampling temperature forwarded to
            :func:`sample_from_categorical`.
        noise_scale: Multiplicative scale for the sampled Gumbel noise.
        n: Number of independent noisy candidate sets to draw.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: A tuple ``(tokens, scores)`` whose leading dimension indexes the
        sampled candidate set.
    """
    expanded_logits = logits.unsqueeze(0).expand(n, *logits.shape)
    gumbel_noise = -torch.log(
        -torch.log(torch.rand_like(expanded_logits) + 1e-8) + 1e-8
    )
    noisy_logits = expanded_logits + noise_scale * gumbel_noise
    return sample_from_categorical(noisy_logits, temperature)


# Logit filtering helpers.

def top_k_top_p_filtering(
    logits: torch.Tensor,
    top_k: int = 0,
    top_p: float = 0.95,
    filter_value: float = -float("Inf"),
) -> torch.Tensor:
    """Applies top-k and/or nucleus filtering to logits.

    Args:
        logits: Unnormalized categorical logits.
        top_k: Number of highest-logit entries to retain per row. A value of
            ``0`` disables top-k filtering.
        top_p: Nucleus-filtering threshold on cumulative probability mass.
        filter_value: Replacement value written into filtered logit entries.

    Returns:
        torch.Tensor: A tensor with the same shape as ``logits`` where filtered entries are
        replaced by ``filter_value``.
    """
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
