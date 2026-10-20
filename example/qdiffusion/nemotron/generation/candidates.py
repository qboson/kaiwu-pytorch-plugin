"""Candidate sampling and proposal-score utilities for Nemotron guidance."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .proposal import ProposalStep


def _sample_k_candidates(
    logits: torch.Tensor,
    num_candidates: int,
    temperature: float = 0.0,
    noise_scale: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Samples candidate token blocks from one block of proposal logits.

    Args:
        logits: Proposal logits shaped ``[block, vocab]``.
        num_candidates: Number of independent samples to draw.
        temperature: Sampling temperature; ``0.0`` picks the argmax of the
            Gumbel-perturbed log probabilities.
        noise_scale: Scale of the Gumbel noise added before sampling.

    Returns:
        Tuple ``(tokens, scores)`` shaped ``[num_candidates, block]`` with
        sampled token ids and their per-position scores.
    """
    expanded = logits.unsqueeze(0).expand(num_candidates, *logits.shape)
    gumbel = -torch.log(-torch.log(torch.rand_like(expanded) + 1e-8) + 1e-8)
    noisy = expanded + noise_scale * gumbel
    if temperature > 0:
        dist = torch.distributions.Categorical(logits=noisy.div(temperature))
        tokens = dist.sample()
        scores = dist.log_prob(tokens)
    else:
        scores, tokens = noisy.log_softmax(dim=-1).max(dim=-1)
    return tokens, scores


def _mask_logits(logits: torch.Tensor, mask_id: int) -> torch.Tensor:
    """Returns a copy of ``logits`` with the mask token suppressed.

    Args:
        logits: Proposal logits to filter.

        mask_id: Token id whose logits are set to ``-inf``.

    Returns:
        Cloned logits with ``mask_id`` suppressed.
    """
    logits = logits.clone()
    logits[..., mask_id] = -float("inf")
    return logits


def _gather_transfer_positions(
    candidate_tokens: torch.Tensor,
    block_tokens: torch.Tensor,
    transfer_index: torch.Tensor,
) -> torch.Tensor:
    """Keeps candidate tokens on transfer positions, native tokens elsewhere.

    Args:
        candidate_tokens: Candidate blocks shaped ``[candidate, block]``.

        block_tokens: Current block tokens used outside transfer positions.

        transfer_index: Boolean mask marking the positions to commit.

    Returns:
        Candidate blocks holding sampled tokens on transfer positions and
        copies of ``block_tokens`` everywhere else.
    """
    result = candidate_tokens.clone()
    result[:, ~transfer_index] = block_tokens[~transfer_index]
    return result


class LogProbScorer:
    """Score candidates under the native proposal distribution."""

    def score(
        self,
        step: ProposalStep,
        candidate_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """Sum native log probabilities over positions that will be committed.

        Args:
            step: Native proposal state containing logits and transfer positions.
            candidate_tokens: Candidate blocks with shape ``[candidate, block]``.

        Returns:
            Proposal log probability for each candidate.
        """

        log_probs = F.log_softmax(step.logits, dim=-1).squeeze(0)
        candidate_count = candidate_tokens.shape[0]
        log_probs = log_probs.unsqueeze(0).expand(candidate_count, -1, -1)
        transfer_mask = step.native_decision.transfer_index.squeeze(0)
        per_position = log_probs.gather(-1, candidate_tokens.unsqueeze(-1)).squeeze(-1)
        return (per_position * transfer_mask.float()).sum(dim=-1)


class GumbelNoiseGenerator:
    """Generate native and stochastic alternatives for one proposal step.

    Args:
        temperature: Temperature used when sampling alternative tokens.
        noise_scale: Gumbel-noise scale applied before sampling.
    """

    def __init__(
        self,
        temperature: float = 0.0,
        noise_scale: float = 1.0,
    ) -> None:
        self.temperature = temperature
        self.noise_scale = noise_scale

    def generate_hybrid(
        self,
        step: ProposalStep,
        num_candidates: int,
    ) -> torch.Tensor:
        """Generate candidates that always include the native proposal.

        Args:
            step: Current native proposal state.
            num_candidates: Total candidate count, including the native
                proposal.

        Returns:
            Native proposal followed by sampled alternative blocks.
        """

        if num_candidates == 1:
            return step.native_decision.tokens.squeeze(0).unsqueeze(0)
        logits = _mask_logits(step.logits.squeeze(0), step.mask_token_id)
        sampled_tokens, _ = _sample_k_candidates(
            logits,
            num_candidates - 1,
            temperature=self.temperature,
            noise_scale=self.noise_scale,
        )
        sampled_tokens = _gather_transfer_positions(
            sampled_tokens,
            step.block_tokens.squeeze(0),
            step.native_decision.transfer_index.squeeze(0),
        )
        native = step.native_decision.tokens.squeeze(0).unsqueeze(0)
        return torch.cat([native, sampled_tokens], dim=0)


def _same_transfer_tokens(
    first: torch.Tensor,
    second: torch.Tensor,
    transfer_mask: torch.Tensor,
) -> bool:
    """Checks whether two candidates agree on every transfer position.

    Args:
        first: First candidate block.

        second: Second candidate block.

        transfer_mask: Boolean mask of the positions to compare.

    Returns:
        ``True`` when both candidates hold identical tokens on all
        transfer positions.
    """
    return bool(torch.equal(first[transfer_mask], second[transfer_mask]))


def build_diverse_transfer_candidates(
    step: ProposalStep,
    candidates: torch.Tensor,
    max_candidates: int,
) -> tuple[torch.Tensor, dict[str, int]]:
    """Deduplicate candidates and fill missing alternatives from native logits.

    Candidate identity is defined only over positions that are about to be
    committed. Low-temperature sampling can duplicate the native proposal, so
    the strongest one-token alternatives fill those missing slots.

    Args:
        step: Current native proposal state.
        candidates: Sampled candidate blocks with the native proposal first.
        max_candidates: Required final candidate count.

    Returns:
        Distinct candidates and counts describing deduplication and fallback.

    Raises:
        ValueError: If candidates have an invalid shape or no transfer position.
    """

    if candidates.ndim != 2:
        raise ValueError("candidates must have shape [candidate, block]")
    if candidates.size(0) == 0:
        raise ValueError("at least the native candidate is required")
    transfer_mask = step.native_decision.transfer_index.squeeze(0)
    if not transfer_mask.any():
        raise ValueError("proposal step has no transfer positions")

    raw_count = int(candidates.size(0))
    native = candidates[0]
    native_collisions = sum(
        _same_transfer_tokens(candidate, native, transfer_mask)
        for candidate in candidates[1:]
    )
    unique: list[torch.Tensor] = []
    for candidate in candidates:
        if any(
            _same_transfer_tokens(candidate, kept, transfer_mask) for kept in unique
        ):
            continue
        unique.append(candidate)
        if len(unique) >= max_candidates:
            break
    sampled_unique_count = len(unique)

    transfer_positions = transfer_mask.nonzero(as_tuple=False).flatten()
    for position in transfer_positions.tolist():
        if len(unique) >= max_candidates:
            break
        position_logits = step.logits[0, position].clone()
        position_logits[step.mask_token_id] = -float("inf")
        position_logits[int(native[position])] = -float("inf")
        top_count = min(max(2 * max_candidates, 8), int(position_logits.numel()))
        values, token_ids = position_logits.topk(top_count)
        for value, token_id in zip(values, token_ids):
            if not torch.isfinite(value):
                continue
            alternative = native.clone()
            alternative[position] = token_id
            if any(
                _same_transfer_tokens(alternative, kept, transfer_mask)
                for kept in unique
            ):
                continue
            unique.append(alternative)
            if len(unique) >= max_candidates:
                break
        if len(unique) >= max_candidates:
            break

    stacked = torch.stack(unique)
    return stacked, {
        "raw_candidate_count": raw_count,
        "candidate_native_collisions": int(native_collisions),
        "candidate_duplicates_removed": raw_count - sampled_unique_count,
        "forced_candidate_count": len(unique) - sampled_unique_count,
        "unique_candidate_count": len(unique),
    }
