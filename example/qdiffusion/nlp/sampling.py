"""BM-guided reverse sampling for natural-language QDiffusion."""

from __future__ import annotations

from typing import Any

import torch


def _sample_categorical(
    weights: torch.Tensor,
    *,
    num_samples: int = 1,
) -> torch.Tensor:
    """Sample token IDs with the float32 exponential-race method.

    Args:
        weights: Non-negative categorical weights with shape
            ``(batch, sequence, vocabulary)``.
        num_samples: Number of independent candidates per sequence.

    Returns:
        Sampled token IDs with shape
        ``(batch, num_samples, sequence)``.

    Raises:
        ValueError: If ``num_samples`` is non-positive or ``weights`` does not
            have three dimensions.
    """

    if num_samples <= 0:
        raise ValueError("num_samples must be positive.")
    if weights.ndim != 3:
        raise ValueError("weights must have shape (batch, sequence, vocab).")
    batch_size, sequence_length, _ = weights.shape
    repeated_weights = weights.float().repeat(num_samples, 1, 1)
    exponential_noise = (
        1e-10
        - (torch.rand_like(repeated_weights) + 1e-10).log()
    )
    return (
        (repeated_weights / exponential_noise)
        .argmax(dim=-1)
        .view(num_samples, batch_size, sequence_length)
        .permute(1, 0, 2)
        .contiguous()
    )


class BMGuidedSampler:
    """Run reverse diffusion with BM-guided candidate selection.

    Args:
        proposal_model: Frozen MDLM proposal returning token log probabilities.
        energy_model: Conditioned BM sequence-energy model.
        mask_id: Diffusion mask-token ID.
        num_candidates: Number of proposal candidates scored at guided steps.
        energy_temperature: Temperature applied to negative BM energies.
        importance_start_t: Upper endpoint of the BM-guidance time window.
        importance_end_t: Lower endpoint of the BM-guidance time window.
        remask_ratio: Fraction of low-confidence revealed tokens eligible for
            remasking.
        eps: Positive terminal diffusion time.

    Attributes:
        last_stats: Statistics from the most recent reverse process.
    """

    def __init__(
        self,
        proposal_model: torch.nn.Module,
        *,
        energy_model: torch.nn.Module,
        mask_id: int,
        num_candidates: int = 2,
        energy_temperature: float = 1.0,
        importance_start_t: float = 1.0,
        importance_end_t: float = 0.0,
        remask_ratio: float = 0.1,
        eps: float = 1e-5,
    ) -> None:
        if num_candidates < 2:
            raise ValueError("BM guidance requires at least two candidates.")
        if energy_temperature <= 0:
            raise ValueError("energy_temperature must be positive.")
        if not 0.0 <= importance_end_t <= importance_start_t <= 1.0:
            raise ValueError(
                "Importance window must satisfy "
                "0 <= end <= start <= 1."
            )
        if not 0.0 <= remask_ratio < 1.0:
            raise ValueError("remask_ratio must be in [0, 1).")
        self.proposal_model = proposal_model
        self.energy_model = energy_model
        self.mask_id = int(mask_id)
        self.num_candidates = int(num_candidates)
        self.energy_temperature = float(energy_temperature)
        self.importance_start_t = float(importance_start_t)
        self.importance_end_t = float(importance_end_t)
        self.remask_ratio = float(remask_ratio)
        self.eps = float(eps)
        self.last_stats: dict[str, Any] = {}

    def _uses_guidance(self, timestep: float) -> bool:
        """Return whether BM guidance is active at a diffusion time."""

        return self.importance_end_t <= timestep <= self.importance_start_t

    def _select_x0(
        self,
        noisy_tokens: torch.Tensor,
        proposal_probs: torch.Tensor,
    ) -> torch.Tensor:
        """Sample proposal candidates and select one using BM energy.

        Args:
            noisy_tokens: Current diffusion state.
            proposal_probs: Proposal probabilities for clean tokens.

        Returns:
            One selected clean-token candidate per input sequence.
        """

        candidates = _sample_categorical(
            proposal_probs,
            num_samples=self.num_candidates,
        )
        energies = self.energy_model.score_candidates_conditioned(
            noisy_tokens=noisy_tokens,
            candidate_tokens=candidates,
            attention_mask=candidates.ne(self.mask_id),
        )
        # Lower BM energy means higher probability; sampling instead of argmin
        # retains diversity while still favoring low-energy candidates.
        logits = -energies.float() / self.energy_temperature
        logits -= logits.max(dim=-1, keepdim=True).values
        selected_indices = torch.multinomial(
            torch.softmax(logits, dim=-1),
            num_samples=1,
        ).squeeze(-1)
        return candidates[
            torch.arange(candidates.size(0), device=candidates.device),
            selected_indices,
        ]

    def _transition(
        self,
        tokens: torch.Tensor,
        timestep: float,
        step_size: float,
        x0_probs: torch.Tensor,
    ) -> torch.Tensor:
        """Apply one absorbing-mask reverse transition.

        Args:
            tokens: Current token state.
            timestep: Current continuous diffusion time.
            step_size: Distance to the next time point.
            x0_probs: Selected or unguided clean-token distribution.

        Returns:
            Token state after one reverse transition.
        """

        move_t = torch.as_tensor(
            timestep,
            device=x0_probs.device,
            dtype=torch.float32,
        )
        move_s = torch.clamp(move_t - step_size, min=self.eps)
        weights = x0_probs * (move_t - move_s)
        weights = weights.clone()
        weights[..., self.mask_id] = move_s
        proposed = _sample_categorical(weights).squeeze(-2)
        return torch.where(tokens.ne(self.mask_id), tokens, proposed)

    def _remask(
        self,
        tokens: torch.Tensor,
        proposal_probs: torch.Tensor,
        step: int,
        num_steps: int,
    ) -> torch.Tensor:
        """Remask low-confidence revealed tokens for later reconsideration.

        Args:
            tokens: Current partially denoised tokens.
            proposal_probs: Proposal probabilities used to reveal the tokens.
            step: Zero-based reverse-process step.
            num_steps: Total number of reverse-process steps.

        Returns:
            Tokens with a scheduled subset replaced by the mask token.
        """

        if self.remask_ratio == 0:
            return tokens
        rate = 1.0 - float(step) / float(num_steps)
        unmasked = tokens.ne(self.mask_id)
        count = unmasked.sum(dim=-1, keepdim=True).float()
        cutoff = (count * rate * self.remask_ratio).long()
        token_scores = proposal_probs.gather(
            -1,
            tokens.unsqueeze(-1),
        ).squeeze(-1)
        token_scores = token_scores.masked_fill(~unmasked, float("inf"))
        sorted_scores = token_scores.sort(dim=-1).values
        threshold = sorted_scores.gather(
            -1,
            (cutoff - 1).clamp_min(0),
        )
        remask = unmasked & (cutoff > 0) & (token_scores <= threshold)
        return tokens.masked_fill(remask, self.mask_id)

    @torch.no_grad()
    def sample(
        self,
        input_tokens: torch.Tensor,
        *,
        num_steps: int,
    ) -> torch.Tensor:
        """Run the reverse process and return fully denoised token IDs.

        Args:
            input_tokens: Initial token state, normally an all-mask tensor.
            num_steps: Number of reverse-diffusion steps.

        Returns:
            Fully denoised token IDs with the same shape as ``input_tokens``.

        Raises:
            ValueError: If ``num_steps`` is non-positive.
        """

        if num_steps <= 0:
            raise ValueError("num_steps must be positive.")
        tokens = input_tokens.clone()
        timesteps = torch.linspace(
            1.0,
            self.eps,
            num_steps + 1,
            device=tokens.device,
        )
        step_sizes = (timesteps[:-1] - timesteps[1:]).cpu().tolist()
        cached_x0_probs: torch.Tensor | None = None
        cached_proposal_probs: torch.Tensor | None = None
        proposal_forwards = 0
        guided_steps = 0

        for step in range(num_steps):
            timestep = float(timesteps[step])
            # Reuse the clean-token prediction until the token state changes.
            # This avoids redundant MDLM forwards on no-op transitions.
            if cached_x0_probs is None:
                cached_proposal_probs = self.proposal_model(tokens).exp()
                proposal_forwards += 1
                if self._uses_guidance(timestep):
                    selected = self._select_x0(
                        tokens,
                        cached_proposal_probs,
                    )
                    cached_x0_probs = torch.nn.functional.one_hot(
                        selected,
                        num_classes=cached_proposal_probs.size(-1),
                    ).to(cached_proposal_probs.dtype)
                    guided_steps += 1
                else:
                    cached_x0_probs = cached_proposal_probs

            next_tokens = self._transition(
                tokens,
                timestep,
                step_sizes[step],
                cached_x0_probs,
            )
            if not torch.equal(next_tokens, tokens):
                if cached_proposal_probs is not None:
                    next_tokens = self._remask(
                        next_tokens,
                        cached_proposal_probs,
                        step,
                        num_steps,
                    )
                cached_x0_probs = None
                cached_proposal_probs = None
            tokens = next_tokens

        # A deterministic final projection guarantees that no mask token
        # remains when the continuous schedule reaches epsilon.
        tokens = self.proposal_model(tokens).argmax(dim=-1)
        proposal_forwards += 1
        self.last_stats = {
            "num_steps": num_steps,
            "proposal_forwards": proposal_forwards,
            "guided_steps": guided_steps,
            "num_candidates": self.num_candidates,
            "remask_ratio": self.remask_ratio,
        }
        return tokens
