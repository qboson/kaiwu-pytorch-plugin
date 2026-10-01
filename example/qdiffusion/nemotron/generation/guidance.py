"""Energy-guided native Nemotron candidate selection."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from kaiwu.torch_plugin import QDiffusion, QDiffusionConfig, SequenceTokenSpec

from .candidates import (
    GumbelNoiseGenerator,
    LogProbScorer,
    build_diverse_transfer_candidates,
)
from ..models.checkpoint import load_checkpoint
from ..models.energy import ContextualEnergyModel
from .proposal import ProposalDecision, ProposalStep


class ContextualEnergyHook:
    """Let BM override Native only when residual evidence is strong enough.

    Args:
        qdiffusion: Plain QDiffusion instance used only for energy scoring.
        num_candidates: Number of native and alternative candidates to score.
        proposal_temperature: Temperature for alternative-token sampling.
        proposal_noise_scale: Gumbel-noise scale for alternative sampling.
        candidate_mask_policy: Whether to score transferred positions only or
            the whole current block.
        energy_lambda: Weight applied to normalized BM energy.
        energy_mean: Checkpoint energy mean used for normalization.
        energy_std: Checkpoint energy standard deviation used for normalization.
        min_residual_gain: Minimum combined-score improvement for an override.
        min_energy_gain: Minimum BM-energy improvement for an override.
        guidance_end_fraction: Last fraction of a block in which BM may act.
    """

    def __init__(
        self,
        qdiffusion: QDiffusion,
        *,
        num_candidates: int = 4,
        proposal_temperature: float = 0.0,
        proposal_noise_scale: float = 1.0,
        candidate_mask_policy: str = "transfer",
        energy_lambda: float = 0.1,
        energy_mean: float = 0.0,
        energy_std: float = 1.0,
        min_residual_gain: float = 0.0,
        min_energy_gain: float = 0.0,
        guidance_end_fraction: float = 1.0,
    ) -> None:
        if num_candidates < 2:
            raise ValueError("result guidance requires at least two candidates")
        if energy_lambda < 0:
            raise ValueError("energy_lambda must be non-negative")
        if energy_std <= 0:
            raise ValueError("energy_std must be positive")
        if not 0.0 <= guidance_end_fraction <= 1.0:
            raise ValueError("guidance_end_fraction must be in [0, 1]")
        if candidate_mask_policy not in {"transfer", "full_block"}:
            raise ValueError("candidate_mask_policy must be 'transfer' or 'full_block'")
        self.qdiffusion = qdiffusion
        token_feature_provider = getattr(
            qdiffusion.proposal_model,
            "get_input_embeddings",
        )()
        if token_feature_provider is None:
            raise RuntimeError("Nemotron does not expose input token embeddings")
        self.token_feature_provider = token_feature_provider
        self.num_candidates = num_candidates
        self.energy_lambda = energy_lambda
        self.candidate_mask_policy = candidate_mask_policy
        self.energy_mean = energy_mean
        self.energy_std = energy_std
        self.min_residual_gain = min_residual_gain
        self.min_energy_gain = min_energy_gain
        self.guidance_end_fraction = guidance_end_fraction
        self.generator = GumbelNoiseGenerator(
            temperature=proposal_temperature,
            noise_scale=proposal_noise_scale,
        )
        self.proposal_scorer = LogProbScorer()
        self.stats: list[dict[str, Any]] = []

    @torch.no_grad()
    def __call__(self, step: ProposalStep) -> ProposalDecision | None:
        """Select a candidate when its combined score beats the native proposal.

        Args:
            step: Current native candidate-selection state.

        Returns:
            A replacement decision, or ``None`` to retain the native one.

        Raises:
            RuntimeError: If the native generation session did not capture
                contextual hidden states.
        """

        if step.hidden_states is None:
            raise RuntimeError(
                "hidden_states not captured; enable capture_hidden_states"
            )
        progress = step.step_index / step.block_tokens.shape[-1]
        if progress > self.guidance_end_fraction:
            return None

        raw_candidates = self.generator.generate_hybrid(step, self.num_candidates)
        candidates, diversity_stats = build_diverse_transfer_candidates(
            step,
            raw_candidates,
            self.num_candidates,
        )
        transfer_mask = step.native_decision.transfer_index.squeeze(0)
        noisy_tokens = step.block_tokens.expand(candidates.shape[0], -1)
        hidden_states = step.hidden_states.expand(candidates.shape[0], -1, -1)
        scoring_mask = (
            transfer_mask
            if self.candidate_mask_policy == "transfer"
            else torch.ones_like(transfer_mask)
        )
        expanded_mask = scoring_mask.unsqueeze(0).expand(candidates.shape[0], -1)
        noisy_features = self.token_feature_provider(noisy_tokens)
        candidate_features = self.token_feature_provider(candidates)
        # Contextual scoring is a model-level capability: the hook calls the
        # energy model directly with named context arguments.
        energies = self.qdiffusion.energy_model.score_conditioned(
            noisy_tokens=noisy_tokens,
            candidate_tokens=candidates,
            attention_mask=expanded_mask,
            hidden_states=hidden_states,
            noisy_features=noisy_features,
            candidate_features=candidate_features,
        ).reshape(-1)
        transfer_count = int(transfer_mask.sum().item())
        proposal_scores = self.proposal_scorer.score(step, candidates) / transfer_count
        normalized_energy = (energies - self.energy_mean) / self.energy_std
        residual_scores = proposal_scores - self.energy_lambda * normalized_energy

        override_lambdas = []
        for candidate_index in range(1, candidates.shape[0]):
            candidate_energy_gain = float(energies[0] - energies[candidate_index])
            if candidate_energy_gain <= 0:
                continue
            candidate_proposal_penalty = float(
                proposal_scores[0] - proposal_scores[candidate_index]
            )
            override_lambdas.append(
                max(candidate_proposal_penalty, 0.0)
                * self.energy_std
                / candidate_energy_gain
            )

        best_index = int(residual_scores.argmax().item())
        residual_gain = float(residual_scores[best_index] - residual_scores[0])
        energy_gain = float(energies[0] - energies[best_index])
        changed_transfer_tokens = int(
            (candidates[best_index, transfer_mask] != candidates[0, transfer_mask])
            .sum()
            .item()
        )
        selected = (
            best_index != 0
            and changed_transfer_tokens > 0
            and residual_gain >= self.min_residual_gain
            and energy_gain >= self.min_energy_gain
        )
        self.stats.append(
            {
                "block": step.block_index,
                "step": step.step_index,
                "nfe": step.nfe,
                "selected": selected,
                "selected_index": best_index,
                "residual_gain": residual_gain,
                "energy_gain": energy_gain,
                "native_energy": float(energies[0]),
                "best_energy": float(energies[best_index]),
                "native_proposal_logprob": float(proposal_scores[0]),
                "best_proposal_logprob": float(proposal_scores[best_index]),
                "actual_token_changes": (changed_transfer_tokens if selected else 0),
                "candidate_mask_policy": self.candidate_mask_policy,
                "min_energy_override_lambda": (
                    min(override_lambdas) if override_lambdas else None
                ),
                **diversity_stats,
            }
        )
        if not selected:
            return None
        return ProposalDecision(
            tokens=candidates[best_index].unsqueeze(0),
            transfer_index=step.native_decision.transfer_index.clone(),
        )

    def get_stats(self) -> list[dict[str, Any]]:
        """Return a copy of per-step candidate-selection statistics.

        Returns:
            Recorded native and BM scores for every guided selection step.
        """

        return list(self.stats)


def build_nemotron_qdiffusion(
    proposal_model: Any,
    energy_model: ContextualEnergyModel,
    *,
    num_candidates: int = 4,
    proposal_temperature: float = 0.0,
    proposal_noise_scale: float = 1.0,
) -> QDiffusion:
    """Build the ordinary QDiffusion instance used only for candidate scoring.

    Args:
        proposal_model: Loaded frozen Nemotron model.
        energy_model: Trained contextual BM energy model.
        num_candidates: Candidate count retained in QDiffusion configuration.
        proposal_temperature: Candidate sampling temperature.
        proposal_noise_scale: Candidate sampling Gumbel-noise scale.

    Returns:
        Plain QDiffusion scorer. Nemotron inference must use its native
        ``generate`` rather than QDiffusion's generic generation methods.
    """

    config = proposal_model.config
    token_spec = SequenceTokenSpec(
        mask_id=int(proposal_model.mask_token_id),
        pad_id=int(config.pad_token_id),
        bos_id=int(config.bos_token_id),
        eos_id=int(config.eos_token_id),
    )
    return QDiffusion(
        proposal_model=proposal_model,
        energy_model=energy_model,
        token_spec=token_spec,
        config=QDiffusionConfig(
            num_candidates=num_candidates,
            proposal_temperature=proposal_temperature,
            proposal_noise_scale=proposal_noise_scale,
        ),
        dtype=next(proposal_model.parameters()).dtype,
        device=None,
        freeze_proposal=True,
    )


def load_nemotron_guidance(
    proposal_model: Any,
    checkpoint_path: str | Path,
    *,
    sampler: Any | None = None,
    num_candidates: int = 4,
    energy_lambda: float | None = None,
    min_residual_gain: float = 0.0,
    min_energy_gain: float = 0.0,
    guidance_end_fraction: float = 1.0,
    proposal_temperature: float = 0.0,
    proposal_noise_scale: float = 1.0,
) -> tuple[QDiffusion, ContextualEnergyHook]:
    """Load a BM checkpoint and return its scorer plus native selector hook.

    Args:
        proposal_model: Loaded frozen Nemotron model.
        checkpoint_path: Contextual BM checkpoint to load.
        sampler: Optional sampler override for the loaded energy model.
        num_candidates: Number of candidates scored at each native step.
        energy_lambda: Optional residual-energy weight overriding checkpoint
            metadata.
        min_residual_gain: Minimum combined-score improvement for an override.
        min_energy_gain: Minimum BM-energy improvement for an override.
        guidance_end_fraction: Last fraction of a block in which BM may act.
        proposal_temperature: Alternative-candidate sampling temperature.
        proposal_noise_scale: Alternative-candidate Gumbel-noise scale.

    Returns:
        Plain QDiffusion scorer and its NativeGenerationSession selector.
    """

    device = next(proposal_model.parameters()).device
    energy_model, payload = load_checkpoint(
        checkpoint_path,
        device=device,
        sampler=sampler,
    )
    normalization = payload["energy_normalization"]
    qdiffusion = build_nemotron_qdiffusion(
        proposal_model,
        energy_model,
        num_candidates=num_candidates,
        proposal_temperature=proposal_temperature,
        proposal_noise_scale=proposal_noise_scale,
    )
    hook = ContextualEnergyHook(
        qdiffusion,
        num_candidates=num_candidates,
        proposal_temperature=proposal_temperature,
        proposal_noise_scale=proposal_noise_scale,
        candidate_mask_policy="transfer",
        energy_lambda=(
            float(payload["energy_lambda"])
            if energy_lambda is None
            else energy_lambda
        ),
        energy_mean=float(normalization["mean"]),
        energy_std=float(normalization["std"]),
        min_residual_gain=min_residual_gain,
        min_energy_gain=min_energy_gain,
        guidance_end_fraction=guidance_end_fraction,
    )
    return qdiffusion, hook
