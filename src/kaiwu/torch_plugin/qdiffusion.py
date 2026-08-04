# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Public QDiffusion module for generic discrete-sequence generation."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np
import torch
from torch import nn

from .full_boltzmann_machine import BoltzmannMachine
from ._qdiffusion_sampling import (
    stochastic_sample_from_categorical,
    stochastic_sample_from_categorical_n,
    top_k_top_p_filtering,
    topk_masking,
)

__all__ = ["EnergyModel", "QDiffusion", "QDiffusionConfig"]


@dataclass(frozen=True)
class SequenceTokenSpec:
    """Special-token metadata required by generic QDiffusion logic.

    Attributes:
        mask_id: Token id used as the diffusion noise token.
        pad_id: Token id used for sequence padding.
        bos_id: Token id used for beginning-of-sequence markers.
        eos_id: Token id used for end-of-sequence markers.
        x_id: Optional token id reserved by some backbones and excluded from
            proposal sampling.
        tokenizer: Optional tokenization helper used for encoding and decoding.
    """

    mask_id: int
    pad_id: int
    bos_id: int
    eos_id: int
    x_id: int | None = None
    tokenizer: Any | None = None


@dataclass
class QDiffusionConfig:
    """Configuration for energy-guided discrete generation.

    Attributes:

        num_diffusion_timesteps: Number of discrete noising steps used by the
            training objective.

        use_coupled_sampling: Whether to use the coupled corruption variant.

        num_candidates: Number of proposal candidates sampled at each decode step.

        proposal_temperature: Temperature used for proposal-side sampling.

        proposal_noise_scale: Gumbel noise scale used during proposal sampling.

        energy_temperature: Temperature used when converting energies into
            reranking weights.

        disable_resample: Whether to disable repetition-collapse resampling.

        resample_ratio: Frequency threshold that triggers resampling.

        resample_top_p: Top-p cutoff used during resampling.

        decoding_strategy: Skeptical-remasking strategy string.
    """

    num_diffusion_timesteps: int = 500
    use_coupled_sampling: bool = False
    num_candidates: int = 1
    proposal_temperature: float = 0.0
    proposal_noise_scale: float = 1.0
    energy_temperature: float = 1.0
    disable_resample: bool = False
    resample_ratio: float = 0.25
    resample_top_p: float = 0.95
    decoding_strategy: str = "reparam-uncond-deterministic-linear"


class EnergyModel(nn.Module):
    """Energy-side model interface used by QDiffusion candidate scoring."""

    def __init__(
        self,
        bm_num_visible: int | None = None,
        bm_num_hidden: int | None = None,
        sampler: Any | None = None,
    ) -> None:
        super().__init__()
        self.bm_num_visible = bm_num_visible
        self.bm_num_hidden = bm_num_hidden
        self.sampler = sampler
        self._last_stats: dict[str, torch.Tensor] = {}
        if bm_num_visible is not None and bm_num_hidden is not None:
            self.energy_bm = BoltzmannMachine(num_nodes=bm_num_visible + bm_num_hidden)

    def forward(
        self,
        noisy_tokens: torch.Tensor,
        candidate_tokens: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor | None:
        """Runs conditioned energy scoring through the PyTorch module API."""
        return self.score_conditioned(
            noisy_tokens=noisy_tokens,
            candidate_tokens=candidate_tokens,
            attention_mask=attention_mask,
        )

    def score_conditioned(
        self,
        noisy_tokens: torch.Tensor,
        candidate_tokens: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor | None:
        """Scores candidates conditioned on noisy tokens."""
        del noisy_tokens, candidate_tokens, attention_mask
        raise NotImplementedError(
            "EnergyModel subclasses must implement score_conditioned()."
        )

    def discretize_visible_state(self, visible_logits: torch.Tensor) -> torch.Tensor:
        """Converts visible logits into normalized BM visible conditions."""
        return torch.sigmoid(visible_logits)

    def sample_hidden_state(
        self,
        visible_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Samples BM hidden states for each visible assignment."""
        if not hasattr(self, "energy_bm") or self.sampler is None:
            raise RuntimeError(
                "BM hidden-state sampling requires bm_num_visible, "
                "bm_num_hidden, and sampler."
            )
        batched_states = []
        split_sizes = []
        for sample_index in range(visible_state.size(0)):
            sampled_states = self.energy_bm.condition_sample(
                self.sampler,
                visible_state[sample_index : sample_index + 1],
                dtype=visible_state.dtype,
            )
            batched_states.append(sampled_states)
            split_sizes.append(sampled_states.size(0))
        return torch.cat(batched_states, dim=0), torch.tensor(
            split_sizes,
            device=visible_state.device,
            dtype=torch.long,
        )

    def _set_last_stats(
        self,
        *,
        visible_state: torch.Tensor,
        hidden_state: torch.Tensor,
    ) -> None:
        self._last_stats = {
            "sampling_mode": torch.tensor(
                1.0,
                dtype=visible_state.dtype,
                device=visible_state.device,
            ),
            "visible_on_ratio": visible_state.detach().mean(),
            "hidden_on_ratio": hidden_state.detach().mean(),
        }

    def get_last_stats(self) -> dict[str, torch.Tensor]:
        """Returns lightweight sampler diagnostics from the last score call."""
        return dict(self._last_stats)

    def score_visible_logits(self, visible_logits: torch.Tensor) -> torch.Tensor:
        """Scores visible logits under the conditioned BM energy model."""
        if self.bm_num_visible is None:
            raise RuntimeError("BM visible-logit scoring requires bm_num_visible.")
        visible_state = self.discretize_visible_state(visible_logits)
        full_states, split_sizes = self.sample_hidden_state(visible_state)
        hidden_state = full_states[:, self.bm_num_visible :]
        self._set_last_stats(
            visible_state=full_states[:, : self.bm_num_visible],
            hidden_state=hidden_state,
        )
        flat_energy = self.energy_bm(full_states).unsqueeze(-1)
        split_energy = torch.split(flat_energy, split_sizes.tolist())
        return torch.stack([energy.mean(dim=0) for energy in split_energy], dim=0)


class QDiffusion(nn.Module):
    """Energy-guided discrete diffusion wrapper over generic sequence backbones.
    Initializes a QDiffusion model.

    The class combines two backbone roles:

    - a proposal model that predicts token logits for the current noisy state
    - an energy model that reranks candidate reconstructions

    It exposes both training-oriented APIs such as ``objective`` and
    decoding-oriented APIs such as ``initialize_state``, ``step``, and
    ``generate``.

    Args:
        proposal_model: Backbone used to predict proposal logits.

        energy_model: Energy-side model used to encode and score candidates.

        token_spec: Special-token metadata required by the generator.

        config: Optional generation/training configuration.

        dtype: Floating point dtype tracked by the wrapper.

        device: Optional target device. When omitted, infer from parameters.

        freeze_proposal: Whether to freeze proposal model parameters.
    """

    def __init__(
        self,
        proposal_model: nn.Module,
        energy_model: EnergyModel,
        token_spec: SequenceTokenSpec,
        config: QDiffusionConfig | None = None,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str | None = None,
        freeze_proposal: bool = True,
    ) -> None:
        """


        """
        super().__init__()
        self.proposal_model = proposal_model
        self.energy_model = energy_model
        self.token_spec = token_spec
        self.config = config or QDiffusionConfig()
        self.dtype = dtype

        if freeze_proposal:
            self.proposal_model.eval()
            for parameter in self.proposal_model.parameters():
                parameter.requires_grad = False

        self.tokenizer = token_spec.tokenizer
        self.mask_id = token_spec.mask_id
        self.pad_id = token_spec.pad_id
        self.bos_id = token_spec.bos_id
        self.eos_id = token_spec.eos_id
        self.x_id = token_spec.x_id
        self.softplus = nn.Softplus()

        if device is None:
            try:
                self.device = next(self.parameters()).device
            except StopIteration:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)
            self.to(device=self.device, dtype=self.dtype)

    def to(self, *args: Any, **kwargs: Any) -> QDiffusion:
        """Moves the module and refreshes cached device/dtype metadata.

        Args:
            ``*args``: Positional arguments forwarded to ``nn.Module.to``.
            ``**kwargs``: Keyword arguments forwarded to ``nn.Module.to``.

        Returns:
            QDiffusion: The moved module instance.
        """
        module = super().to(*args, **kwargs)
        try:
            self.device = next(self.parameters()).device
            self.dtype = next(self.parameters()).dtype
        except StopIteration:
            self.device = torch.device("cpu")
        return module

    def forward(self, noisy_tokens: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Runs the proposal model on the current noisy state.

        Args:
            noisy_tokens: Current noisy token tensor.
            ``**kwargs``: Additional keyword arguments forwarded to the proposal model.

        Returns:
            torch.Tensor: Proposal logits over the token vocabulary.

        Raises:
            TypeError: If the proposal model does not implement ``forward``.
        """
        if hasattr(self.proposal_model, "forward"):
            return self.proposal_model.forward(noisy_tokens, **kwargs)
        raise TypeError("proposal_model must implement forward().")

    def proposal(self, noisy_tokens: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Semantic alias around ``forward`` for proposal-side calls.

        Args:
            noisy_tokens: Current noisy token tensor.
            ``**kwargs``: Additional keyword arguments forwarded to the proposal model.

        Returns:
            torch.Tensor: Proposal logits over the token vocabulary.
        """
        return self.forward(noisy_tokens, **kwargs)

    def energy(
        self,
        noisy_tokens: torch.Tensor,
        candidate_tokens: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Scores candidate reconstructions conditioned on the noisy state.

        Args:
            noisy_tokens: Noisy token tensor used as conditioning input.

            candidate_tokens: Candidate clean token tensor to score.

            attention_mask: Optional attention mask for the energy model.

        Returns:
            torch.Tensor: A tensor of scalar energies with shape ``[batch, 1]``.
        """
        if attention_mask is None:
            attention_mask = candidate_tokens.ne(self.pad_id)

        return self.energy_model.score_conditioned(
            noisy_tokens=noisy_tokens,
            candidate_tokens=candidate_tokens,
            attention_mask=attention_mask,
        )

    def objective(
        self, batch: dict[str, torch.Tensor], weighting: str = "constant"
    ) -> dict[str, torch.Tensor]:
        """Builds the one-step training objective used by an external loop.

        Args:
            batch: Batch dictionary containing at least ``batch["targets"]``.
            weighting: Per-sample timestep weighting mode.

        Returns:
            dict[str, torch.Tensor]: A dictionary containing proposal logits,
            supervision masks, loss weights, and the EBM objective term.
        """
        target = batch["targets"]
        first_timestep, second_timestep = torch.randint(
            1,
            self.config.num_diffusion_timesteps + 1,
            (2 * target.size(0),),
            device=target.device,
        ).chunk(2)

        if self.config.use_coupled_sampling:
            sample_outputs = self._sample_coupled(
                target,
                first_timestep,
                second_timestep,
                self.get_non_special_symbol_mask(target),
            )
            target = target.repeat(2, 1)
        else:
            sample_outputs = self._sample(
                target,
                first_timestep,
                self.get_non_special_symbol_mask(target),
            )

        noisy_tokens = sample_outputs["x_t"]
        timesteps = sample_outputs["t"]
        loss_mask = sample_outputs["loss_mask"]

        with torch.no_grad():
            logits = self.forward(noisy_tokens).detach()

        negative_tokens, _ = self._sample_candidates(logits, self.config.num_candidates)
        positive_energy = self.energy(noisy_tokens, target, target.ne(self.pad_id))
        positive_stats = self._collect_energy_model_stats()
        negative_energy = self._score_candidates(noisy_tokens, negative_tokens).mean(
            dim=1, keepdim=True
        )
        negative_stats = self._collect_energy_model_stats()

        energy_objective = (
            self.softplus(positive_energy)
            + self.softplus(-negative_energy)
        )
        weight = self._compute_loss_weight(timesteps, weighting)
        outputs = {
            "logits": logits,
            "targets": target,
            "loss_mask": loss_mask,
            "weight": weight,
            "energy_objective": energy_objective,
            "positive_energy_mean": positive_energy.mean().detach(),
            "negative_energy_mean": negative_energy.mean().detach(),
        }
        for prefix, stats in (
            ("positive", positive_stats),
            ("negative", negative_stats),
        ):
            for key, value in stats.items():
                outputs[f"{prefix}_{key}"] = value.detach()
        return outputs

    def initialize_state(
        self,
        input_tokens: torch.Tensor,
        partial_masks: torch.Tensor | None = None,
        max_steps: int = 500,
        temperature: float = 1.0,
    ) -> dict[str, Any]:
        """Creates the initial decoding state for an external generation loop.

        Args:
            input_tokens: Initial token tensor.
            partial_masks: Optional boolean mask of fixed positions.
            max_steps: Planned number of decode iterations.
            temperature: Sampling temperature stored in the state payload.

        Returns:
            dict[str, Any]: A mutable state dictionary suitable for repeated ``step`` calls.
        """
        output_tokens, output_scores = self._initialize_output_tokens(
            input_tokens, partial_masks=partial_masks
        )
        return {
            "output_tokens": output_tokens,
            "output_scores": output_scores,
            "output_masks": self.get_non_special_symbol_mask(
                output_tokens, partial_masks=partial_masks
            ),
            "step": 0,
            "max_steps": max_steps,
            "history": [output_tokens.clone()],
            "temperature": temperature,
            "partial_masks": partial_masks,
        }

    def step(
        self,
        state: dict[str, Any],
        partial_masks: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        """Runs one denoising/reranking step and returns updated state.

        Args:
            state: Current decode state created by ``initialize_state``.
            partial_masks: Optional boolean mask of fixed positions.

        Returns:
            dict[str, Any]: The updated decode state after one iteration.
        """
        partial_masks = (
            partial_masks if partial_masks is not None else state.get("partial_masks")
        )
        step_outputs = self._decode_step(state, partial_masks=partial_masks)

        editable_token_mask = self.get_non_special_symbol_mask(
            state["output_tokens"], partial_masks=partial_masks
        )
        output_masks, result_tokens, result_scores = self._reparam_decoding(
            output_tokens=state["output_tokens"].clone(),
            output_scores=state["output_scores"].clone(),
            step_tokens=step_outputs["output_tokens"].clone(),
            step_scores=step_outputs["output_scores"].clone(),
            decoding_strategy=self.config.decoding_strategy,
            still_noisy_mask=state["output_masks"],
            editable_token_mask=editable_token_mask,
            t=state["step"] + 1,
            max_step=state["max_steps"],
            noise=self.mask_id,
        )

        new_state = dict(state)
        new_state.update(
            output_tokens=result_tokens,
            output_scores=result_scores,
            output_masks=output_masks,
            step=state["step"] + 1,
            history=step_outputs["history"],
            partial_masks=partial_masks,
        )
        return new_state

    def generate(
        self,
        input_tokens: torch.Tensor,
        *,
        max_steps: int = 500,
        partial_masks: torch.Tensor | None = None,
        temperature: float = 1.0,
        return_state: bool = False,
    ) -> torch.Tensor | dict[str, Any]:
        """Runs a complete iterative decoding loop inside the core class.

        Args:
            input_tokens: Initial token tensor.
            max_steps: Number of decode iterations to run.
            partial_masks: Optional boolean mask of fixed positions.
            temperature: Sampling temperature stored in the decode state.
            return_state: Whether to return the full final state dictionary.

        Returns:
            torch.Tensor | dict[str, Any]: Either the final token tensor or the full decode state.
        """
        state = self.initialize_state(
            input_tokens=input_tokens,
            partial_masks=partial_masks,
            max_steps=max_steps,
            temperature=temperature,
        )
        for _ in range(max_steps):
            state = self.step(state, partial_masks=partial_masks)

        if return_state:
            return state
        return state["output_tokens"]

    # Internal decode and training helpers follow. These stay in the model
    # class for now, but they are not part of the intended public surface.

    def get_non_special_symbol_mask(
        self, output_tokens: torch.Tensor, partial_masks: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Returns a boolean mask of editable non-special-token positions.

        Args:
            output_tokens: Token tensor to inspect.
            partial_masks: Optional boolean mask of positions that should remain fixed.

        Returns:
            torch.Tensor: A boolean mask where ``True`` marks editable non-special positions.
        """
        editable_token_mask = (
            output_tokens.ne(self.pad_id)
            & output_tokens.ne(self.bos_id)
            & output_tokens.ne(self.eos_id)
        )
        if partial_masks is not None:
            editable_token_mask &= ~partial_masks
        return editable_token_mask

    def _collect_energy_model_stats(self) -> dict[str, torch.Tensor]:
        """Collects optional energy-model diagnostics from the last score call."""
        get_last_stats = getattr(self.energy_model, "get_last_stats", None)
        if not callable(get_last_stats):
            return {}
        return get_last_stats()

    def _initialize_output_tokens(
        self, input_tokens: torch.Tensor, partial_masks: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Builds the initial fully masked token state for decoding.

        Args:
            input_tokens: Initial token tensor.
            partial_masks: Optional boolean mask of fixed positions.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple
            ``(output_tokens, output_scores)`` for the initial decode state.
        """
        output_mask = self.get_non_special_symbol_mask(
            input_tokens, partial_masks=partial_masks
        )
        output_tokens = input_tokens.masked_fill(output_mask, self.mask_id)
        output_scores = torch.zeros_like(output_tokens, dtype=torch.float)
        return output_tokens, output_scores

    def _sample(
        self,
        clean_tokens: torch.Tensor,
        first_timestep: torch.Tensor,
        maskable_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Applies one-step discrete corruption for training.

        Args:
            clean_tokens: Clean target tokens.
            first_timestep: Sampled diffusion timesteps.
            maskable_mask: Boolean mask of positions eligible for masking.

        Returns:
            dict[str, torch.Tensor]: A dictionary containing corrupted tokens,
            timesteps, and loss mask.
        """
        noise = torch.rand_like(clean_tokens, dtype=torch.float)
        first_timestep_mask = (
            noise < (first_timestep / self.config.num_diffusion_timesteps)[:, None]
        ) & maskable_mask
        noisy_tokens = clean_tokens.masked_fill(first_timestep_mask, self.mask_id)
        return {
            "x_t": noisy_tokens,
            "t": first_timestep,
            "loss_mask": first_timestep_mask,
        }

    def _sample_coupled(
        self,
        clean_tokens: torch.Tensor,
        first_timestep: torch.Tensor,
        second_timestep: torch.Tensor,
        maskable_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Applies the coupled corruption variant used by RDM-style training.

        Args:
            clean_tokens: Clean target tokens.
            first_timestep: First sampled timestep tensor.
            second_timestep: Second sampled timestep tensor.
            maskable_mask: Boolean mask of positions eligible for masking.

        Returns:
            dict[str, torch.Tensor]: A dictionary containing paired
            corruptions, timesteps, and loss mask.
        """
        same_timestep_mask = first_timestep == second_timestep
        first_timestep, second_timestep = (
            torch.maximum(first_timestep, second_timestep).float(),
            torch.minimum(first_timestep, second_timestep).float(),
        )

        noise = torch.rand_like(clean_tokens, dtype=torch.float)
        first_timestep_mask = (
            noise < (first_timestep / self.config.num_diffusion_timesteps)[:, None]
        ) & maskable_mask
        first_noisy_tokens = clean_tokens.masked_fill(first_timestep_mask, self.mask_id)

        noise = torch.rand_like(clean_tokens, dtype=torch.float)
        second_timestep_mask = first_timestep_mask & (
            noise > ((first_timestep - second_timestep) / first_timestep)[:, None]
        )
        noise = torch.rand_like(clean_tokens[same_timestep_mask], dtype=torch.float)
        second_timestep_mask[same_timestep_mask] = (
            noise
            < (
                first_timestep[same_timestep_mask] / self.config.num_diffusion_timesteps
            )[:, None]
        ) & maskable_mask[same_timestep_mask]
        second_noisy_tokens = clean_tokens.masked_fill(
            second_timestep_mask, self.mask_id
        )

        return {
            "x_t": torch.cat([first_noisy_tokens, second_noisy_tokens], dim=0),
            "t": torch.cat([first_timestep, second_timestep]),
            "loss_mask": torch.cat([first_timestep_mask, second_timestep_mask], dim=0),
        }

    def _compute_loss_weight(
        self, timesteps: torch.Tensor, weighting: str
    ) -> torch.Tensor:
        """Converts sampled timesteps into per-sample loss weights.

        Args:
            timesteps: Sampled diffusion timesteps.
            weighting: Weighting strategy name.

        Returns:
            torch.Tensor: A column vector of normalized per-sample weights.
        """
        num_timesteps = self.config.num_diffusion_timesteps
        weight = {
            "linear": num_timesteps - (timesteps - 1),
            "constant": num_timesteps * torch.ones_like(timesteps),
        }[weighting]
        return weight[:, None].float() / num_timesteps

    def _mask_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Suppresses special-token logits before categorical sampling.

        Args:
            logits: Raw proposal logits.

        Returns:
            torch.Tensor: A cloned logits tensor with special-token entries masked out.
        """
        logits = logits.clone()
        logits[..., self.mask_id] = -math.inf
        if self.x_id is not None:
            logits[..., self.x_id] = -math.inf
        logits[..., self.pad_id] = -math.inf
        logits[..., self.bos_id] = -math.inf
        logits[..., self.eos_id] = -math.inf
        return logits

    def _reshape_candidates(
        self, tensor: torch.Tensor, batch_size: int, num_candidates: int
    ) -> torch.Tensor:
        """Normalizes sampled candidate tensors to ``[batch, k, seq_len]``.

        Args:
            tensor: Candidate tensor in one of the supported layouts.
            batch_size: Expected batch size.
            num_candidates: Expected candidate count.

        Returns:
            torch.Tensor: A candidate tensor shaped as ``[batch, num_candidates, seq_len]``.

        Raises:
            ValueError: If the incoming tensor shape is not recognized.
        """
        if tensor.shape[0] == num_candidates and tensor.shape[1] == batch_size:
            return tensor.permute(1, 0, 2).contiguous()
        if tensor.shape[0] == batch_size and tensor.shape[1] == num_candidates:
            return tensor.contiguous()
        if tensor.shape[0] == batch_size * num_candidates:
            return tensor.view(batch_size, num_candidates, -1)
        raise ValueError(f"Unexpected candidate tensor shape {tuple(tensor.shape)}.")

    def _sample_candidates(
        self, logits: torch.Tensor, num_candidates: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Samples proposal candidates from logits.

        Args:
            logits: Proposal logits for the current decode step.
            num_candidates: Number of candidates to sample per sequence.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple ``(tokens, scores)`` shaped as
            ``[batch, num_candidates, seq_len]``.
        """
        samples, scores = stochastic_sample_from_categorical_n(
            logits,
            temperature=self.config.proposal_temperature,
            noise_scale=self.config.proposal_noise_scale,
            n=num_candidates,
        )
        batch_size = logits.size(0)
        return (
            self._reshape_candidates(samples, batch_size, num_candidates),
            self._reshape_candidates(scores, batch_size, num_candidates),
        )

    def _score_candidates(
        self, noisy_tokens: torch.Tensor, candidate_tokens: torch.Tensor
    ) -> torch.Tensor:
        """Scores a batch of candidate reconstructions with the energy model.

        Args:
            noisy_tokens: Current noisy token tensor.
            candidate_tokens: Candidate reconstructions shaped as
                ``[batch, num_candidates, seq_len]``.

        Returns:
            torch.Tensor: Candidate energies shaped as ``[batch, num_candidates]``.
        """
        batch_size, num_candidates, seq_len = candidate_tokens.shape
        flat_noisy_tokens = (
            noisy_tokens.unsqueeze(1)
            .expand(-1, num_candidates, -1)
            .reshape(batch_size * num_candidates, seq_len)
        )
        flat_candidate_tokens = candidate_tokens.reshape(
            batch_size * num_candidates, seq_len
        )
        flat_attention_mask = flat_candidate_tokens.ne(self.pad_id)
        energy = self.energy(
            flat_noisy_tokens, flat_candidate_tokens, flat_attention_mask
        )
        return energy.view(batch_size, num_candidates)

    def _select_candidates(
        self,
        noisy_tokens: torch.Tensor,
        candidate_tokens: torch.Tensor,
        candidate_scores: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Selects one candidate per batch element by energy-based reranking.

        Args:
            noisy_tokens: Current noisy token tensor.
            candidate_tokens: Candidate reconstructions.
            candidate_scores: Proposal-side candidate scores.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple ``(tokens, scores)``
            for the selected candidate per sample.
        """
        energies = self._score_candidates(noisy_tokens, candidate_tokens)
        neg_energies = -energies
        neg_energies = neg_energies - neg_energies.max(dim=-1, keepdim=True)[0]
        weights = torch.softmax(neg_energies / self.config.energy_temperature, dim=-1)
        selected_idx = torch.multinomial(weights, 1).squeeze(-1)
        batch_idx = torch.arange(noisy_tokens.size(0), device=noisy_tokens.device)
        return (
            candidate_tokens[batch_idx, selected_idx],
            candidate_scores[batch_idx, selected_idx],
        )

    def _resample(self, tokens: torch.Tensor, scores: torch.Tensor) -> None:
        """Mitigates repetition collapse by masked resampling in place.

        Args:
            tokens: Candidate token tensor updated in place.
            scores: Candidate score tensor updated in place.
        """
        to_be_resampled = []
        resample_input = []
        resample_masks = []
        resample_scores = []

        for batch_index, sequence in enumerate(tokens):
            token_positions = {}
            max_frequency = -1
            for position, token in enumerate(sequence):
                token = int(token)
                token_positions.setdefault(token, []).append(position)
                max_frequency = max(max_frequency, len(token_positions[token]))

            if max_frequency <= len(sequence) * self.config.resample_ratio:
                continue

            mask = torch.zeros_like(sequence).bool()
            for token, positions in token_positions.items():
                if len(positions) > len(sequence) * self.config.resample_ratio:
                    mask |= sequence.eq(token)

            to_be_resampled.append(batch_index)
            resample_scores.append(scores[batch_index])
            resample_masks.append(mask)
            resample_input.append(sequence.masked_fill(mask, self.mask_id))

        if not to_be_resampled:
            return

        resample_input = torch.stack(resample_input, dim=0).type_as(tokens)
        resample_scores = torch.stack(resample_scores, dim=0).type_as(scores)
        resample_masks = torch.stack(resample_masks, dim=0).bool()
        logits = self._mask_logits(self.forward(resample_input))
        logits = top_k_top_p_filtering(logits, top_p=self.config.resample_top_p)
        new_tokens, new_scores = stochastic_sample_from_categorical(
            logits, temperature=0.0
        )
        resample_input.masked_scatter_(resample_masks, new_tokens[resample_masks])
        resample_scores.masked_scatter_(resample_masks, new_scores[resample_masks])
        tokens[to_be_resampled] = resample_input
        scores[to_be_resampled] = resample_scores

    def _decode_step(
        self, state: dict[str, Any], partial_masks: torch.Tensor | None = None
    ) -> dict[str, Any]:
        """Runs proposal, reranking, and optional resampling for one step.

        Args:
            state: Current decode state.
            partial_masks: Optional boolean mask of fixed positions.

        Returns:
            dict[str, Any]: Intermediate decode outputs before skeptical remasking.
        """
        output_tokens = state["output_tokens"].clone()
        output_scores = state["output_scores"].clone()
        output_masks = self.get_non_special_symbol_mask(
            output_tokens, partial_masks=partial_masks
        )

        logits = self._mask_logits(self.forward(output_tokens))
        if logits.dtype != output_scores.dtype:
            logits = logits.type_as(output_scores)

        candidate_tokens, candidate_scores = self._sample_candidates(
            logits, self.config.num_candidates
        )
        selected_tokens, selected_scores = self._select_candidates(
            output_tokens, candidate_tokens, candidate_scores
        )

        if not self.config.disable_resample:
            self._resample(selected_tokens, selected_scores)

        output_tokens.masked_scatter_(output_masks, selected_tokens[output_masks])
        output_scores.masked_scatter_(output_masks, selected_scores[output_masks])

        history = list(state["history"])
        history.append(output_tokens.clone())
        return {
            "output_tokens": output_tokens,
            "output_scores": output_scores,
            "history": history,
        }

    def _reparam_decoding(
        self,
        output_tokens: torch.Tensor,
        output_scores: torch.Tensor,
        step_tokens: torch.Tensor,
        step_scores: torch.Tensor,
        decoding_strategy: str,
        still_noisy_mask: torch.Tensor,
        editable_token_mask: torch.Tensor,
        t: int,
        max_step: int,
        noise: int | float | torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Applies skeptical remasking to produce the next decode state.

        Args:
            output_tokens: Previous-step output tokens.
            output_scores: Previous-step token scores.
            step_tokens: Current-step candidate tokens.
            step_scores: Current-step candidate scores.
            decoding_strategy: Skeptical-remasking strategy string.
            still_noisy_mask: Boolean mask tracking which positions remain noisy.
            editable_token_mask: Editable non-special-token mask.
            t: Current decode step index, starting from ``1``.
            max_step: Total decode step count.
            noise: Mask token id or per-position noise tensor.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple
            ``(new_mask, new_tokens, new_scores)`` describing the next decode
            state.
        """
        _, condition, topk_mode, schedule = decoding_strategy.split("-")

        if schedule == "linear":
            rate = 1 - t / max_step
        elif schedule == "cosine":
            rate = np.cos(t / max_step * np.pi * 0.5)
        else:
            raise NotImplementedError(f"Unknown schedule: {schedule}")

        cutoff_len = (
            editable_token_mask.sum(1, keepdim=True).type_as(output_scores) * rate
        ).long()
        scores_for_topk = step_scores.masked_fill(~editable_token_mask, 1000.0)

        if topk_mode.startswith("stochastic"):
            noise_scale = float(topk_mode.replace("stochastic", ""))
            lowest_k_mask = topk_masking(
                scores_for_topk,
                cutoff_len,
                stochastic=True,
                temp=noise_scale * rate,
            )
        elif topk_mode == "deterministic":
            lowest_k_mask = topk_masking(scores_for_topk, cutoff_len, stochastic=False)
        else:
            raise NotImplementedError(f"Unknown topk mode: {topk_mode}")

        if condition == "cond":
            keep_masked_from_previous = (
                (step_tokens == output_tokens)
                & (step_scores < output_scores)
                & lowest_k_mask
            )
        elif condition == "uncond":
            keep_masked_from_previous = lowest_k_mask
        else:
            raise NotImplementedError(f"Unknown condition mode: {condition}")

        keep_masked_this_step = lowest_k_mask
        masked_to_noise = (~still_noisy_mask & keep_masked_from_previous) | (
            still_noisy_mask & keep_masked_this_step
        )
        if isinstance(noise, torch.Tensor):
            output_tokens.masked_scatter_(masked_to_noise, noise[masked_to_noise])
        else:
            output_tokens.masked_fill_(masked_to_noise, noise)
        output_scores.masked_fill_(masked_to_noise, -math.inf)

        masked_to_candidate_tokens = still_noisy_mask & ~keep_masked_this_step
        output_tokens.masked_scatter_(
            masked_to_candidate_tokens, step_tokens[masked_to_candidate_tokens]
        )
        output_scores.masked_scatter_(
            masked_to_candidate_tokens, step_scores[masked_to_candidate_tokens]
        )

        new_still_noisy_mask = (
            still_noisy_mask | keep_masked_from_previous
        ) & keep_masked_this_step
        return new_still_noisy_mask, output_tokens, output_scores
