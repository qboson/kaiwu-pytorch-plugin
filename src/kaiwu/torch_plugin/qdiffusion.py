# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Public single-file QDiffusion module.

This module packages the public energy-guided discrete diffusion API together
with the private DPLM/ESM helpers it needs at runtime. End users are expected
to interact with :class:`QDiffusion` and :class:`QDiffusionConfig`.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, field
import math
import os
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import numpy as np
from omegaconf import OmegaConf
import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoConfig, AutoModelForMaskedLM, AutoTokenizer
from transformers.modeling_outputs import (
    BaseModelOutputWithPoolingAndCrossAttentions,
)
from transformers.models.esm.modeling_esm import (
    EsmAttention,
    EsmContactPredictionHead,
    EsmEmbeddings,
    EsmEncoder,
    EsmForMaskedLM,
    EsmIntermediate,
    EsmLayer,
    EsmLMHead,
    EsmModel,
    EsmOutput,
    EsmPooler,
    EsmPreTrainedModel,
    EsmSelfAttention,
    EsmSelfOutput,
)

SOFTPLUS = nn.Softplus()

__all__ = ["QDiffusion", "QDiffusionConfig"]


def topk_masking(
    scores: torch.Tensor,
    cutoff_len: torch.Tensor,
    stochastic: bool = False,
    temp: float = 1.0,
) -> torch.Tensor:
    """Selects the lowest-score positions used by skeptical remasking.

    Args:
        scores: Ranking scores for each editable token position.
        cutoff_len: Per-sample count of positions to remask.
        stochastic: Whether to perturb scores with Gumbel noise before ranking.
        temp: Temperature applied to the stochastic ranking noise.

    Returns:
        A boolean mask where ``True`` marks positions selected for remasking.
    """
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
    """Samples tokens from categorical logits.

    Args:
        logits: Token logits with vocabulary dimension in the last axis.
        temperature: Sampling temperature. When set to ``0``, argmax decoding is
            used instead of stochastic sampling.

    Returns:
        A tuple ``(tokens, scores)`` containing sampled token ids and their
        corresponding log-probability style scores.
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
        logits: Token logits with vocabulary dimension in the last axis.
        temperature: Sampling temperature passed to the categorical sampler.
        noise_scale: Multiplier for the sampled Gumbel noise.

    Returns:
        A tuple ``(tokens, scores)`` after noisy categorical sampling.
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
        logits: Token logits with vocabulary dimension in the last axis.
        temperature: Sampling temperature passed to the categorical sampler.
        noise_scale: Multiplier for the sampled Gumbel noise.
        n: Number of independent candidate samples to draw.

    Returns:
        A tuple ``(tokens, scores)`` with an additional leading candidate axis.
    """
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
    """Applies top-k and/or nucleus filtering to logits.

    Args:
        logits: Token logits with vocabulary dimension in the last axis.
        top_k: Maximum number of highest-logit tokens to keep.
        top_p: Cumulative probability threshold for nucleus filtering.
        filter_value: Value written into filtered logits.

    Returns:
        Filtered logits with the same shape as ``logits``.
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


@dataclass
class QDiffusionConfig:
    """Configuration for energy-guided discrete generation.

    Attributes:
        num_diffusion_timesteps: Number of discrete noising steps used by the
            training objective.
        rdm_couple: Whether to use the coupled corruption variant.
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
    rdm_couple: bool = False
    num_candidates: int = 1
    proposal_temperature: float = 0.0
    proposal_noise_scale: float = 1.0
    energy_temperature: float = 1.0
    disable_resample: bool = False
    resample_ratio: float = 0.25
    resample_top_p: float = 0.95
    decoding_strategy: str = "reparam-uncond-deterministic-linear"


class QDiffusion(nn.Module):
    """Energy-guided discrete diffusion wrapper with built-in DPLM loaders.

    The class combines two backbone roles:

    - a proposal model that predicts token logits for the current noisy state
    - an energy model that reranks candidate reconstructions

    It exposes both training-oriented APIs such as :meth:`objective` and
    decoding-oriented APIs such as :meth:`initialize_state`, :meth:`step`, and
    :meth:`generate`.
    """

    def __init__(
        self,
        proposal_model: nn.Module,
        energy_model: nn.Module,
        config: QDiffusionConfig | None = None,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str | None = None,
        freeze_proposal: bool = True,
        energy_head: nn.Module | None = None,
    ) -> None:
        """Initializes a QDiffusion model.

        Args:
            proposal_model: Backbone used to predict proposal logits.
            energy_model: Backbone used to score candidate reconstructions.
            config: Optional generation/training configuration.
            dtype: Floating point dtype tracked by the wrapper.
            device: Optional target device. When omitted, infer from parameters.
            freeze_proposal: Whether to freeze proposal model parameters.
            energy_head: Optional custom sequence-level energy head.
        """
        super().__init__()
        self.proposal_model = proposal_model
        self.energy_model = energy_model
        self.config = config or QDiffusionConfig()
        self.dtype = dtype

        if freeze_proposal:
            self.proposal_model.eval()
            for parameter in self.proposal_model.parameters():
                parameter.requires_grad = False

        hidden_size = self.energy_model.net.config.hidden_size
        self.energy_head = energy_head or nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )
        self.embed = self.energy_model.net.get_input_embeddings()
        self.vocab_proj = nn.Linear(2 * hidden_size, hidden_size, bias=True)

        self.tokenizer = self.proposal_model.tokenizer
        self.mask_id = self.proposal_model.mask_id
        self.pad_id = self.proposal_model.pad_id
        self.bos_id = self.proposal_model.bos_id
        self.eos_id = self.proposal_model.eos_id
        self.x_id = self.proposal_model.x_id

        if device is None:
            try:
                self.device = next(self.parameters()).device
            except StopIteration:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)
            self.to(device=self.device, dtype=self.dtype)

    @classmethod
    def load_backbone(
        cls,
        model_name_or_path: str,
        *,
        cfg_override: dict[str, Any] | None = None,
        net_override: dict[str, Any] | None = None,
        from_huggingface: bool = True,
    ) -> _DiffusionProteinLanguageModel:
        """Loads one DPLM backbone compatible with QDiffusion.

        Args:
            model_name_or_path: Hugging Face model id or local checkpoint path.
            cfg_override: Optional config overrides for the wrapper.
            net_override: Optional keyword overrides forwarded to the network
                loader.
            from_huggingface: Whether to treat ``model_name_or_path`` as a
                Hugging Face checkpoint instead of a local training artifact.

        Returns:
            A private backbone wrapper compatible with ``QDiffusion``.
        """
        return _DiffusionProteinLanguageModel.from_pretrained(
            model_name_or_path,
            cfg_override=cfg_override,
            net_override=net_override,
            from_huggingface=from_huggingface,
        )

    @classmethod
    def build(
        cls,
        proposal_ckpt: str,
        energy_ckpt: str,
        *,
        num_candidates: int = 1,
        proposal_temperature: float = 0.0,
        proposal_noise_scale: float = 1.0,
        energy_temperature: float = 1.0,
        disable_resample: bool = False,
        resample_ratio: float = 0.25,
        resample_top_p: float = 0.95,
        freeze_proposal: bool = True,
        proposal_cfg_override: dict[str, Any] | None = None,
        energy_cfg_override: dict[str, Any] | None = None,
    ) -> QDiffusion:
        """Builds a QDiffusion instance from proposal and energy checkpoints.

        Args:
            proposal_ckpt: Proposal backbone checkpoint or model id.
            energy_ckpt: Energy backbone checkpoint or model id.
            num_candidates: Number of candidates sampled per decode step.
            proposal_temperature: Temperature used during proposal sampling.
            proposal_noise_scale: Gumbel noise scale for proposal sampling.
            energy_temperature: Temperature used for energy-based reranking.
            disable_resample: Whether to disable repetition-collapse resampling.
            resample_ratio: Frequency threshold that triggers resampling.
            resample_top_p: Top-p cutoff used during resampling.
            freeze_proposal: Whether to freeze proposal model parameters.
            proposal_cfg_override: Optional proposal wrapper config overrides.
            energy_cfg_override: Optional energy wrapper config overrides.

        Returns:
            A configured ``QDiffusion`` instance.
        """
        proposal_model = cls.load_backbone(
            proposal_ckpt,
            cfg_override=proposal_cfg_override,
        )
        energy_model = cls.load_backbone(
            energy_ckpt,
            cfg_override=energy_cfg_override,
        )
        config = QDiffusionConfig(
            num_candidates=num_candidates,
            proposal_temperature=proposal_temperature,
            proposal_noise_scale=proposal_noise_scale,
            energy_temperature=energy_temperature,
            disable_resample=disable_resample,
            resample_ratio=resample_ratio,
            resample_top_p=resample_top_p,
        )
        return cls(
            proposal_model=proposal_model,
            energy_model=energy_model,
            config=config,
            freeze_proposal=freeze_proposal,
        )

    @classmethod
    def from_pretrained(
        cls,
        proposal_ckpt: str,
        energy_ckpt: str | None = None,
        **kwargs: Any,
    ) -> QDiffusion:
        """Creates a QDiffusion instance from pretrained checkpoints.

        Args:
            proposal_ckpt: Proposal backbone checkpoint or model id.
            energy_ckpt: Optional energy backbone checkpoint or model id. When
                omitted, reuse ``proposal_ckpt``.
            **kwargs: Additional keyword arguments forwarded to :meth:`build`.

        Returns:
            A configured ``QDiffusion`` instance.
        """
        return cls.build(
            proposal_ckpt=proposal_ckpt,
            energy_ckpt=energy_ckpt or proposal_ckpt,
            **kwargs,
        )

    def to(self, *args: Any, **kwargs: Any) -> QDiffusion:
        """Moves the module and refreshes cached device/dtype metadata.

        Args:
            *args: Positional arguments forwarded to ``nn.Module.to``.
            **kwargs: Keyword arguments forwarded to ``nn.Module.to``.

        Returns:
            The moved module instance.
        """
        module = super().to(*args, **kwargs)
        try:
            self.device = next(self.parameters()).device
            self.dtype = next(self.parameters()).dtype
        except StopIteration:
            self.device = torch.device("cpu")
        return module

    def forward(self, xt: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Runs the proposal model on the current noisy state.

        Args:
            xt: Current noisy token tensor.
            **kwargs: Additional keyword arguments forwarded to the proposal
                model.

        Returns:
            Proposal logits over the token vocabulary.

        Raises:
            TypeError: If the proposal model does not implement ``forward``.
        """
        if hasattr(self.proposal_model, "forward"):
            return self.proposal_model.forward(xt, **kwargs)
        raise TypeError("proposal_model must implement forward().")

    def proposal(self, xt: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Semantic alias around :meth:`forward` for proposal-side calls.

        Args:
            xt: Current noisy token tensor.
            **kwargs: Additional keyword arguments forwarded to the proposal
                model.

        Returns:
            Proposal logits over the token vocabulary.
        """
        return self.forward(xt, **kwargs)

    def energy(
        self,
        xt: torch.Tensor,
        x0: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Scores candidate reconstructions conditioned on the noisy state.

        Args:
            xt: Noisy token tensor used as conditioning input.
            x0: Candidate clean token tensor to score.
            attention_mask: Optional attention mask for the energy model.
            **kwargs: Reserved for future pair-scoring extensions.

        Returns:
            A tensor of scalar energies with shape ``[batch, 1]``.
        """
        del kwargs
        if attention_mask is None:
            attention_mask = x0.ne(self.pad_id)

        if xt.device.type == "cuda":
            outer_context = torch.amp.autocast("cuda", dtype=torch.float32)
            inner_context = torch.amp.autocast("cuda", dtype=torch.bfloat16)
        else:
            outer_context = nullcontext()
            inner_context = nullcontext()

        with outer_context:
            xt_embed = self.embed(xt)
            x0_embed = self.embed(x0)
            fused_embed = self.vocab_proj(torch.cat([xt_embed, x0_embed], dim=-1))

            with inner_context:
                hidden = self.energy_model.net(
                    input_ids=xt,
                    inputs_embeds=fused_embed,
                    attention_mask=attention_mask,
                )["last_hidden_state"]
                pooled = hidden.mean(dim=1)
                return self.energy_head(pooled)

    def objective(
        self, batch: dict[str, torch.Tensor], weighting: str = "constant"
    ) -> dict[str, torch.Tensor]:
        """Builds the one-step training objective used by an external loop.

        Args:
            batch: Batch dictionary containing at least ``batch["targets"]``.
            weighting: Per-sample timestep weighting mode.

        Returns:
            A dictionary containing proposal logits, supervision masks, loss
            weights, and the EBM objective term.
        """
        target = batch["targets"]
        t1, t2 = torch.randint(
            1,
            self.config.num_diffusion_timesteps + 1,
            (2 * target.size(0),),
            device=target.device,
        ).chunk(2)

        if self.config.rdm_couple:
            q_outputs = self._q_sample_coupled(
                target,
                t1,
                t2,
                self.get_non_special_symbol_mask(target),
            )
            target = target.repeat(2, 1)
        else:
            q_outputs = self._q_sample(
                target,
                t1,
                self.get_non_special_symbol_mask(target),
            )

        x_t = q_outputs["x_t"]
        timesteps = q_outputs["t"]
        loss_mask = q_outputs["loss_mask"]

        with torch.no_grad():
            logits = self.forward(x_t).detach()

        negative_tokens, _ = self._sample_candidates(logits, self.config.num_candidates)
        positive_energy = self.energy(x_t, target, target.ne(self.pad_id))
        negative_energy = self._score_candidates(x_t, negative_tokens).mean(
            dim=1, keepdim=True
        )
        objective_ebm = SOFTPLUS(positive_energy) + SOFTPLUS(-negative_energy)
        weight = self._compute_loss_weight(timesteps, weighting)

        return {
            "logits": logits,
            "targets": target,
            "loss_mask": loss_mask,
            "weight": weight,
            "objective_ebm": objective_ebm,
        }

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
            A mutable state dictionary suitable for repeated :meth:`step` calls.
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

    def step(self, state: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        """Runs one denoising/reranking step and returns updated state.

        Args:
            state: Current decode state created by :meth:`initialize_state`.
            **kwargs: Optional overrides such as ``partial_masks``.

        Returns:
            The updated decode state after one iteration.
        """
        partial_masks = kwargs.get("partial_masks", state.get("partial_masks"))
        decoder_out = self._decode_step(state, partial_masks=partial_masks)

        non_special_sym_mask = self.get_non_special_symbol_mask(
            state["output_tokens"], partial_masks=partial_masks
        )
        output_masks, result_tokens, result_scores = self._reparam_decoding(
            output_tokens=state["output_tokens"].clone(),
            output_scores=state["output_scores"].clone(),
            cur_tokens=decoder_out["output_tokens"].clone(),
            cur_scores=decoder_out["output_scores"].clone(),
            decoding_strategy=self.config.decoding_strategy,
            xt_neq_x0=state["output_masks"],
            non_special_sym_mask=non_special_sym_mask,
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
            history=decoder_out["history"],
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
        **kwargs: Any,
    ) -> torch.Tensor | dict[str, Any]:
        """Runs a complete iterative decoding loop inside the core class.

        Args:
            input_tokens: Initial token tensor.
            max_steps: Number of decode iterations to run.
            partial_masks: Optional boolean mask of fixed positions.
            temperature: Sampling temperature stored in the decode state.
            return_state: Whether to return the full final state dictionary.
            **kwargs: Additional keyword arguments forwarded to :meth:`step`.

        Returns:
            Either the final token tensor or the full decode state.
        """
        state = self.initialize_state(
            input_tokens=input_tokens,
            partial_masks=partial_masks,
            max_steps=max_steps,
            temperature=temperature,
        )
        for _ in range(max_steps):
            state = self.step(state, partial_masks=partial_masks, **kwargs)

        if return_state:
            return state
        return state["output_tokens"]

    def get_non_special_symbol_mask(
        self, output_tokens: torch.Tensor, partial_masks: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Returns a boolean mask of editable non-special-token positions.

        Args:
            output_tokens: Token tensor to inspect.
            partial_masks: Optional boolean mask of positions that should remain
                fixed.

        Returns:
            A boolean mask where ``True`` marks editable non-special positions.
        """
        non_special_sym_mask = (
            output_tokens.ne(self.pad_id)
            & output_tokens.ne(self.bos_id)
            & output_tokens.ne(self.eos_id)
        )
        if partial_masks is not None:
            non_special_sym_mask &= ~partial_masks
        return non_special_sym_mask

    def _initialize_output_tokens(
        self, input_tokens: torch.Tensor, partial_masks: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Builds the initial fully masked token state for decoding.

        Args:
            input_tokens: Initial token tensor.
            partial_masks: Optional boolean mask of fixed positions.

        Returns:
            A tuple ``(output_tokens, output_scores)`` for the initial decode
            state.
        """
        output_mask = self.get_non_special_symbol_mask(
            input_tokens, partial_masks=partial_masks
        )
        output_tokens = input_tokens.masked_fill(output_mask, self.mask_id)
        output_scores = torch.zeros_like(output_tokens, dtype=torch.float)
        return output_tokens, output_scores

    def _q_sample(
        self, x_0: torch.Tensor, t1: torch.Tensor, maskable_mask: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Applies one-step discrete corruption for training.

        Args:
            x_0: Clean target tokens.
            t1: Sampled diffusion timesteps.
            maskable_mask: Boolean mask of positions eligible for masking.

        Returns:
            A dictionary containing corrupted tokens, timesteps, and loss mask.
        """
        noise = torch.rand_like(x_0, dtype=torch.float)
        t1_mask = (
            noise < (t1 / self.config.num_diffusion_timesteps)[:, None]
        ) & maskable_mask
        x_t = x_0.masked_fill(t1_mask, self.mask_id)
        return {"x_t": x_t, "t": t1, "loss_mask": t1_mask}

    def _q_sample_coupled(
        self,
        x_0: torch.Tensor,
        t1: torch.Tensor,
        t2: torch.Tensor,
        maskable_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Applies the coupled corruption variant used by RDM-style training.

        Args:
            x_0: Clean target tokens.
            t1: First sampled timestep tensor.
            t2: Second sampled timestep tensor.
            maskable_mask: Boolean mask of positions eligible for masking.

        Returns:
            A dictionary containing paired corruptions, timesteps, and loss mask.
        """
        same_t_mask = t1 == t2
        t1, t2 = torch.maximum(t1, t2).float(), torch.minimum(t1, t2).float()

        noise = torch.rand_like(x_0, dtype=torch.float)
        t1_mask = (
            noise < (t1 / self.config.num_diffusion_timesteps)[:, None]
        ) & maskable_mask
        x_t1 = x_0.masked_fill(t1_mask, self.mask_id)

        noise = torch.rand_like(x_0, dtype=torch.float)
        t2_mask = t1_mask & (noise > ((t1 - t2) / t1)[:, None])
        noise = torch.rand_like(x_0[same_t_mask], dtype=torch.float)
        t2_mask[same_t_mask] = (
            noise < (t1[same_t_mask] / self.config.num_diffusion_timesteps)[:, None]
        ) & maskable_mask[same_t_mask]
        x_t2 = x_0.masked_fill(t2_mask, self.mask_id)

        return {
            "x_t": torch.cat([x_t1, x_t2], dim=0),
            "t": torch.cat([t1, t2]),
            "loss_mask": torch.cat([t1_mask, t2_mask], dim=0),
        }

    def _compute_loss_weight(
        self, timesteps: torch.Tensor, weighting: str
    ) -> torch.Tensor:
        """Converts sampled timesteps into per-sample loss weights.

        Args:
            timesteps: Sampled diffusion timesteps.
            weighting: Weighting strategy name.

        Returns:
            A column vector of normalized per-sample weights.
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
            A cloned logits tensor with special-token entries masked out.
        """
        logits = logits.clone()
        logits[..., self.mask_id] = -math.inf
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
            A candidate tensor shaped as ``[batch, num_candidates, seq_len]``.

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
            A tuple ``(tokens, scores)`` shaped as
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
        self, xt: torch.Tensor, candidate_tokens: torch.Tensor
    ) -> torch.Tensor:
        """Scores a batch of candidate reconstructions with the energy model.

        Args:
            xt: Current noisy token tensor.
            candidate_tokens: Candidate reconstructions shaped as
                ``[batch, num_candidates, seq_len]``.

        Returns:
            Candidate energies shaped as ``[batch, num_candidates]``.
        """
        batch_size, num_candidates, seq_len = candidate_tokens.shape
        flat_xt = (
            xt.unsqueeze(1)
            .expand(-1, num_candidates, -1)
            .reshape(batch_size * num_candidates, seq_len)
        )
        flat_x0 = candidate_tokens.reshape(batch_size * num_candidates, seq_len)
        flat_attention_mask = flat_x0.ne(self.pad_id)
        energy = self.energy(flat_xt, flat_x0, flat_attention_mask)
        return energy.view(batch_size, num_candidates)

    def _select_candidates(
        self,
        xt: torch.Tensor,
        candidate_tokens: torch.Tensor,
        candidate_scores: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Selects one candidate per batch element by energy-based reranking.

        Args:
            xt: Current noisy token tensor.
            candidate_tokens: Candidate reconstructions.
            candidate_scores: Proposal-side candidate scores.

        Returns:
            A tuple ``(tokens, scores)`` for the selected candidate per sample.
        """
        energies = self._score_candidates(xt, candidate_tokens)
        neg_energies = -energies
        neg_energies = neg_energies - neg_energies.max(dim=-1, keepdim=True)[0]
        weights = torch.softmax(neg_energies / self.config.energy_temperature, dim=-1)
        selected_idx = torch.multinomial(weights, 1).squeeze(-1)
        batch_idx = torch.arange(xt.size(0), device=xt.device)
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
            Intermediate decode outputs before skeptical remasking.
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
        cur_tokens: torch.Tensor,
        cur_scores: torch.Tensor,
        decoding_strategy: str,
        xt_neq_x0: torch.Tensor,
        non_special_sym_mask: torch.Tensor,
        t: int,
        max_step: int,
        noise: int | float | torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Applies skeptical remasking to produce the next decode state.

        Args:
            output_tokens: Previous-step output tokens.
            output_scores: Previous-step token scores.
            cur_tokens: Current-step candidate tokens.
            cur_scores: Current-step candidate scores.
            decoding_strategy: Skeptical-remasking strategy string.
            xt_neq_x0: Boolean mask tracking which positions remain noisy.
            non_special_sym_mask: Editable non-special-token mask.
            t: Current decode step index, starting from ``1``.
            max_step: Total decode step count.
            noise: Mask token id or per-position noise tensor.

        Returns:
            A tuple ``(new_mask, new_tokens, new_scores)`` describing the next
            decode state.
        """
        _, condition, topk_mode, schedule = decoding_strategy.split("-")

        if schedule == "linear":
            rate = 1 - t / max_step
        elif schedule == "cosine":
            rate = np.cos(t / max_step * np.pi * 0.5)
        else:
            raise NotImplementedError(f"Unknown schedule: {schedule}")

        cutoff_len = (
            non_special_sym_mask.sum(1, keepdim=True).type_as(output_scores) * rate
        ).long()
        scores_for_topk = cur_scores.masked_fill(~non_special_sym_mask, 1000.0)

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
            not_v1_t = (
                (cur_tokens == output_tokens)
                & (cur_scores < output_scores)
                & lowest_k_mask
            )
        elif condition == "uncond":
            not_v1_t = lowest_k_mask
        else:
            raise NotImplementedError(f"Unknown condition mode: {condition}")

        not_v2_t = lowest_k_mask
        masked_to_noise = (~xt_neq_x0 & not_v1_t) | (xt_neq_x0 & not_v2_t)
        if isinstance(noise, torch.Tensor):
            output_tokens.masked_scatter_(masked_to_noise, noise[masked_to_noise])
        else:
            output_tokens.masked_fill_(masked_to_noise, noise)
        output_scores.masked_fill_(masked_to_noise, -math.inf)

        masked_to_x0 = xt_neq_x0 & ~not_v2_t
        output_tokens.masked_scatter_(masked_to_x0, cur_tokens[masked_to_x0])
        output_scores.masked_scatter_(masked_to_x0, cur_scores[masked_to_x0])

        new_xt_neq_x0 = (xt_neq_x0 | not_v1_t) & not_v2_t
        return new_xt_neq_x0, output_tokens, output_scores


@dataclass
class _NetConfig:
    arch_type: str = "esm"
    name: str = "esm2_t33_650M_UR50D"
    dropout: float = 0.1
    pretrain: bool = False
    pretrained_model_name_or_path: str = ""


@dataclass
class _LoRAConfig:
    enable: bool = field(default=False)
    lora_rank: int = field(default=16)
    lora_dropout: float = field(default=0.1)
    lora_target_module: str = field(default="")
    modules_to_save: str = field(default="")


@dataclass
class _DPLMConfig:
    num_diffusion_timesteps: int = field(default=500)
    lora: _LoRAConfig = field(default_factory=_LoRAConfig)
    net: _NetConfig = field(default_factory=_NetConfig)
    gradient_ckpt: bool = field(default=False)
    rdm_couple: bool = field(default=False)


def _load_yaml_config(fpath: str):
    cfg = OmegaConf.load(fpath)
    OmegaConf.resolve(cfg)
    return cfg


def _get_net_class(dplm_type: str):
    if dplm_type != "dplm_esm":
        raise ValueError(f"Unsupported dplm_type for QDiffusion: {dplm_type}")
    return _EsmForDPLM


def _get_net(cfg):
    if cfg.net.arch_type != "esm":
        raise NotImplementedError(
            f"Unsupported arch_type for QDiffusion: {cfg.net.arch_type}"
        )

    config = AutoConfig.from_pretrained(cfg.net.name)
    net = _EsmForDPLM(config, dropout=cfg.net.dropout)

    if cfg.net.pretrain:
        pretrained_model_name_or_path = cfg.net.pretrained_model_name_or_path
        is_local = os.path.exists(pretrained_model_name_or_path)
        if is_local:
            pretrained_state_dict = torch.load(
                pretrained_model_name_or_path, map_location="cpu"
            )["state_dict"]
            from collections import OrderedDict

            new_pretrained_state_dict = OrderedDict()
            for key, value in pretrained_state_dict.items():
                new_pretrained_state_dict[key[10:]] = value
            net.load_state_dict(new_pretrained_state_dict, strict=True)
        else:
            pretrained_net = AutoModelForMaskedLM.from_pretrained(
                pretrained_model_name_or_path
            )
            net.load_state_dict(pretrained_net.state_dict(), strict=True)
            del pretrained_net

    return net


class _DiffusionProteinLanguageModel(nn.Module):
    """Minimal private DPLM backbone wrapper used by QDiffusion."""

    _default_cfg = _DPLMConfig()

    def __init__(self, cfg=None, net=None):
        super().__init__()
        self._update_cfg(cfg or {})

        self.net = _get_net(self.cfg) if net is None else net
        self.tokenizer = self.net.tokenizer

        self.mask_id = self.net.mask_id
        self.pad_id = self.net.pad_id
        self.bos_id = self.net.bos_id
        self.eos_id = self.net.eos_id
        self.x_id = self.net.x_id

        if self.cfg.gradient_ckpt:
            self.net.supports_gradient_checkpointing = True
            self.net.gradient_checkpointing_enable()

    @classmethod
    def from_pretrained(
        cls,
        net_name,
        cfg_override=None,
        net_override=None,
        from_huggingface=True,
    ):
        """Loads a private DPLM backbone wrapper from a checkpoint.

        Args:
            net_name: Hugging Face model id or local checkpoint path.
            cfg_override: Optional wrapper config overrides.
            net_override: Optional keyword overrides forwarded to the network
                loader.
            from_huggingface: Whether ``net_name`` should be loaded through the
                Hugging Face API.

        Returns:
            A configured private backbone wrapper.
        """
        cfg_override = cfg_override or {}
        net_override = net_override or {}

        if not from_huggingface:
            from collections import OrderedDict

            cfg_path = Path(net_name).parents[1]
            cfg_path = Path(cfg_path, ".hydra", "config.yaml")
            cfg = _load_yaml_config(str(cfg_path)).model
            cfg.net.pretrain = False
            cfg.pop("_target_")
            model = cls(cfg)

            pretrained_state_dict = torch.load(
                net_name, map_location=torch.device("cpu")
            )["state_dict"]
            new_pretrained_state_dict = OrderedDict()
            for key, value in pretrained_state_dict.items():
                new_pretrained_state_dict[key[6:]] = value

            model.load_state_dict(new_pretrained_state_dict, strict=False)
            return model

        dplm_type = AutoConfig.from_pretrained(net_name).dplm_type
        net_class = _get_net_class(dplm_type)
        net = net_class.from_pretrained(net_name, **net_override)
        return cls(cfg=cfg_override, net=net)

    def _update_cfg(self, cfg):
        """Merges runtime config overrides onto the default backbone config."""
        self.cfg = OmegaConf.merge(self._default_cfg, cfg)

    def forward(self, input_ids, return_last_hidden_state=False, **kwargs):
        """Runs the wrapped DPLM backbone.

        Args:
            input_ids: Input token ids.
            return_last_hidden_state: Whether to return hidden states together
                with logits.
            **kwargs: Unused compatibility keyword arguments.

        Returns:
            Either logits alone or a tuple ``(logits, last_hidden_state)``.
        """
        del kwargs
        outputs = self.net(input_ids=input_ids)
        logits = outputs["logits"]
        if return_last_hidden_state:
            return logits, outputs["last_hidden_state"]
        return logits


class _ModifiedEsmSelfAttention(EsmSelfAttention):
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor, ...], ...]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, ...]:
        del output_attentions
        mixed_query_layer = self.query(hidden_states)
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention and past_key_value is not None:
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

        query_layer = self.transpose_for_scores(mixed_query_layer)
        query_layer = query_layer * self.attention_head_size**-0.5

        if self.is_decoder:
            past_key_value = (key_layer, value_layer)

        if self.position_embedding_type == "rotary":
            query_layer, key_layer = self.rotary_embeddings(query_layer, key_layer)

        if self.position_embedding_type in {"relative_key", "relative_key_query"}:
            raise NotImplementedError
        if head_mask is not None:
            raise NotImplementedError

        query_layer = query_layer.contiguous()
        key_layer = key_layer.contiguous()
        value_layer = value_layer.contiguous()
        context_layer = F.scaled_dot_product_attention(
            query_layer,
            key_layer,
            value_layer,
            attn_mask=attention_mask,
            scale=1.0,
        )

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)
        outputs = (context_layer,)

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs


class _ModifiedEsmAttention(EsmAttention):
    def __init__(self, config):
        nn.Module.__init__(self)
        self.self = _ModifiedEsmSelfAttention(config)
        self.output = EsmSelfOutput(config)
        self.pruned_heads = set()
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)


class _ModifiedEsmLayer(EsmLayer):
    def __init__(self, config):
        nn.Module.__init__(self)
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = _ModifiedEsmAttention(config)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            if not self.is_decoder:
                raise RuntimeError(
                    f"{self} should be used as a decoder model if cross attention is added"
                )
            self.crossattention = _ModifiedEsmAttention(config)
        self.intermediate = EsmIntermediate(config)
        self.output = EsmOutput(config)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)


class _ModifiedEsmEncoder(EsmEncoder):
    def __init__(self, config):
        nn.Module.__init__(self)
        self.config = config
        self.layer = nn.ModuleList(
            [_ModifiedEsmLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.emb_layer_norm_after = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )
        self.gradient_checkpointing = False


class _ModifiedEsmModel(EsmModel):
    def __init__(self, config, add_pooling_layer=True):
        EsmPreTrainedModel.__init__(self, config)
        self.config = config
        self.embeddings = EsmEmbeddings(config)
        self.encoder = _ModifiedEsmEncoder(config)
        self.pooler = EsmPooler(config) if add_pooling_layer else None
        self.contact_head = EsmContactPredictionHead(
            in_features=config.num_hidden_layers * config.num_attention_heads,
            bias=True,
        )
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[
        Tuple[torch.Tensor, ...], BaseModelOutputWithPoolingAndCrossAttentions
    ]:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        elif input_ids is not None:
            input_shape = input_ids.size()
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        past_key_values_length = (
            past_key_values[0][0].shape[2] if past_key_values is not None else 0
        )

        if attention_mask is None:
            attention_mask = torch.ones(
                ((batch_size, seq_length + past_key_values_length)),
                device=device,
            )

        extended_attention_mask = self.get_extended_attention_mask(
            attention_mask, input_shape
        )

        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(
                encoder_attention_mask
            )
        else:
            encoder_extended_attention_mask = encoder_attention_mask

        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )


class _EsmForDPLM(EsmForMaskedLM):
    def __init__(self, config, dropout=0.1):
        tokenizer = AutoTokenizer.from_pretrained(config._name_or_path)
        config.hidden_dropout_prob = dropout

        EsmPreTrainedModel.__init__(self, config)
        self.esm = _ModifiedEsmModel(config, add_pooling_layer=False)
        self.lm_head = EsmLMHead(config)
        self.init_weights()

        self.mask_id = tokenizer.mask_token_id
        self.pad_id = tokenizer.pad_token_id
        self.bos_id = tokenizer.cls_token_id
        self.eos_id = tokenizer.eos_token_id
        self.x_id = tokenizer._token_to_id["X"]

        self.contact_head = None
        self.tokenizer = tokenizer

    def forward(
        self,
        input_ids,
        attention_mask=None,
        inputs_embeds=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        decoder_inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
    ):
        del (
            attention_mask,
            decoder_input_ids,
            decoder_attention_mask,
            decoder_inputs_embeds,
            labels,
            output_attentions,
            output_hidden_states,
            return_dict,
        )

        attention_mask = input_ids.ne(self.pad_id)
        outputs = self.esm(
            input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
        )

        sequence_output = outputs[0]
        logits = self.lm_head(sequence_output)
        return {
            "logits": logits,
            "last_hidden_state": sequence_output,
        }
