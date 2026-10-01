"""Candidate-selection hook for Nemotron's native diffusion generator."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Callable

import torch


@dataclass(frozen=True)
class ProposalDecision:
    """Tokens and positions to commit during one native denoising step.

    Attributes:
        tokens: Full token block proposed for the current denoising step.
        transfer_index: Boolean positions from ``tokens`` to commit.
    """

    tokens: torch.Tensor
    transfer_index: torch.Tensor


@dataclass(frozen=True)
class ProposalStep:
    """Read-only state made available at Nemotron's selection point.

    Attributes:
        block_index: Zero-based index of the current generated block.
        step_index: Zero-based denoising step within the current block.
        nfe: Native model evaluation count at this selection point.
        prompt_length: Number of tokens in the original prompt.
        block_start: Position of the current block in ``sequence_tokens``.
        sequence_tokens: Prompt, completed blocks, and current noisy block.
        block_tokens: Current block before committing this decision.
        mask_index: Boolean mask of positions that remain noisy.
        mask_token_id: Nemotron mask-token id.
        logits: Native model logits for the current block.
        num_transfer_tokens: Native number of tokens to commit per batch item.
        temperature: Native proposal temperature.
        threshold: Native confidence threshold, if configured.
        native_decision: Decision produced by the native transfer helper.
        hidden_states: Final-layer states for the current block, when captured.
    """

    block_index: int
    step_index: int
    nfe: int
    prompt_length: int
    block_start: int
    sequence_tokens: torch.Tensor
    block_tokens: torch.Tensor
    mask_index: torch.Tensor
    mask_token_id: int
    logits: torch.Tensor
    num_transfer_tokens: torch.Tensor
    temperature: float
    threshold: float | None
    native_decision: ProposalDecision
    hidden_states: torch.Tensor | None = None


ProposalHook = Callable[[ProposalStep], ProposalDecision | None]


class NativeGenerationSession:
    """Run the loaded model's ``generate`` with an optional selection hook.

    Nemotron's remote-code ``generate`` owns the cache, denoising schedule and
    stopping rules. Its module-level transfer helper is the one point where a
    candidate decision exists before it is committed, so this session wraps
    that helper only for the duration of one serial generation call.

    Args:
        model: Loaded Nemotron remote-code model.
        proposal_hook: Optional candidate selector. Returning ``None`` keeps
            the native decision unchanged.
        capture_hidden_states: Whether to expose final-layer states to the
            selector.
    """

    def __init__(
        self,
        model: Any,
        proposal_hook: ProposalHook | None = None,
        *,
        capture_hidden_states: bool = False,
    ) -> None:
        self.model = model
        self.proposal_hook = proposal_hook
        self.capture_hidden_states = capture_hidden_states
        self._module = importlib.import_module(type(model).__module__)
        self._original_get_transfer_index = getattr(
            self._module, "_get_transfer_index"
        )
        self._hidden_hook_handle: Any | None = None
        self._last_hidden_states: torch.Tensor | None = None
        self._prompt_tokens: torch.Tensor | None = None
        self._completed_tokens: torch.Tensor | None = None
        self._current_block: torch.Tensor | None = None
        self._current_block_pointer: int | None = None
        self._block_index = -1
        self._step_index = 0
        self._denoising_steps = 0
        self._block_length = 0

    def __enter__(self) -> "NativeGenerationSession":
        setattr(self._module, "_get_transfer_index", self._get_transfer_index)
        if self.capture_hidden_states:
            self._hidden_hook_handle = self.model.encoder.layers[
                -1
            ].register_forward_hook(self._capture_hidden_state)
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        del exc_type, exc_value, traceback
        self.close()

    def close(self) -> None:
        """Restore the native transfer helper and remove the state hook."""

        setattr(
            self._module,
            "_get_transfer_index",
            self._original_get_transfer_index,
        )
        if self._hidden_hook_handle is not None:
            self._hidden_hook_handle.remove()
            self._hidden_hook_handle = None

    def _capture_hidden_state(
        self,
        _module: Any,
        _inputs: Any,
        output: Any,
    ) -> None:
        """Store the final-layer output from the current native model call.

        Args:
            _module: Layer that produced ``output``.
            _inputs: Positional layer inputs.
            output: Layer output tensor or tuple containing that tensor.
        """
        if isinstance(output, tuple):
            output = output[0]
        self._last_hidden_states = output.detach()

    @staticmethod
    def _validate_decision(
        decision: ProposalDecision,
        block_tokens: torch.Tensor,
        mask_index: torch.Tensor,
        native_decision: ProposalDecision,
        mask_token_id: int,
    ) -> None:
        """Check that a selector preserves the native commit contract.

        Args:
            decision: Selector-provided decision to validate.
            block_tokens: Current noisy block.
            mask_index: Positions that may be committed.
            native_decision: Native decision whose transfer count is retained.
            mask_token_id: Token id that must never be committed.

        Raises:
            ValueError: If the decision changes the block shape, commit count,
                selectable positions, or commits a mask token.
        """
        if (
            decision.tokens.shape != block_tokens.shape
            or decision.transfer_index.shape != block_tokens.shape
        ):
            raise ValueError("proposal decision must match the current block shape")
        if decision.transfer_index.dtype != torch.bool:
            raise ValueError("proposal transfer_index must be boolean")
        if (decision.transfer_index & ~mask_index).any():
            raise ValueError("proposal may only select masked positions")
        if not torch.equal(
            decision.transfer_index.sum(dim=1),
            native_decision.transfer_index.sum(dim=1),
        ):
            raise ValueError("proposal must preserve the native transfer count")
        if (decision.tokens[decision.transfer_index] == mask_token_id).any():
            raise ValueError("proposal may not commit the mask token")

    def generate(
        self,
        prompt_ids: torch.Tensor,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, int]:
        """Call the model's untouched native ``generate`` implementation.

        Args:
            prompt_ids: Prompt tokens with shape ``[batch, sequence]``.
            **kwargs: Native ``generate`` arguments. ``block_length`` is
                required so hook metadata can identify block boundaries.

        Returns:
            The native output tokens and native model evaluation count.
        """

        self._prompt_tokens = prompt_ids
        self._completed_tokens = prompt_ids
        self._block_length = int(kwargs["block_length"])
        self._current_block = None
        self._current_block_pointer = None
        self._block_index = -1
        self._step_index = 0
        self._denoising_steps = 0
        return self.model.generate(prompt_ids, **kwargs)

    def _start_block(self, block_tokens: torch.Tensor) -> None:
        """Record a new block and preserve the completed native prefix.

        Args:
            block_tokens: Newly appended noisy block from native generation.
        """
        if self._current_block is not None:
            assert self._completed_tokens is not None
            self._completed_tokens = torch.cat(
                [self._completed_tokens, self._current_block], dim=1
            )
        self._block_index += 1
        self._step_index = 0
        self._current_block_pointer = block_tokens.data_ptr()
        self._current_block = block_tokens.clone()

    def _get_transfer_index(
        self,
        logits: torch.Tensor,
        temperature: float,
        mask_index: torch.Tensor,
        block_tokens: torch.Tensor,
        num_transfer_tokens: torch.Tensor,
        threshold: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Delegate native proposal generation, then optionally select a candidate.

        Args:
            logits: Native logits for the current noisy block.
            temperature: Native proposal temperature.
            mask_index: Positions that remain noisy.
            block_tokens: Current block before tokens are committed.
            num_transfer_tokens: Native per-batch transfer counts.
            threshold: Optional native confidence threshold.

        Returns:
            Tokens and boolean transfer positions for native generation to
            commit.
        """

        native_tokens, native_transfer_index = self._original_get_transfer_index(
            logits,
            temperature,
            mask_index,
            block_tokens,
            num_transfer_tokens,
            threshold=threshold,
        )
        block_pointer = block_tokens.data_ptr()
        if block_pointer != self._current_block_pointer:
            self._start_block(block_tokens)

        native_decision = ProposalDecision(native_tokens, native_transfer_index)
        decision = native_decision
        if self.proposal_hook is not None:
            assert self._completed_tokens is not None
            assert self._prompt_tokens is not None
            step = ProposalStep(
                block_index=self._block_index,
                step_index=self._step_index,
                nfe=self._denoising_steps + self._block_index + 1,
                prompt_length=self._prompt_tokens.size(1),
                block_start=self._prompt_tokens.size(1)
                + self._block_index * self._block_length,
                sequence_tokens=torch.cat(
                    [self._completed_tokens, block_tokens], dim=1
                ),
                block_tokens=block_tokens,
                mask_index=mask_index,
                mask_token_id=int(self.model.mask_token_id),
                logits=logits,
                num_transfer_tokens=num_transfer_tokens,
                temperature=temperature,
                threshold=threshold,
                native_decision=native_decision,
                hidden_states=self._last_hidden_states,
            )
            selected = self.proposal_hook(step)
            if selected is not None:
                self._validate_decision(
                    selected,
                    block_tokens,
                    mask_index,
                    native_decision,
                    int(self.model.mask_token_id),
                )
                decision = selected

        current_block = block_tokens.clone()
        current_block[decision.transfer_index] = decision.tokens[
            decision.transfer_index
        ]
        self._current_block = current_block
        self._step_index += 1
        self._denoising_steps += 1
        return decision.tokens, decision.transfer_index
