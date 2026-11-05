# Copyright (C) 2022-2025 Beijing QBoson Quantum Technology Co., Ltd.
#
# SPDX-License-Identifier: Apache-2.0
"""Private input validation helpers for the public QDiffusion model."""

from __future__ import annotations

import torch


def validate_token_tensor(tensor: torch.Tensor, name: str) -> None:
    """Rejects decode inputs that are not two-dimensional token tensors.

    Args:
        tensor: Token or mask tensor supplied by the caller.
        name: Argument name used in the error message.

    Raises:
        TypeError: If ``tensor`` is not a torch tensor.
        ValueError: If ``tensor`` is not two-dimensional.
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor).__name__}")
    if tensor.ndim != 2:
        raise ValueError(
            f"{name} must have shape (batch, seq_len), got {tuple(tensor.shape)}"
        )


def validate_partial_masks(
    partial_masks: torch.Tensor | None,
    tokens: torch.Tensor,
) -> None:
    """Rejects a partial mask that does not align with the token tensor.

    Args:
        partial_masks: Optional boolean mask of fixed positions.
        tokens: Token tensor the mask must match.

    Raises:
        TypeError: If ``partial_masks`` is not a torch tensor.
        ValueError: If ``partial_masks`` is not two-dimensional or does not have
            the same shape as ``tokens``.
    """
    if partial_masks is None:
        return
    validate_token_tensor(partial_masks, "partial_masks")
    if partial_masks.shape != tokens.shape:
        raise ValueError(
            "partial_masks must have the same shape as input_tokens: "
            f"expected {tuple(tokens.shape)}, got {tuple(partial_masks.shape)}"
        )
