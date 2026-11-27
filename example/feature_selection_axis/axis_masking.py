# Copyright (C) 2022-2025 Beijing QBoson Quantum Technology Co., Ltd.
#
# SPDX-License-Identifier: Apache-2.0
"""Mask features along a non-default axis with ``FeatureSelectionWrapper``.

``FeatureSelectionWrapper`` masks one axis of its input, selected by
``input_feature_axis``. The default ``-1`` matches the usual
``(batch, features)`` layout; feature-first data such as
``(features, batch)`` needs ``input_feature_axis=0``.

This example installs a known mask on both layouts and checks that exactly the
unselected features are zeroed, that ``forward`` passes the masked tensor to
the wrapped model, and that a mismatched feature-axis size is rejected.
"""

from __future__ import annotations

import torch
from torch import nn

from kaiwu.torch_plugin import FeatureSelectionWrapper

FEATURE_DIM = 6
BATCH = 8
KEEP = torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
DROPPED = [1, 3, 5]


class FeatureFirstModel(nn.Module):
    """Toy model for ``(features, batch)`` inputs that sums over features."""

    def __init__(self) -> None:
        super().__init__()
        self.last_input = None

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Records the received tensor and reduces over the feature axis."""
        self.last_input = inputs.detach().clone()
        return inputs.sum(dim=0, keepdim=True)


def build_selector(model: nn.Module | None = None, axis: int = 0) -> FeatureSelectionWrapper:
    """Builds a wrapper whose selectable features live on ``axis``."""
    return FeatureSelectionWrapper(
        model if model is not None else FeatureFirstModel(),
        feature_dim=FEATURE_DIM,
        input_feature_axis=axis,
        min_selected_features=2,
        max_selected_features=4,
    )


def install_mask(selector: FeatureSelectionWrapper) -> None:
    """Installs a fixed mask that keeps features 0, 2 and 4."""
    with torch.no_grad():
        selector.mask.copy_(KEEP)


def main() -> None:
    """Runs the feature-first and batch-first masking checks."""
    inputs = torch.arange(FEATURE_DIM * BATCH, dtype=torch.float32).reshape(
        FEATURE_DIM, BATCH
    )

    model = FeatureFirstModel()
    selector = build_selector(model, axis=0)
    install_mask(selector)
    masked = selector.apply_mask(inputs)
    zeroed_rows = [
        index
        for index in range(FEATURE_DIM)
        if torch.count_nonzero(masked[index]) == 0
    ]
    print("feature-first mask:", selector.mask.int().tolist())
    print("zeroed feature rows:", zeroed_rows)
    assert zeroed_rows == DROPPED
    assert torch.equal(masked[[0, 2, 4]], inputs[[0, 2, 4]])

    output = selector(inputs)
    assert model.last_input is not None and torch.equal(model.last_input, masked)
    print("forward output shape:", tuple(output.shape))
    assert tuple(output.shape) == (1, BATCH)

    try:
        selector.apply_mask(torch.zeros(FEATURE_DIM - 1, BATCH))
    except ValueError as error:
        print("mismatched feature axis rejected:", error)
    else:
        raise AssertionError("apply_mask must reject a mismatched feature axis")

    batch_first = build_selector(FeatureFirstModel(), axis=-1)
    install_mask(batch_first)
    column_masked = batch_first.apply_mask(inputs.t())
    zeroed_columns = [
        index
        for index in range(FEATURE_DIM)
        if torch.count_nonzero(column_masked[:, index]) == 0
    ]
    print("batch-first (axis -1) zeroed feature columns:", zeroed_columns)
    assert zeroed_columns == DROPPED
    print("axis contract OK")


if __name__ == "__main__":
    main()
