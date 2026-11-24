# -*- coding: utf-8 -*-
# Copyright (C) 2022-2025 Beijing QBoson Quantum Technology Co., Ltd.
#
# SPDX-License-Identifier: Apache-2.0

"""Anomaly scoring built exclusively on the public GBRBM API.

Both scoring functions follow the same sign convention: **a higher score means
a more anomalous sample**, which matches ``roc_auc_score`` and
``average_precision_score`` directly.
"""

import numpy as np
import torch

from kaiwu.torch_plugin.gbrbm import GaussianBernoulliRestrictedBoltzmannMachine

SCORE_METHODS = ("free_energy", "reconstruction")


class GBRBMAnomalyScorer:
    """Score continuous samples with a trained Gaussian-Bernoulli RBM.

    Args:
        model (GaussianBernoulliRestrictedBoltzmannMachine): A GBRBM whose visible
            side is Gaussian, i.e. ``is_visible_gaussian=True``.

    Raises:
        ValueError: If the model does not expose Gaussian visible units.
    """

    def __init__(self, model: GaussianBernoulliRestrictedBoltzmannMachine) -> None:
        if not getattr(model, "is_visible_gaussian", False):
            raise ValueError("GBRBMAnomalyScorer requires a model with is_visible_gaussian=True")
        self.model = model

    def _prepare_inputs(self, inputs) -> torch.Tensor:
        """Validate the input matrix and align it to the model device and dtype.

        Args:
            inputs: Array-like or tensor of shape ``(batch, num_gaussian)``.

        Returns:
            torch.Tensor: Contiguous tensor on ``model.device`` and ``model.dtype``.

        Raises:
            ValueError: If the input is not a non-empty finite 2D matrix with the
                expected number of features.
        """
        tensor = torch.as_tensor(inputs)
        if tensor.ndim != 2:
            raise ValueError("inputs must be a 2D array of shape (batch, num_features)")
        if tensor.shape[0] == 0:
            raise ValueError("inputs must contain at least one sample")
        if tensor.shape[1] != self.model.num_gaussian:
            raise ValueError(
                f"inputs must have {self.model.num_gaussian} features, "
                f"got {tensor.shape[1]}"
            )
        if not torch.isfinite(tensor).all():
            raise ValueError("inputs must contain only finite values")
        return tensor.to(device=self.model.device, dtype=self.model.dtype)

    def score_free_energy(self, inputs) -> torch.Tensor:
        """Return the exact free energy ``F(v)`` for every sample.

        ``p(v)`` is proportional to ``exp(-F(v))``, so a low-probability anomaly
        obtains a high score without any sign flip.

        Args:
            inputs: Array-like or tensor of shape ``(batch, num_gaussian)``.

        Returns:
            torch.Tensor: Finite scores of shape ``(batch,)`` without gradients.
        """
        tensor = self._prepare_inputs(inputs)
        with torch.no_grad():
            return self.model.marginal_energy(tensor).detach()

    def score_reconstruction_error(self, inputs) -> torch.Tensor:
        """Return the deterministic mean-field reconstruction error.

        The encoder keeps the conditional activation probability and the decoder
        uses the conditional mean, so the score carries no sampling noise.

        Args:
            inputs: Array-like or tensor of shape ``(batch, num_gaussian)``.

        Returns:
            torch.Tensor: Finite scores of shape ``(batch,)`` without gradients.
        """
        tensor = self._prepare_inputs(inputs)
        with torch.no_grad():
            hidden_prob = self.model.infer_from_gaussian(tensor, binarize=False)[
                :, self.model.num_gaussian :
            ]
            reconstructed = self.model.infer_from_bernoulli(hidden_prob, no_random=True)[
                :, : self.model.num_gaussian
            ]
            return (tensor - reconstructed).square().mean(dim=1).detach()

    def score(self, inputs, method: str = "free_energy") -> torch.Tensor:
        """Dispatch to the requested scoring method.

        Args:
            inputs: Array-like or tensor of shape ``(batch, num_gaussian)``.
            method (str, optional): ``"free_energy"`` or ``"reconstruction"``.

        Returns:
            torch.Tensor: Finite scores of shape ``(batch,)`` without gradients.

        Raises:
            ValueError: If ``method`` is not a supported scoring method.
        """
        if method == "free_energy":
            return self.score_free_energy(inputs)
        if method == "reconstruction":
            return self.score_reconstruction_error(inputs)
        raise ValueError(f"Unsupported score method: {method!r}; expected one of {SCORE_METHODS}")

    def score_numpy(self, inputs, method: str = "free_energy") -> np.ndarray:
        """Return the scores as a CPU ``float64`` array for scikit-learn metrics.

        Args:
            inputs: Array-like or tensor of shape ``(batch, num_gaussian)``.
            method (str, optional): ``"free_energy"`` or ``"reconstruction"``.

        Returns:
            np.ndarray: One-dimensional ``float64`` scores of shape ``(batch,)``.
        """
        scores = self.score(inputs, method)
        return scores.detach().to(device="cpu", dtype=torch.float64).numpy()
