# -*- coding: utf-8 -*-
"""GBRBM public state methods must accept mixed-dtype inputs."""

from __future__ import annotations

import numpy as np
import torch

from kaiwu.torch_plugin.gbrbm import GaussianBernoulliRestrictedBoltzmannMachine


def _double_model() -> GaussianBernoulliRestrictedBoltzmannMachine:
    return GaussianBernoulliRestrictedBoltzmannMachine(2, 2).double()


def test_energy_accepts_float32_states_for_double_model() -> None:
    model = _double_model()
    states = torch.randn(2, 4, dtype=torch.float32)

    energy = model.energy(states)

    assert energy.shape == (2,)
    assert torch.isfinite(energy).all()


def test_marginal_energy_accepts_float32_states() -> None:
    model = _double_model()
    gaussian = torch.randn(2, 2, dtype=torch.float32)

    energy = model.marginal_energy(gaussian)

    assert energy.shape == (2,)
    assert torch.isfinite(energy).all()


def test_infer_from_gaussian_accepts_float32_states() -> None:
    model = _double_model()
    gaussian = torch.randn(2, 2, dtype=torch.float32)

    states = model.infer_from_gaussian(gaussian, no_random=True)

    assert states.shape == (2, 4)
    assert torch.isfinite(states).all()


def test_infer_from_bernoulli_accepts_float32_states() -> None:
    model = _double_model()
    bernoulli = torch.ones(2, 2, dtype=torch.float32)

    states = model.infer_from_bernoulli(bernoulli, no_random=True)

    assert states.shape == (2, 4)
    assert torch.isfinite(states).all()


def test_positive_phase_accepts_float32_states() -> None:
    model = _double_model()
    gaussian = torch.randn(2, 2, dtype=torch.float32)

    energy = model.positive_phase_energy_expectation(gaussian, enable_grad=False)

    assert energy.shape == (2,)
    assert torch.isfinite(energy).all()


def test_gibbs_sample_accepts_float32_initial_state() -> None:
    model = _double_model()
    gaussian = torch.zeros(2, 2, dtype=torch.float32)

    samples = model.gibbs_sample(n_step=1, s_gaussian=gaussian)

    assert samples.shape == (2, 4)
    assert torch.isfinite(samples).all()


def test_energy_accepts_numpy_states() -> None:
    model = _double_model()
    states = np.zeros((2, 4), dtype=np.float32)

    energy = model.energy(states)

    assert energy.shape == (2,)
    assert torch.isfinite(energy).all()
