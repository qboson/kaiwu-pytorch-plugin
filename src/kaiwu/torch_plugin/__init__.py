# -*- coding: utf-8 -*-
"""Kaiwu-PyTorch-Plugin public API."""

from .dbn import UnsupervisedDBN
from .full_boltzmann_machine import BoltzmannMachine
from .maifs import FeatureSelectionWrapper, QuadraticLinearSolver
from .qdiffusion import EnergyModel, QDiffusion, QDiffusionConfig
from .qvae import QVAE
from .restricted_boltzmann_machine import RestrictedBoltzmannMachine

from .usage_stats import (
    enable_usage_stats,
    disable_usage_stats,
    is_usage_stats_enabled,
)

__version__ = "0.2.0"

__all__ = [
    "RestrictedBoltzmannMachine",
    "BoltzmannMachine",
    "EnergyModel",
    "QVAE",
    "UnsupervisedDBN",
    "QDiffusion",
    "QDiffusionConfig",
    "FeatureSelectionWrapper",
    "QuadraticLinearSolver",
    "enable_usage_stats",
    "disable_usage_stats",
    "is_usage_stats_enabled",
]
