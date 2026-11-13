# -*- coding: utf-8 -*-
"""Kaiwu-PyTorch-Plugin public API."""

import sys

from .bm import (
    abstract_boltzmann_machine,
    dbn,
    full_boltzmann_machine,
    gbrbm,
    restricted_boltzmann_machine,
)
from .bm.dbn import UnsupervisedDBN
from .bm.full_boltzmann_machine import BoltzmannMachine
from .bm.restricted_boltzmann_machine import RestrictedBoltzmannMachine
from .optimization import maifs
from .optimization.maifs import FeatureSelectionWrapper, QuadraticLinearSolver
from .optimization.maifs import plugin, qubo
from .quantum_ai import _qdiffusion_sampling, qdeq, qdiffusion, qvae, qvae_dist_util
from .quantum_ai.qdeq import QDEQ, QDEQConfig
from .quantum_ai.qdiffusion import EnergyModel, QDiffusion, QDiffusionConfig
from .quantum_ai.qvae import QVAE
from .usage_stats import (
    enable_usage_stats,
    disable_usage_stats,
    is_usage_stats_enabled,
)

# Preserve existing import paths and names stored in model files.
for _name, _module in {
    "abstract_boltzmann_machine": abstract_boltzmann_machine,
    "restricted_boltzmann_machine": restricted_boltzmann_machine,
    "full_boltzmann_machine": full_boltzmann_machine,
    "gbrbm": gbrbm,
    "dbn": dbn,
    "qdeq": qdeq,
    "qdiffusion": qdiffusion,
    "_qdiffusion_sampling": _qdiffusion_sampling,
    "qvae": qvae,
    "qvae_dist_util": qvae_dist_util,
    "maifs": maifs,
    "maifs.plugin": plugin,
    "maifs.qubo": qubo,
}.items():
    _LEGACY_NAME = f"{__name__}.{_name}"
    sys.modules[_LEGACY_NAME] = _module
    for _value in vars(_module).values():
        if getattr(_value, "__module__", None) == _module.__name__:
            _value.__module__ = _LEGACY_NAME

__version__ = "0.3.0"

__all__ = [
    "RestrictedBoltzmannMachine",
    "BoltzmannMachine",
    "EnergyModel",
    "QVAE",
    "UnsupervisedDBN",
    "QDEQ",
    "QDEQConfig",
    "QDiffusion",
    "QDiffusionConfig",
    "FeatureSelectionWrapper",
    "QuadraticLinearSolver",
    "enable_usage_stats",
    "disable_usage_stats",
    "is_usage_stats_enabled",
]
