# -*- coding: utf-8 -*-
# Copyright (C) 2022-2026 Beijing QBoson Quantum Technology Co., Ltd.
#
# SPDX-License-Identifier: Apache-2.0
"""
Quantum Stochastic Variational Inference (Q-SVI) training engine.

This module re-exports Q_SVI and QSVI from kaiwu.torch_plugin.qvae for
backward compatibility, modular design, andPyro-style inference.
"""

from .qvae import Q_SVI, QSVI

__all__ = ["Q_SVI", "QSVI"]
