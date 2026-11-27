# -*- coding: utf-8 -*-
# Copyright (C) 2022-2025 Beijing QBoson Quantum Technology Co., Ltd.
#
# SPDX-License-Identifier: Apache-2.0

"""Sampler construction for the GBRBM anomaly detection case.

The default sampler is the local ``SimulatedAnnealingOptimizer``, which never
submits a cloud task. The ``cim`` entry point is kept ready for licensed
hardware runs and normalizes the Kaiwu 1.3.1-1.4.3 interface differences through
``kaiwu.torch_plugin._kaiwu_compat``.
"""

import os

from ad_config import SamplerConfig

import kaiwu as kw
from kaiwu.torch_plugin._kaiwu_compat import PrecisionReducer, normalize_task_mode

# Accepted (user id, sdk code) environment variable pairs, in priority order.
LICENSE_ENV_PAIRS = (
    ("USER_ID", "SDK_CODE"),
    ("LICENSE_USER_ID", "LICENSE_SDK_CODE"),
)


def init_kaiwu_license_from_env() -> bool:
    """Initialize the Kaiwu license from environment variables when available.

    Returns:
        bool: ``True`` when a credential pair was found and ``kw.license.init``
        was invoked, ``False`` when the process keeps its current license state.
    """
    for user_key, code_key in LICENSE_ENV_PAIRS:
        user_id = os.environ.get(user_key)
        sdk_code = os.environ.get(code_key)
        if user_id and sdk_code:
            kw.license.init(user_id, sdk_code)
            return True
    return False


def _build_sa_sampler(config: SamplerConfig):
    """Build the local simulated annealing optimizer (no cloud submission)."""
    return kw.classical.SimulatedAnnealingOptimizer(
        initial_temperature=config.sa_initial_temperature,
        alpha=config.sa_alpha,
        cutoff_temperature=config.sa_cutoff_temperature,
        iterations_per_t=config.sa_iterations_per_t,
        size_limit=config.sa_size_limit,
        rand_seed=config.seed,
    )


def _build_cim_sampler(config: SamplerConfig):
    """Build the CIM optimizer, optionally wrapped by ``PrecisionReducer``."""
    if config.cim_tmp_dir:
        kw.common.CheckpointManager.save_dir = config.cim_tmp_dir
    optimizer = kw.cim.CIMOptimizer(
        task_name=config.cim_task_name,
        wait=config.cim_wait,
        project_no=config.cim_project_no,
        task_mode=normalize_task_mode(config.cim_task_mode),
        sample_number=config.cim_sample_number,
    )
    if not config.cim_use_precision_reducer:
        return optimizer
    return PrecisionReducer(
        optimizer,
        precision=config.cim_precision,
        truncated_precision=config.cim_truncated_precision,
        target_bits=config.cim_target_bits,
    )


def build_sampler(config: SamplerConfig):
    """Build the Ising sampler selected by ``config.sampler_type``.

    Args:
        config (SamplerConfig): Sampler settings.

    Returns:
        The constructed Kaiwu optimizer.

    Raises:
        ValueError: If ``sampler_type`` is neither ``"sa"`` nor ``"cim"``.
    """
    if config.sampler_type == "sa":
        return _build_sa_sampler(config)
    if config.sampler_type == "cim":
        return _build_cim_sampler(config)
    raise ValueError(f"Unsupported sampler_type: {config.sampler_type!r}")
