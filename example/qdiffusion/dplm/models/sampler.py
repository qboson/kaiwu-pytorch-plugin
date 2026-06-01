"""Sampler-construction helpers for solver-backed BM rerankers."""

from __future__ import annotations

import importlib
from typing import Any


def build_bm_sampler(
    *,
    sampler_type: str,
    sampler_kwargs: dict[str, Any] | None = None,
) -> Any:
    """Builds one solver object for BM hidden-state search."""
    sampler_kwargs = dict(sampler_kwargs or {})
    if sampler_type == "sa":
        from kaiwu.classical import SimulatedAnnealingOptimizer

        # Keep the local SA path self-contained so the example can run without
        # any remote solver configuration.
        default_kwargs = {
            "alpha": 0.95,
            "size_limit": 10,
        }
        default_kwargs.update(sampler_kwargs)
        return SimulatedAnnealingOptimizer(**default_kwargs)

    if sampler_type == "cim":
        import kaiwu as kw

        # CIM lives behind the main ``kaiwu.cim`` namespace and may expose
        # different public class names depending on the runtime version.
        kaiwu_cim = importlib.import_module("kaiwu.cim")
        optimizer_cls = getattr(kaiwu_cim, "CIMOptimizer", None)
        if optimizer_cls is None:
            optimizer_cls = getattr(kaiwu_cim, "Optimizer", None)
        if optimizer_cls is None:
            raise ValueError(
                "kaiwu.cim does not expose CIMOptimizer/Optimizer in "
                "the current runtime."
            )

        tmp_dir = sampler_kwargs.get("tmp_dir")
        if tmp_dir:
            # Some CIM runtimes require a checkpoint/cache directory before the
            # optimizer can submit or resume remote jobs.
            kw.common.CheckpointManager.save_dir = tmp_dir

        task_mode = sampler_kwargs.get("task_mode")
        if isinstance(task_mode, str):
            # Workflow configs keep task_mode as a user-friendly string; convert
            # it lazily only when the runtime exports the enum.
            task_mode_enum = getattr(kaiwu_cim, "TaskMode", None)
            if task_mode_enum is not None and hasattr(task_mode_enum, task_mode):
                task_mode = getattr(task_mode_enum, task_mode)

        optimizer_kwargs = {
            "task_name": sampler_kwargs.get("task_name", "qdiffusion_bm"),
            "wait": sampler_kwargs.get("wait", False),
            "interval": sampler_kwargs.get("interval", 1),
        }
        optional_optimizer_keys = (
            "project_no",
            "sample_number",
        )
        for key in optional_optimizer_keys:
            if key in sampler_kwargs and sampler_kwargs[key] is not None:
                optimizer_kwargs[key] = sampler_kwargs[key]
        if task_mode is not None:
            optimizer_kwargs["task_mode"] = task_mode

        sampler = optimizer_cls(**optimizer_kwargs)
        if not sampler_kwargs.get("use_precision_reducer", False):
            return sampler

        # PrecisionReducer is optional: keep the common path simple, but allow
        # callers to wrap the remote solver when they explicitly need it.
        precision_reducer = getattr(kaiwu_cim, "PrecisionReducer", None)
        if precision_reducer is None:
            raise ValueError(
                "use_precision_reducer=True but kaiwu.cim.PrecisionReducer "
                "is unavailable in the current runtime."
            )
        precision_kwargs = {
            "precision": sampler_kwargs.get("precision", 8),
            "truncated_precision": sampler_kwargs.get("truncated_precision", 10),
            "target_bits": sampler_kwargs.get("target_bits", 550),
            "only_feasible_solution": sampler_kwargs.get(
                "only_feasible_solution", False
            ),
        }
        return precision_reducer(sampler, **precision_kwargs)

    raise ValueError(f"Unsupported BM sampler_type for DPLM examples: {sampler_type}")
