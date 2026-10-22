"""Sampler construction for the NLP BM energy model."""

from __future__ import annotations

from typing import Any


def build_bm_sampler(
    *,
    sampler_type: str = "sa",
    sampler_kwargs: dict[str, Any] | None = None,
) -> Any:
    """Build the sampler used for conditioned BM hidden-state sampling.

    Args:
        sampler_type: Sampler backend. The BM-only example supports ``"sa"``.
        sampler_kwargs: Overrides for the simulated-annealing optimizer.

    Returns:
        A configured Kaiwu simulated-annealing optimizer.

    Raises:
        ValueError: If a sampler other than simulated annealing is requested.
    """

    sampler_kwargs = dict(sampler_kwargs or {})
    if sampler_type != "sa":
        raise ValueError(
            "The BM-only NLP example supports sampler_type='sa' only."
        )

    from kaiwu.classical import SimulatedAnnealingOptimizer

    defaults = {"alpha": 0.95, "size_limit": 10}
    defaults.update(sampler_kwargs)
    return SimulatedAnnealingOptimizer(**defaults)
