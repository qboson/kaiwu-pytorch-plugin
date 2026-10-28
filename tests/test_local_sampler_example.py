# -*- coding: utf-8 -*-
"""Tests for the bring-your-own-sampler example."""

from __future__ import annotations

import itertools
import os
import subprocess
import sys

import numpy as np
import pytest

EXAMPLE_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../example/local_sampler")
)
if EXAMPLE_ROOT not in sys.path:
    sys.path.insert(0, EXAMPLE_ROOT)

from local_sampler import (  # noqa: E402  # pylint: disable=wrong-import-position
    MetropolisSampler,
    build_demo_bm,
    enumerate_energies,
    train_smoke_test,
    verify_sampler_against_enumeration,
)


def test_sampler_output_contract() -> None:
    """solve() returns (reads, N+1) spins in {-1, +1}, best objective first."""
    model = build_demo_bm()
    sampler = MetropolisSampler(num_reads=4, num_sweeps=40, seed=0)
    spins = sampler.solve(model.get_ising_matrix())

    assert spins.shape == (4, 5)
    assert set(np.unique(spins)).issubset({-1, 1})

    matrix = model.get_ising_matrix()
    values = [float(row @ matrix @ row) for row in spins]
    assert values == sorted(values, reverse=True)


def test_sampler_returns_maximisers_of_the_ising_form() -> None:
    """Every returned spin vector reaches the exhaustive maximum of s^T M s."""
    model = build_demo_bm()
    matrix = model.get_ising_matrix()
    exhaustive = max(
        float(np.array(spins) @ matrix @ np.array(spins))
        for spins in itertools.product([-1, 1], repeat=matrix.shape[0])
    )

    sampler = MetropolisSampler(num_reads=3, num_sweeps=60, seed=1)
    spins = sampler.solve(matrix)

    for row in spins:
        assert np.isclose(float(row @ matrix @ row), exhaustive)


def test_verification_matches_enumeration() -> None:
    """model.sample(sampler) decodes to states with the minimal energy."""
    model = build_demo_bm()
    sampler = MetropolisSampler(num_reads=3, num_sweeps=60, seed=2)
    report = verify_sampler_against_enumeration(model, sampler, num_draws=1)

    energies = enumerate_energies(model)
    assert report["all_draws_minimal"] is True
    assert report["num_minimal"] == report["num_sampled"]
    for state in report["sampled"]:
        assert np.isclose(energies[state], report["ground_energy"])


def test_sampler_rejects_invalid_matrices() -> None:
    """Shapes and non-finite coefficients are rejected."""
    sampler = MetropolisSampler(num_reads=2, num_sweeps=10)
    with pytest.raises(ValueError):
        sampler.solve(np.zeros((2, 3)))
    with pytest.raises(ValueError):
        sampler.solve(np.array([[0.0, np.nan], [0.0, 0.0]]))


def test_sampler_validates_settings() -> None:
    """Invalid read/sweep/temperature settings fail early."""
    with pytest.raises(ValueError):
        MetropolisSampler(num_reads=0)
    with pytest.raises(ValueError):
        MetropolisSampler(num_sweeps=0)
    with pytest.raises(ValueError):
        MetropolisSampler(initial_temperature=0.1, final_temperature=1.0)


def test_training_smoke_test_runs() -> None:
    """A few steps of the documented loop return finite objectives."""
    result = train_smoke_test(steps=2, reads=2, sweeps=10, seed=0)
    assert set(result) == {"first_objective", "last_objective"}
    assert all(np.isfinite(value) for value in result.values())


def test_cli_entry_point_passes() -> None:
    """The documented command verifies the sampler and exits successfully."""
    script = os.path.join(EXAMPLE_ROOT, "local_sampler.py")
    proc = subprocess.run(
        [sys.executable, script, "--steps", "2", "--reads", "3"],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=EXAMPLE_ROOT,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "PASS" in proc.stdout
    assert "ground energy" in proc.stdout


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
