# -*- coding: utf-8 -*-
"""The documented RISC-V glibc floor must satisfy the pinned torch wheel."""

from __future__ import annotations

import re
from pathlib import Path

RELEASE_NOTES = (
    Path(__file__).resolve().parents[1]
    / "docs"
    / "source"
    / "about"
    / "release_notes.md"
)


def test_riscv_glibc_floor_covers_the_torch_wheel() -> None:
    """A manylinux_2_N torch wheel cannot be installed on glibc older than N."""
    text = RELEASE_NOTES.read_text(encoding="utf-8")

    wheel = re.search(
        r"/torch-[^/\s]+-manylinux_2_(\d+)_riscv64\.whl",
        text,
    )
    assert wheel is not None, "the RISC-V torch wheel URL is missing"

    requirement = re.search(r"glibc ≥ (\d+)\.(\d+)", text)
    assert requirement is not None, "the RISC-V glibc requirement is missing"

    documented_minor = int(requirement.group(2))
    wheel_minor = int(wheel.group(1))
    assert documented_minor >= wheel_minor, (
        f"documented glibc 2.{documented_minor} cannot install the "
        f"manylinux_2_{wheel_minor} torch wheel"
    )
