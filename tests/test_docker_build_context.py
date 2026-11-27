# -*- coding: utf-8 -*-
# Copyright (C) 2022-2026 Beijing QBoson Quantum Technology Co., Ltd.
#
# SPDX-License-Identifier: Apache-2.0
"""Keep the Dockerfile inside the build context declared by docker-compose.

``requirements/docker-compose.yml`` builds with ``context: .`` (the
``requirements`` directory). A ``COPY ../requirements.txt`` therefore fails
before any dependency is installed, which breaks the Docker installation path
from the README.
"""

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
REQUIREMENTS_DIR = REPO_ROOT / "requirements"
COMPOSE_FILE = REQUIREMENTS_DIR / "docker-compose.yml"
DOCKERFILE = REQUIREMENTS_DIR / "docker" / "Dockerfile"


def build_context() -> Path:
    """Return the absolute build context declared by docker-compose."""
    text = COMPOSE_FILE.read_text(encoding="utf-8")
    match = re.search(r"^\s*context:\s*(\S+)\s*$", text, re.MULTILINE)
    assert match, "docker-compose.yml has no build context"
    context = (REQUIREMENTS_DIR / match.group(1)).resolve()
    return context


def copy_sources(dockerfile: Path) -> list[str]:
    """Return the source paths of COPY/ADD instructions without stages."""
    sources = []
    for line in dockerfile.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split()
        if len(parts) < 3 or parts[0].upper() not in ("COPY", "ADD"):
            continue
        if parts[1].startswith("--from=") or parts[1].startswith("http"):
            continue
        sources.extend(parts[1:-1])
    return sources


def test_dockerfile_sources_stay_inside_the_build_context():
    """Every COPY/ADD source must resolve inside the compose context."""
    context = build_context()
    assert context == REQUIREMENTS_DIR.resolve()

    for source in copy_sources(DOCKERFILE):
        resolved = (context / source).resolve()
        assert context in resolved.parents or resolved == context, (
            f"COPY source {source!r} escapes the build context {context}"
        )


def test_requirements_file_exists_in_the_context():
    """The Dockerfile copies ``requirements.txt`` from the context root."""
    assert (build_context() / "requirements.txt").is_file()
