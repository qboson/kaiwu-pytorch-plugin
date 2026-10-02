"""Regression test for the Docker build context of the requirements image.

``requirements/docker-compose.yml`` declares ``context: .`` (relative to the
compose file, i.e. the ``requirements/`` directory) with
``dockerfile: ./docker/Dockerfile``, and the installation docs build with
``cd requirements && docker build -f docker/Dockerfile .``.

Every ``COPY`` source is resolved against that context, never against the
Dockerfile's own directory, so a source that escapes it aborts the documented
build:

    Step 5/6 : COPY ../requirements.txt requirements.txt
    COPY failed: forbidden path outside the build context: ../requirements.txt ()

The check is static so it runs in CI without a Docker daemon.
"""

import pathlib
import re

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
REQUIREMENTS_DIR = REPO_ROOT / "requirements"
DOCKERFILE = REQUIREMENTS_DIR / "docker" / "Dockerfile"
COMPOSE_FILE = REQUIREMENTS_DIR / "docker-compose.yml"


def _build_context() -> pathlib.Path:
    """Return the directory Docker uses as the build context."""
    text = COMPOSE_FILE.read_text(encoding="utf-8")
    match = re.search(r"^\s*context:\s*(\S+)\s*$", text, re.MULTILINE)
    assert match, "docker-compose.yml must declare a build context"
    raw = match.group(1)
    # A relative context is resolved against the compose file's directory.
    return (COMPOSE_FILE.parent / raw).resolve()


def _copy_sources(dockerfile_text: str) -> list[str]:
    """Return every local COPY source (``--from=`` stages excluded)."""
    sources: list[str] = []
    for line in dockerfile_text.splitlines():
        stripped = line.strip()
        if not stripped.upper().startswith("COPY "):
            continue
        parts = stripped.split()[1:]
        parts = [part for part in parts if not part.startswith("--")]
        # Last entry is the destination.
        sources.extend(parts[:-1])
    return sources


def _escapes_context(source: str, context: pathlib.Path) -> bool:
    """Return True when ``source`` resolves outside ``context``."""
    resolved = (context / source).resolve()
    return context not in resolved.parents and resolved != context


@pytest.fixture(scope="module")
def dockerfile_text() -> str:
    return DOCKERFILE.read_text(encoding="utf-8")


def test_copy_sources_stay_inside_the_build_context(dockerfile_text):
    context = _build_context()

    escaping = [s for s in _copy_sources(dockerfile_text) if _escapes_context(s, context)]

    assert not escaping, (
        "COPY sources must live inside the build context "
        f"{context} (context is `requirements/`, not the Dockerfile's directory); "
        f"escaping sources: {escaping}"
    )


def test_copy_sources_exist(dockerfile_text):
    context = _build_context()

    missing = [
        source
        for source in _copy_sources(dockerfile_text)
        if not (context / source).exists()
    ]

    assert not missing, f"COPY sources are missing from the build context: {missing}"


def test_checker_flags_the_original_bug():
    """The check above must reject the path that broke the documented build."""
    context = _build_context()

    assert _escapes_context("../requirements.txt", context)
    assert not _escapes_context("requirements.txt", context)
