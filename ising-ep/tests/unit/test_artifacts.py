"""Tests for the ``.env`` lookup order in ``load_dotenv``; only fake credentials."""

# Test functions document intent in their summary.
# pylint: disable=missing-param-doc,missing-return-doc

from __future__ import annotations

from pathlib import Path

import pytest

from examples.mnist_kaiwu import artifacts


@pytest.fixture(name="layout")
def fixture_layout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    """Point the module at a fake project inside a fake workspace."""
    project = tmp_path / "workspace" / "proj"
    module = project / "examples" / "mnist_kaiwu" / "artifacts.py"
    monkeypatch.setattr(artifacts, "__file__", str(module))
    monkeypatch.delenv("KAIWU_USER_ID", raising=False)
    monkeypatch.delenv("KAIWU_SDK_CODE", raising=False)
    project.mkdir(parents=True)
    return project, project.parent


def _write(directory: Path, user: str) -> None:
    """Write a fake ``.env`` into ``directory``."""
    (directory / ".env").write_text(
        f"KAIWU_USER_ID={user}\nKAIWU_SDK_CODE=fake-code\n", encoding="utf-8"
    )


def test_project_env_wins_over_workspace_env(layout: tuple[Path, Path]) -> None:
    """The project ``.env`` is found first."""
    project, workspace = layout
    _write(project, "project-user")
    _write(workspace, "workspace-user")

    assert artifacts.load_dotenv() == ("project-user", "fake-code")


def test_falls_back_to_the_parent_directory(layout: tuple[Path, Path]) -> None:
    """Without a project ``.env`` the workspace one is used."""
    _, workspace = layout
    _write(workspace, "workspace-user")

    assert artifacts.load_dotenv() == ("workspace-user", "fake-code")


def test_returns_none_when_no_env_exists(layout: tuple[Path, Path]) -> None:
    """No file and no variables means no credentials."""
    assert layout
    assert artifacts.load_dotenv() is None


def test_environment_variables_beat_the_file(
    layout: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A shell export overrides whatever the file says."""
    _, workspace = layout
    _write(workspace, "file-user")
    monkeypatch.setenv("KAIWU_USER_ID", "shell-user")

    assert artifacts.load_dotenv() == ("shell-user", "fake-code")


def test_explicit_path_skips_the_search(layout: tuple[Path, Path], tmp_path: Path) -> None:
    """An explicit path is used alone."""
    project, _ = layout
    _write(project, "project-user")
    other = tmp_path / "other"
    other.mkdir()
    _write(other, "explicit-user")

    assert artifacts.load_dotenv(other / ".env") == ("explicit-user", "fake-code")
