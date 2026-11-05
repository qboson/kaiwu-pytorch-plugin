# -*- coding: utf-8 -*-
"""Clone commands in the docs must match the directory used by the next cd."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLONE_RE = re.compile(r"git clone\s+(https?://\S+)")
CD_RE = re.compile(r"^\s*cd\s+([^\s`]+)")


def _documents() -> list[Path]:
    documents = [ROOT / "README.md", ROOT / "README_ZH.md"]
    documents.extend(sorted((ROOT / "docs").rglob("*.md")))
    return [path for path in documents if path.exists()]


def _mismatches(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    mismatches = []
    for index, line in enumerate(lines):
        match = CLONE_RE.search(line)
        if match is None:
            continue
        url = match.group(1).rstrip("/")
        clone_name = url[:-4] if url.endswith(".git") else url
        clone_name = clone_name.rsplit("/", 1)[-1]
        for following in lines[index + 1 : index + 5]:
            cd_match = CD_RE.match(following)
            if cd_match is None:
                continue
            cd_path = cd_match.group(1).strip("\"'`")
            cd_name = cd_path.split("/", 1)[0]
            if cd_name != clone_name:
                mismatches.append(
                    f"{path.relative_to(ROOT)}:{index + 1}: clone creates "
                    f"{clone_name!r} but {following.strip()!r} uses {cd_name!r}"
                )
            break
    return mismatches


def test_clone_commands_match_the_following_cd() -> None:
    """Linux is case-sensitive: a case mismatch breaks the quick start."""
    mismatches = []
    for document in _documents():
        mismatches.extend(_mismatches(document))

    assert not mismatches, "clone/cd case mismatch:\n" + "\n".join(mismatches)
