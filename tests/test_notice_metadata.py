"""Guard the NOTICE attribution metadata against template placeholders."""

from __future__ import annotations

from pathlib import Path

NOTICE_PATH = Path(__file__).resolve().parents[1] / "NOTICE"
PLACEHOLDER_HOSTS = ("yourcompany.com", "example.com", "changeme", "<company")


def _notice_text() -> str:
    return NOTICE_PATH.read_text(encoding="utf-8")


def test_notice_has_no_placeholder_hosts() -> None:
    text = _notice_text()
    for host in PLACEHOLDER_HOSTS:
        assert host not in text, f"NOTICE still contains the placeholder {host!r}"


def test_notice_names_the_company_site() -> None:
    assert "https://www.qboson.com/" in _notice_text()


def test_notice_ends_with_a_newline() -> None:
    assert _notice_text().endswith("\n")
