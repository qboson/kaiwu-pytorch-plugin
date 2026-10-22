"""Tests for the license gate in the QVAE-MNIST example entry point."""

import os
import sys

import pytest

for _module in ("gif", "imageio", "torchvision", "torchmetrics", "pandas", "seaborn", "tqdm"):
    pytest.importorskip(_module)

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../example/qvae_mnist"))
)
import run_pipeline  # noqa: E402  pylint: disable=wrong-import-position


def test_missing_credentials_exit_with_instructions(monkeypatch):
    monkeypatch.delenv("USER_ID", raising=False)
    monkeypatch.delenv("SDK_CODE", raising=False)

    with pytest.raises(SystemExit) as excinfo:
        run_pipeline._ensure_license()  # pylint: disable=protected-access

    message = str(excinfo.value)
    assert "Kaiwu SDK license" in message
    assert "USER_ID" in message and "SDK_CODE" in message


def test_main_checks_the_license_before_parsing(monkeypatch):
    monkeypatch.delenv("USER_ID", raising=False)
    monkeypatch.delenv("SDK_CODE", raising=False)

    with pytest.raises(SystemExit) as excinfo:
        run_pipeline.main()

    assert "Kaiwu SDK license" in str(excinfo.value)
