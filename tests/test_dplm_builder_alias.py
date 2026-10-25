"""Contract test for the backwards-compatible ``build_dplm_qdiffusion`` alias."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
QDIFFUSION_EXAMPLE = REPO / "example" / "qdiffusion"


def _import_builder():
    sys.path.insert(0, str(QDIFFUSION_EXAMPLE))
    try:
        from dplm.utils import dplm_builder
    finally:
        sys.path.remove(str(QDIFFUSION_EXAMPLE))
    return dplm_builder


def test_forwards_every_positional_and_keyword_argument(monkeypatch):
    builder = _import_builder()
    sentinel = object()
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def fake_build(*args, **kwargs):
        calls.append((args, kwargs))
        return sentinel

    monkeypatch.setattr(builder, "build_qdiffusion", fake_build)

    result = builder.build_dplm_qdiffusion(1, 2, num_candidates=7, freeze_proposal=True)

    assert result is sentinel
    assert calls == [((1, 2), {"num_candidates": 7, "freeze_proposal": True})]


def test_returns_whatever_the_canonical_builder_returns(monkeypatch):
    builder = _import_builder()
    monkeypatch.setattr(builder, "build_qdiffusion", lambda *args, **kwargs: "built")

    assert builder.build_dplm_qdiffusion() == "built"


def test_is_exported_from_the_utils_package():
    _import_builder()
    utils = sys.modules["dplm.utils"]

    assert utils.build_dplm_qdiffusion is not None
    assert utils.__all__ == ["build_dplm_qdiffusion", "build_qdiffusion"]


@pytest.mark.parametrize("kwargs", [{"dtype": "float32"}, {"device": "cpu"}])
def test_forwards_optional_construction_options(monkeypatch, kwargs):
    builder = _import_builder()
    seen: list[dict[str, object]] = []
    monkeypatch.setattr(
        builder, "build_qdiffusion", lambda *args, **kw: seen.append(kw) or kw
    )

    assert builder.build_dplm_qdiffusion(**kwargs) == kwargs
    assert seen == [kwargs]
