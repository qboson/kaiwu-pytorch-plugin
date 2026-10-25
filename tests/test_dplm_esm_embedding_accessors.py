"""Contract test for the ESM wrapper embedding accessors."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
ESM_PATCH = REPO / "example" / "qdiffusion" / "dplm" / "models" / "esm_patch.py"


def _load_esm_patch():
    spec = importlib.util.spec_from_file_location("dplm_esm_embeddings_uut", ESM_PATCH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def esm_patch():
    return _load_esm_patch()


class _Embeddings:
    def __init__(self) -> None:
        self.position_embeddings = object()
        self.word_embeddings = object()


class _Model:
    def __init__(self) -> None:
        self.embeddings = _Embeddings()


def test_bare_model_exposes_the_position_table(esm_patch):
    model = _Model()

    assert (
        esm_patch._ModifiedEsmModel.get_position_embeddings(model)
        is model.embeddings.position_embeddings
    )


def test_bare_model_replaces_only_the_word_table(esm_patch):
    model = _Model()
    original_position = model.embeddings.position_embeddings
    replacement = object()

    esm_patch._ModifiedEsmModel.set_input_embeddings(model, replacement)

    assert model.embeddings.word_embeddings is replacement
    assert model.embeddings.position_embeddings is original_position


def test_bare_model_refuses_to_resize_position_embeddings(esm_patch):
    with pytest.raises(NotImplementedError, match="position embeddings"):
        esm_patch._ModifiedEsmModel.resize_position_embeddings(_Model(), 8)


class _InnerEsm:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []

    def get_position_embeddings(self):
        self.calls.append(("get", None))
        return "position-table"

    def set_input_embeddings(self, value):
        self.calls.append(("set", value))

    def resize_position_embeddings(self, count):
        self.calls.append(("resize", count))
        raise NotImplementedError("inner refuses resizing")


class _Wrapper:
    def __init__(self) -> None:
        self.esm = _InnerEsm()


def test_wrapper_delegates_the_position_table(esm_patch):
    wrapper = _Wrapper()

    assert esm_patch._EsmForDPLM.get_position_embeddings(wrapper) == "position-table"
    assert wrapper.esm.calls == [("get", None)]


def test_wrapper_delegates_embedding_replacement(esm_patch):
    wrapper = _Wrapper()
    replacement = object()

    esm_patch._EsmForDPLM.set_input_embeddings(wrapper, replacement)

    assert wrapper.esm.calls == [("set", replacement)]


def test_wrapper_delegates_the_resize_rejection(esm_patch):
    wrapper = _Wrapper()

    with pytest.raises(NotImplementedError, match="inner refuses"):
        esm_patch._EsmForDPLM.resize_position_embeddings(wrapper, 12)

    assert wrapper.esm.calls == [("resize", 12)]
