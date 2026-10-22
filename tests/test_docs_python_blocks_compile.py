"""Regression test: every Python snippet in the docs must at least compile.

Documentation snippets are copied and executed by users, so a fence that does
not compile is a defect even if the page renders fine.  This caught the CIM
sampler snippet in ``getting_started/introduction.md``, whose continuation
lines contained U+00A0 non-breaking spaces pasted from a browser:

    SyntaxError: invalid non-printable character U+00A0
"""

import pathlib
import re
import textwrap

import pytest

DOCS_ROOT = pathlib.Path(__file__).resolve().parents[1] / "docs" / "source"


def _python_blocks(text):
    for match in re.finditer(r"```(?:python|py)\n(.*?)```", text, re.S):
        yield match.start(), match.group(1)
    for match in re.finditer(r"\{code-block\}\s+python\n(.*?)(?=\n\S|\Z)", text, re.S):
        yield match.start(), textwrap.dedent(match.group(1))


def _collect():
    blocks = []
    for path in sorted(DOCS_ROOT.rglob("*.md")):
        text = path.read_text(encoding="utf-8")
        for position, body in _python_blocks(text):
            line = text.count("\n", 0, position) + 1
            blocks.append(pytest.param(body, id=f"{path.name}:{line}"))
    return blocks


@pytest.mark.parametrize("block", _collect())
def test_docs_python_block_compiles(block):
    compile(block, "<docs>", "exec")
