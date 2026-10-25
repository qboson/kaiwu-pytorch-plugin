"""Documentation snippets building a CIM sampler must set the SDK save directory.

The Kaiwu SDK refuses to construct a ``CIMOptimizer`` while
``kaiwu.common.CheckpointManager.save_dir`` is unset::

    ValueError: The save directory is required

(verified against ``kaiwu==1.3.1``; raised in ``kaiwu/cim/_optimizer_adapter.py``
before any credential handling). Every runnable example in ``example/`` and the
Quick Start set it right before constructing the sampler, but the snippets in the
tutorials and the introduction page omitted it, so copying them failed with that
``ValueError`` instead of reaching the documented credential step. This guard keeps
the documented snippets consistent with the SDK prerequisite.
"""

import pathlib
import re
import unittest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
DOCS_ROOT = REPO_ROOT / "docs" / "source"
PYTHON_BLOCK = re.compile(r"```(?:python|py)\n(.*?)```", re.S)
SAVE_DIR_ASSIGNMENT = "CheckpointManager.save_dir"


def _python_blocks():
    """Yield ``(path, index, source)`` for every Python block under ``docs/source``."""
    for doc in sorted(DOCS_ROOT.rglob("*.md")):
        if "locale" in doc.parts:
            continue
        text = doc.read_text(encoding="utf-8")
        for index, block in enumerate(PYTHON_BLOCK.findall(text)):
            yield doc, index, block


class TestDocumentedCimSamplerSnippets(unittest.TestCase):
    """The CIM sampler blocks users copy must be constructible as written."""

    def test_cim_blocks_are_discovered(self):
        """Guard against the scan silently finding nothing to check."""
        blocks = [block for _, _, block in _python_blocks() if "CIMOptimizer(" in block]
        self.assertGreaterEqual(len(blocks), 3)

    def test_cim_blocks_set_the_save_directory_before_construction(self):
        """A CIM snippet is only runnable once the save directory is configured."""
        missing = []
        for doc, index, block in _python_blocks():
            construction = block.find("CIMOptimizer(")
            if construction == -1:
                continue
            assignment = block.find(SAVE_DIR_ASSIGNMENT)
            if assignment == -1 or assignment > construction:
                missing.append(f"{doc.relative_to(REPO_ROOT)} block #{index}")
        self.assertEqual(
            missing,
            [],
            "CIM snippets must set kw.common.CheckpointManager.save_dir before "
            "constructing CIMOptimizer; missing in: " + ", ".join(missing),
        )


if __name__ == "__main__":
    unittest.main()
