"""Compile authored Python snippets without executing documentation or examples.

MyST supplies Markdown structure and directive options. External literalinclude
and generated API/eval-rst content are outside this inline-snippet guard.
"""

import ast
from dataclasses import dataclass
from pathlib import Path
import re
import tempfile
import unittest

from myst_parser.config.main import MdParserConfig
from myst_parser.mdit_to_docutils.base import DocutilsRenderer
from myst_parser.parsers.directives import MarkupError, parse_directive_text
from myst_parser.parsers.mdit import create_md_parser
from sphinx.directives.code import CodeBlock, dedent_lines


@dataclass
class PythonBlock:
    code: str
    first_line: int
    fence_line: int


def documentation_config(path):
    """Read literal parser settings without importing the Sphinx configuration."""
    settings = {}
    wanted = {"myst_enable_extensions", "myst_fence_as_directive"}
    for statement in ast.parse(path.read_bytes().decode("utf-8")).body:
        if isinstance(statement, ast.Assign):
            for target in statement.targets:
                if isinstance(target, ast.Name) and target.id in wanted:
                    settings[target.id] = ast.literal_eval(statement.value)
    return MdParserConfig(
        enable_extensions=set(settings.get("myst_enable_extensions", [])),
        fence_as_directive=set(settings.get("myst_fence_as_directive", [])),
    )


def python_blocks(source, config):
    """Extract explicit python/py fences using MyST, without rendering directives."""
    parser = create_md_parser(config, DocutilsRenderer)
    blocks = []
    for token in parser.parse(source, {}):
        if token.type not in {"fence", "colon_fence"}:
            continue
        parts = token.info.strip().split(maxsplit=1)
        name = parts[0] if parts else ""
        argument = parts[1] if len(parts) > 1 else ""
        first_line = token.map[0] + 2
        if name in {"python", "py"}:
            code = token.content
        elif name == "{code-block}" and argument.split()[:1] in [["python"], ["py"]]:
            try:
                parsed = parse_directive_text(
                    CodeBlock, argument, token.content, line=token.map[0] + 1
                )
            except MarkupError as error:
                raise ValueError(f"line {token.map[0] + 1}: {error}") from error
            if parsed.warnings:
                messages = "; ".join(warning.msg for warning in parsed.warnings)
                raise ValueError(f"line {token.map[0] + 1}: {messages}")
            code = "\n".join(parsed.body)
            first_line += parsed.body_offset
            if "dedent" in parsed.options:
                code = "".join(dedent_lines(code.splitlines(True), parsed.options["dedent"]))
        else:
            # Container directives can contain Markdown, unlike literal code fences.
            # Do not silently promise coverage of an unsupported nested directive.
            if name.startswith("{") and name not in {
                "{code-block}", "{literalinclude}", "{eval-rst}"
            } and re.search(
                r"(?m)^\s*(?:`{3,}|~{3,}|:{3,})\s*(?:python\b|py\b|\{code-block\}\s+(?:python|py)\b)",
                token.content,
            ):
                raise ValueError(f"line {token.map[0] + 1}: nested Python in unsupported {name}")
            continue
        blocks.append(PythonBlock(code, first_line, token.map[0] + 1))
    return blocks


def syntax_failure(block, document, number):
    """Return a document-line diagnostic; compiling never executes the snippet."""
    try:
        compile("\n" * (block.first_line - 1) + block.code,
                str(document), "exec", dont_inherit=True)
    except SyntaxError as error:
        return (f"{document}:{error.lineno}:{error.offset}: block {number} "
                f"(fence line {block.fence_line}): {error.msg}")
    return None


def check_corpus(root, config):
    documents = sorted(path for path in root.rglob("*")
                       if path.is_file() and path.suffix in {".md", ".txt", ".rst"})
    if not documents:
        raise ValueError(f"No documentation files found under {root}")
    failures = []
    count = 0
    for document in documents:
        if document.suffix == ".rst":
            failures.append(f"{document}: reStructuredText extraction is not supported")
            continue
        try:
            blocks = python_blocks(document.read_bytes().decode("utf-8"), config)
        except ValueError as error:
            failures.append(f"{document}: {error}")
            continue
        for number, block in enumerate(blocks, 1):
            count += 1
            failure = syntax_failure(block, document, number)
            if failure:
                failures.append(failure)
    if not count:
        failures.append(f"No authored Python blocks found under {root}")
    return count, failures


class DocumentationPythonBlocksTests(unittest.TestCase):
    def setUp(self):
        self.config = MdParserConfig(enable_extensions={"colon_fence"})

    def test_comments_and_non_python_examples_are_not_code(self):
        source = ("<!--\n```python\ninvalid !\n```\n-->\n"
                  "````text\n```python\ninvalid !\n```\n````\n"
                  "```py\nmessage = 'caf\u00e9'\n```\n")
        blocks = python_blocks(source, self.config)
        self.assertEqual([block.code for block in blocks], ["message = 'caf\u00e9'\n"])
        self.assertIsNone(syntax_failure(blocks[0], "fixture.md", 1))

    def test_directive_options_and_structural_indentation(self):
        source = (" ```python\n value = 1\n ```\n"
                  ":::{code-block} python\n:linenos:\n:dedent: 1\n\n value = 2\n:::\n")
        blocks = python_blocks(source, self.config)
        self.assertEqual([block.code for block in blocks], ["value = 1\n", "value = 2"])
        self.assertEqual([block.first_line for block in blocks], [2, 8])
        for number, block in enumerate(blocks, 1):
            self.assertIsNone(syntax_failure(block, "fixture.md", number))
        extra_space = python_blocks("```{code-block} python\n\n value = 1\n```\n", self.config)[0]
        self.assertTrue(extra_space.code.startswith(" "))
        self.assertIn("unexpected indent", syntax_failure(extra_space, "fixture.md", 1))

    def test_syntax_diagnostics_preserve_unicode_and_source_lines(self):
        blocks = python_blocks("# title\n\n```python\n\u00a0value = 1\n```\n", self.config)
        self.assertTrue(blocks[0].code.startswith("\u00a0"))
        self.assertIn("fixture.md:4:1: block 1", syntax_failure(blocks[0], "fixture.md", 1))
        valid = python_blocks("```python\nvalue = '\u00a0'\n```\n", self.config)[0]
        self.assertIsNone(syntax_failure(valid, "fixture.md", 1))
        invalid = python_blocks("```python\nif True\n    pass\n```\n", self.config)[0]
        self.assertIn("fixture.md:2:", syntax_failure(invalid, "fixture.md", 1))

    def test_parser_valid_eof_and_malformed_directives(self):
        blocks = python_blocks("```python\nvalue = 1\n", self.config)
        self.assertEqual(len(blocks), 1)
        self.assertIsNone(syntax_failure(blocks[0], "fixture.md", 1))
        with self.assertRaisesRegex(ValueError, "line 1"):
            python_blocks("```{code-block} python\n:unknown-option: yes\n\nvalue = 1\n```\n", self.config)
        with self.assertRaisesRegex(ValueError, "unsupported"):
            python_blocks("````{note}\n```python\nvalue = 1\n```\n````\n", self.config)

    def test_empty_scan_is_not_success(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "No documentation files"):
                check_corpus(root, self.config)
            (root / "page.md").write_text("# No snippets\n", encoding="utf-8")
            count, failures = check_corpus(root, self.config)
            self.assertEqual(count, 0)
            self.assertIn("No authored Python blocks", "\n".join(failures))

    def test_all_authored_documentation_python_blocks_compile(self):
        repository = Path(__file__).resolve().parents[1]
        config = documentation_config(repository / "docs/conf.py")
        count, failures = check_corpus(repository / "docs/source", config)
        self.assertGreater(count, 0)
        self.assertEqual(failures, [], "\n".join(failures))


if __name__ == "__main__":
    unittest.main()
