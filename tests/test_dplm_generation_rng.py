"""Diagnostic sampling must not change the generation random stream."""

import ast
import importlib.util
from pathlib import Path

import torch


def load_helpers():
    root = Path(__file__).resolve().parents[1] / "example/qdiffusion/dplm"

    def load_file(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    io = load_file("rng_test_io", root / "utils/io.py")
    runtime = load_file("rng_test_runtime", root / "utils/runtime.py")
    path = root / "workflows/workflow_helpers.py"
    parsed = ast.parse(path.read_text())
    function = next(node for node in parsed.body if isinstance(node, ast.FunctionDef)
                    and node.name == "run_generation_over_records")
    # Execute the real function without importing optional pretrained-model dependencies.
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[
        ast.alias(name="annotations")], level=0), function], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = dict(torch=torch, Path=Path, seed_torch=runtime.seed_torch,
                     encode_sequence=runtime.encode_sequence,
                     normalize_decoded_sequence=io.normalize_decoded_sequence,
                     write_fasta_records=io.write_fasta_records,
                     write_tsv_rows=io.write_tsv_rows)
    exec(compile(module, str(path), "exec"), namespace)
    return namespace["run_generation_over_records"]


class Tokenizer:
    def __call__(self, sequence, **kwargs):
        return {"input_ids": torch.tensor([[1, 3, 3, 2]])}

    def batch_decode(self, tokens, **kwargs):
        return ["".join(chr(65 + int(value)) for value in row) for row in tokens]


class Generator:
    device = "cpu"
    tokenizer = Tokenizer()

    def __init__(self, diagnostic_draws):
        self.diagnostic_draws = diagnostic_draws
        self.generated = []

    def objective(self, batch):
        torch.rand(self.diagnostic_draws)
        return {"energy_objective": torch.tensor([[2.5]])}

    def generate(self, tokens, **kwargs):
        result = torch.randint(0, 20, (1, 8))
        self.generated.append(result.clone())
        return result


def test_generation_is_independent_of_diagnostic_rng_and_keeps_reports(tmp_path):
    run = load_helpers()
    results = []
    for draws in (1, 19):
        generator = Generator(draws)
        generated, rows = run(
            generator, [("a", "AC"), ("b", "DE")], max_steps=2, seed_base=42,
            output_dir=tmp_path / str(draws), label="guided",
        )
        results.append(generator.generated)
        assert [header for header, _ in generated] == ["a", "b"]
        assert [row["energy_objective_mean"] for row in rows] == [2.5, 2.5]
        assert (tmp_path / str(draws) / "guided_generated_sequences.fasta").exists()
        assert (tmp_path / str(draws) / "guided_generation_summary.tsv").exists()
    for first, second in zip(*results):
        torch.testing.assert_close(first, second)
    for index, generated in enumerate(results[0], start=1):
        expected = torch.randint(0, 20, (1, 8), generator=torch.Generator().manual_seed(42 + index))
        torch.testing.assert_close(generated, expected)
