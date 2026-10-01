"""Actual DPLM evaluation keeps record identity with a small CPU sequence model."""
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import runpy
import sys
import types

import pytest
import torch
from torch import nn


class TinySequenceModel(nn.Module):
    """Real contextual embeddings with residue, BOS and EOS pooling oracles."""
    num_layers = 1

    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(torch.tensor(
            [[0., 0.], [0., 0.], [0., 0.], [1., 0.], [0., 1.], [1., 1.]],
            dtype=torch.float64,
        ))

    def forward(self, tokens, **kwargs):
        representations = self.embedding(tokens)
        residues = tokens.ge(3)
        average = (representations * residues.unsqueeze(-1)).sum(1) / residues.sum(1, keepdim=True)
        representations[:, 0] = average + torch.tensor([1., 2.])
        for row in range(tokens.size(0)):
            eos = int(residues[row].sum()) + 1
            representations[row, eos] = 2 * average[row] + torch.tensor([2., 1.])
        return {"representations": {1: representations}}


class TinyAlphabet:
    """Use the documented ESM batch-converter interface without downloads."""
    padding_idx = 0

    def get_batch_converter(self):
        def convert(records):
            tokens = torch.zeros(len(records), max(len(s) for _, s in records) + 2, dtype=torch.long)
            for row, (_, sequence) in enumerate(records):
                tokens[row, 0] = 1
                tokens[row, 1:len(sequence) + 1] = torch.tensor(
                    [{"A": 3, "C": 4, "G": 5}[x] for x in sequence]
                )
                tokens[row, len(sequence) + 1] = 2
            return [h for h, _ in records], [s for _, s in records], tokens
        return convert


def expected_embedding(sequence, pooling):
    """Compute pooled values from residue counts, independently of the model/helper."""
    average = torch.tensor([
        (sequence.count("A") + sequence.count("G")) / len(sequence),
        (sequence.count("C") + sequence.count("G")) / len(sequence),
    ], dtype=torch.float64)
    if pooling in {"bos", "cls"}:
        return average + torch.tensor([1., 2.])
    if pooling == "eos":
        return 2 * average + torch.tensor([2., 1.])
    return average


@pytest.fixture
def evaluation_modules(monkeypatch):
    """Load full actual modules; only bypass unrelated pretrained-builder imports."""
    root = Path(__file__).resolve().parents[1]
    dplm = root / "example/qdiffusion/dplm"
    prefix = "dplm_evaluation_fixture"
    for suffix, directory in (("", dplm), (".utils", dplm / "utils"), (".workflows", dplm / "workflows")):
        package = types.ModuleType(prefix + suffix)
        package.__path__ = [str(directory)]
        monkeypatch.setitem(sys.modules, package.__name__, package)
    # The helper's eager optional import is not used by embedding supplied models.
    monkeypatch.setitem(sys.modules, "esm", types.ModuleType("esm"))
    builder = types.ModuleType(prefix + ".utils.dplm_builder")
    builder.build_qdiffusion = lambda **kwargs: pytest.fail("No pretrained generator may be built")
    monkeypatch.setitem(sys.modules, builder.__name__, builder)
    loaded = []
    for filename in ("esm2_eval_helpers", "esm2_eval"):
        name = prefix + ".workflows." + filename
        path = dplm / "workflows" / (filename + ".py")
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
        assert Path(module.__file__).resolve() == path.resolve()
        loaded.append(module)
    return tuple(loaded)


def embed(helper, records, pooling="mean", batch_size=2, **kwargs):
    return helper.embed_sequences(
        records, model=TinySequenceModel().eval(), alphabet=TinyAlphabet(),
        device=torch.device("cpu"), batch_size=batch_size, pooling=pooling, **kwargs,
    )


@pytest.mark.parametrize("pooling", ["mean", "bos", "eos"])
@pytest.mark.parametrize("batch_size", [1, 2])
def test_position_keys_preserve_repeated_headers_across_batches(evaluation_modules, pooling, batch_size):
    helper, _ = evaluation_modules
    records = [("same", "AAA"), ("same", "CCC"), ("same", "AAA"), ("other", "ACG")]
    embeddings = embed(helper, records, pooling, batch_size, key_mode="position")
    assert list(embeddings) == list(range(len(records)))
    for index, (_, sequence) in enumerate(records):
        torch.testing.assert_close(embeddings[index], expected_embedding(sequence, pooling))


@pytest.mark.parametrize("case", ["identical", "reordered", "different"])
@pytest.mark.parametrize("pooling", ["mean", "bos", "eos"])
def test_order_distances_use_each_position_instead_of_header(evaluation_modules, case, pooling):
    helper, _ = evaluation_modules
    reference = [("same", "AAA"), ("same", "CCC"), ("same", "AAA")]
    candidates = reference if case == "identical" else [reference[1], reference[0], reference[2]]
    if case == "different":
        candidates = [("same", "GG"), ("same", "AC"), ("same", "CCC")]
    rows, summary = helper.evaluate_candidate_set(
        label=case, reference_records=reference, candidate_records=candidates,
        reference_embeddings=embed(helper, reference, pooling, key_mode="position"),
        candidate_embeddings=embed(helper, candidates, pooling, key_mode="position"),
        pair_mode="order", key_mode="position",
    )
    expected = []
    for row, (_, ref), (_, candidate) in zip(rows, reference, candidates):
        first, second = expected_embedding(ref, pooling), expected_embedding(candidate, pooling)
        cosine = 1 - float(first @ second / (first.norm() * second.norm()))
        distance = float((first - second).square().sum().sqrt())
        assert row.cosine_distance == pytest.approx(cosine, abs=1e-14)
        assert row.l2_distance == pytest.approx(distance, abs=1e-14)
        expected.append(cosine)
    assert summary.mean_cosine_distance == pytest.approx(sum(expected) / len(expected), abs=1e-14)


def test_default_mapping_and_header_pairing_preserve_unique_header_consumers(evaluation_modules):
    helper, _ = evaluation_modules
    reference = [("first", "AAA"), ("second", "CCC")]
    candidates = list(reversed(reference))
    embeddings = embed(helper, reference)
    assert set(embeddings) == {"first", "second"}
    torch.testing.assert_close(embeddings["first"], expected_embedding("AAA", "mean"))
    rows, _ = helper.evaluate_candidate_set(
        label="headers", reference_records=reference, candidate_records=candidates,
        reference_embeddings=embeddings, candidate_embeddings=embed(helper, candidates),
        pair_mode="header",
    )
    assert [row.cosine_distance for row in rows] == [0., 0.]


def test_default_header_mapping_still_supports_order_pairing(evaluation_modules):
    helper, _ = evaluation_modules
    reference = [("reference-first", "AAA"), ("reference-second", "CCC")]
    candidates = [("candidate-first", "AAA"), ("candidate-second", "CCC")]
    rows, _ = helper.evaluate_candidate_set(
        label="legacy-order", reference_records=reference, candidate_records=candidates,
        reference_embeddings=embed(helper, reference), candidate_embeddings=embed(helper, candidates),
        pair_mode="order",
    )
    assert [row.cosine_distance for row in rows] == [0., 0.]


def test_position_mapping_cannot_be_used_for_header_pairing(evaluation_modules):
    helper, _ = evaluation_modules
    records = [("first", "AAA"), ("second", "CCC")]
    embeddings = embed(helper, records, key_mode="position")
    with pytest.raises(ValueError, match="require.*order"):
        helper.evaluate_candidate_set(
            label="incompatible", reference_records=records, candidate_records=records,
            reference_embeddings=embeddings, candidate_embeddings=embeddings,
            pair_mode="header", key_mode="position",
        )


@pytest.mark.parametrize("same_sequence", [True, False])
def test_header_mapping_rejects_ambiguous_duplicate_headers(evaluation_modules, same_sequence):
    helper, _ = evaluation_modules
    records = [("same", "AAA"), ("same", "AAA" if same_sequence else "CCC")]
    with pytest.raises(ValueError, match="[Dd]uplicate.*header"):
        embed(helper, records)


@pytest.mark.parametrize("pair_mode", ["order", "header"])
def test_legacy_header_mapping_cannot_silently_misuse_duplicate_headers(evaluation_modules, pair_mode):
    helper, _ = evaluation_modules
    records = [("same", "AAA"), ("same", "CCC")]
    with pytest.raises(ValueError, match="[Dd]uplicate.*header"):
        helper.evaluate_candidate_set(
            label="ambiguous", reference_records=records, candidate_records=records,
            reference_embeddings={"same": torch.tensor([0., 1.])},
            candidate_embeddings={"same": torch.tensor([0., 1.])}, pair_mode=pair_mode,
        )


@pytest.mark.parametrize("pair_mode", ["order", "header"])
@pytest.mark.parametrize("entrypoint", [False, True])
def test_actual_main_uses_consistent_embedding_identity_and_writes_correct_reports(
    evaluation_modules, monkeypatch, tmp_path, pair_mode, entrypoint
):
    _, workflow = evaluation_modules
    reference = [("same" if pair_mode == "order" else "first", "AAA"),
                 ("same" if pair_mode == "order" else "second", "CCC")]
    candidates = [("candidate-first", "AAA"), ("candidate-second", "CCC")] if pair_mode == "order" else list(reversed(reference))
    paths = [tmp_path / filename for filename in ("reference.fasta", "baseline.fasta", "guided.fasta")]
    for path, records in zip(paths, (reference, candidates, candidates)):
        path.write_text("".join(f">{h}\n{s}\n" for h, s in records), encoding="utf-8")
    factory = workflow.build_sa_eval_config

    def offline_config(**kwargs):
        return replace(factory(**kwargs), reference_fasta=paths[0], output_dir=tmp_path / "reports",
                       device="cpu", pair_mode=pair_mode, pooling="mean", batch_size=1)

    monkeypatch.setattr(workflow, "build_sa_eval_config", offline_config)
    monkeypatch.setattr(workflow, "generate_candidate_fastas", lambda **kwargs: tuple(paths[1:]))
    monkeypatch.setattr(workflow, "load_esm2_model", lambda *args: (TinySequenceModel().eval(), TinyAlphabet()))
    if entrypoint:
        package = types.ModuleType("workflows")
        package.__path__ = [str(Path(workflow.__file__).parent)]
        monkeypatch.setitem(sys.modules, "workflows", package)
        monkeypatch.setitem(sys.modules, "workflows.esm2_eval", workflow)
        path = Path(workflow.__file__).parents[1] / "eval_esm2_distances.py"
        runpy.run_path(str(path), run_name="__main__")
    else:
        workflow.main()
    for label in ("baseline", "guided"):
        payload = json.loads((tmp_path / "reports" / f"{label}_summary.json").read_text())
        assert payload["paired_count"] == 2
        assert payload["mean_cosine_distance"] == 0
        assert payload["mean_l2_distance"] == 0
    assert (tmp_path / "reports/REPORT.md").exists()
