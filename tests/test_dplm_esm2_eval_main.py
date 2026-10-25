"""Contract test for the ``esm2_eval.main`` evaluation pipeline."""

from __future__ import annotations

import sys
import types
from pathlib import Path
from types import SimpleNamespace

import torch

REPO = Path(__file__).resolve().parents[1]
QDIFFUSION = REPO / "example" / "qdiffusion"


def _load_module():
    esm_stub = types.ModuleType("esm")
    previous = sys.modules.get("esm")
    sys.modules["esm"] = esm_stub
    sys.path.insert(0, str(QDIFFUSION))
    try:
        from dplm.workflows import esm2_eval
    finally:
        sys.path.remove(str(QDIFFUSION))
        if previous is None:
            sys.modules.pop("esm", None)
        else:
            sys.modules["esm"] = previous
    return esm2_eval


def _summary(label, count=1, cosine=0.1, l2=0.2):
    return SimpleNamespace(
        label=label,
        paired_count=count,
        mean_cosine_distance=cosine,
        mean_l2_distance=l2,
    )


def test_runs_both_candidate_branches_and_writes_every_artifact(tmp_path, monkeypatch, capsys):
    module = _load_module()
    output_dir = tmp_path / "eval"
    config = SimpleNamespace(
        device="cpu",
        output_dir=output_dir,
        reference_fasta=Path("/tmp/reference.fa"),
        max_records=2,
        esm2_model="esm2_t33_650M_UR50D",
        batch_size=1,
        pooling="mean",
        pair_mode="order",
    )
    state = {
        "reads": [],
        "limits": [],
        "generated": [],
        "embedded": [],
        "evaluated": [],
        "writes": [],
        "reports": [],
    }
    module.build_sa_eval_config = lambda **kwargs: config
    module.default_fasta_path = lambda: Path("/tmp/reference.fa")
    module.default_outputs_root = lambda: tmp_path
    module.read_fasta_records = lambda path: (
        state["reads"].append(path)
        or [("h1", "AA"), ("h2", "BB"), ("h3", "CC")]
    )
    module.maybe_limit_records = lambda records, limit: (
        state["limits"].append(limit) or records[:limit]
    )
    baseline_path = tmp_path / "baseline.fa"
    guided_path = tmp_path / "guided.fa"

    def fake_generate(**kwargs):
        state["generated"].append(kwargs)
        return baseline_path, guided_path

    module.generate_candidate_fastas = fake_generate
    def fake_load(name, device):
        state["model"] = (name, device)
        return "model", "alphabet"

    module.load_esm2_model = fake_load
    module.embed_sequences = lambda records, **kwargs: (
        state["embedded"].append((records, kwargs)) or {"records": records}
    )

    def fake_evaluate(**kwargs):
        state["evaluated"].append(kwargs)
        return [("pair",)], _summary(kwargs["label"])

    module.evaluate_candidate_set = fake_evaluate
    module.write_rows_csv = lambda path, rows: state["writes"].append(("rows", path))
    module.write_summary_json = lambda path, summary: state["writes"].append(
        ("summary", path)
    )
    module.write_report = lambda path, **kwargs: state["reports"].append((path, kwargs))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    module.main()

    assert state["reads"] == [
        Path("/tmp/reference.fa"),
        baseline_path,
        guided_path,
    ]
    assert state["limits"] == [2]
    assert len(state["generated"]) == 1
    generated = state["generated"][0]
    assert generated["config"] is config
    assert generated["reference_records"] == [("h1", "AA"), ("h2", "BB")]
    assert generated["device"] == torch.device("cpu")
    assert generated["output_dir"] == output_dir
    assert state["model"] == ("esm2_t33_650M_UR50D", torch.device("cpu"))
    assert len(state["embedded"]) == 3
    for records, kwargs in state["embedded"]:
        assert kwargs["model"] == "model"
        assert kwargs["alphabet"] == "alphabet"
        assert kwargs["device"] == torch.device("cpu")
        assert kwargs["batch_size"] == 1
        assert kwargs["pooling"] == "mean"
    assert [call["label"] for call in state["evaluated"]] == ["baseline", "guided"]
    assert all(call["pair_mode"] == "order" for call in state["evaluated"])
    assert state["writes"] == [
        ("rows", output_dir / "baseline_pair_distances.csv"),
        ("summary", output_dir / "baseline_summary.json"),
        ("rows", output_dir / "guided_pair_distances.csv"),
        ("summary", output_dir / "guided_summary.json"),
    ]
    report_path, report_kwargs = state["reports"][0]
    assert report_path == output_dir / "REPORT.md"
    assert [summary.label for summary in report_kwargs["summaries"]] == [
        "baseline",
        "guided",
    ]
    assert report_kwargs["model_name"] == "esm2_t33_650M_UR50D"
    assert report_kwargs["pair_mode"] == "order"
    assert report_kwargs["pooling"] == "mean"
    output = capsys.readouterr().out
    assert "[baseline] pairs=1" in output
    assert "[guided] pairs=1" in output
