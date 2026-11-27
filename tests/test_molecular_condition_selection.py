"""Validate strict experiment selections before data, model, or solver work."""

from pathlib import Path
import runpy
import sys

import pytest

pytest.importorskip("pandas")


@pytest.fixture(params=["sqa", "cim_sa"])
def parse_args(request, monkeypatch, tmp_path):
    root = Path(__file__).resolve().parents[1] / "example" / "molecular_design"
    script = root / request.param / "run_generation_strict.py"
    parse = runpy.run_path(str(script))["parse_args"]

    def parse_cli(*args):
        monkeypatch.setattr(sys, "argv", [str(script), "--save-dir", str(tmp_path), *args])
        return parse()

    return parse_cli


@pytest.mark.parametrize(("option", "value", "error"), [
    ("--reference-ids", "R00,R00", "duplicate"),
    ("--reference-ids", " , ", "at least one"),
    ("--bands", "1,01", "duplicate"),
    ("--bands", " , ", "at least one"),
    ("--bands", "x", "invalid"),
])
def test_rejects_ambiguous_or_empty_selections(parse_args, capsys, option, value, error):
    with pytest.raises(SystemExit) as exc:
        parse_args(option, value)
    assert exc.value.code == 2
    message = capsys.readouterr().err
    assert option in message
    assert error in message


def test_preserves_requested_order_and_the_all_default(parse_args):
    defaults = parse_args()
    assert defaults.reference_ids is None and defaults.bands is None
    selected = parse_args("--bands", "3, 1", "--reference-ids", "R02, R00")
    assert selected.bands == [3, 1]
    assert selected.reference_ids == ["R02", "R00"]
