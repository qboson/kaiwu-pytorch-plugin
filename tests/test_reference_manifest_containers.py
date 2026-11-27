"""Exercise both reference-manifest containers through the real example packages."""

from importlib.util import find_spec
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest


_MISSING_DEPENDENCIES = [
    name for name in ("numpy", "pandas", "rdkit", "torch", "tqdm")
    if find_spec(name) is None
]


# Each backend exports the same package name, so import each in a fresh process.
_CHECK_MANIFESTS = r'''
from dataclasses import asdict
import json
from pathlib import Path
import tempfile

import pandas as pd
from qc_mol_design.data import references

records = [
    dict(reference_id="water", test_position=0, source_index=0,
         smiles="O", canonical_smiles="O", n_heavy_atoms=1, n_qubo_vars=0,
         formal_charge=0, size_stratum="tiny"),
    dict(reference_id="ethanol", test_position=1, source_index=1,
         smiles="CCO", canonical_smiles="CCO", n_heavy_atoms=3, n_qubo_vars=3,
         formal_charge=0, size_stratum="tiny"),
]
expected = [dict(record, legacy_source="") for record in records]
results = []

with tempfile.TemporaryDirectory() as directory:
    path = Path(directory) / "manifest.json"

    def check(name, payload, *, error=None, count=2, test_df=None, raw=False):
        path.write_text(payload if raw else json.dumps(payload), encoding="utf-8")
        try:
            actual = references.load_reference_manifest(
                path, expected_count=count, test_df=test_df
            )
        except Exception as exc:
            passed = error is not None and isinstance(exc, error)
            detail = type(exc).__name__ + ": " + str(exc)
        else:
            passed = error is None and [asdict(record) for record in actual] == expected
            detail = "returned " + repr(actual)
        results.append(dict(name=name, passed=passed, detail=detail))

    check("object", {"references": records})
    check("object_with_metadata", {"references": records, "description": "synthetic"})
    check("bare_list", records)
    check("test_split_membership", {"references": records},
          test_df=pd.DataFrame({"smiles": ["O", "CCO"]}))
    check("count", {"references": records}, count=3, error=ValueError)
    check("duplicate_id",
          {"references": [records[0], dict(records[1], reference_id="water")]},
          error=ValueError)
    check("duplicate_canonical",
          {"references": [records[0], dict(records[0], reference_id="other")]},
          error=ValueError)
    check("canonical_validation",
          {"references": [dict(records[0], canonical_smiles="CO"), records[1]]},
          error=ValueError)
    check("atom_count_validation", {"references": [dict(records[0], n_heavy_atoms=2), records[1]]},
          error=ValueError)
    check("qubo_count_validation", {"references": [records[0], dict(records[1], n_qubo_vars=0)]},
          error=ValueError)
    check("missing_from_test_split", {"references": records},
          test_df=pd.DataFrame({"smiles": ["O"]}), error=ValueError)
    # Rejection is required; no new exception-type contract is imposed on these shapes.
    check("scalar", 7, error=Exception)
    check("null", None, error=Exception)
    check("invalid_references_container", {"references": None}, error=Exception)
    check("non_object_entry", {"references": [7]}, error=Exception)
    check("missing_field_entry", {"references": [{"reference_id": "incomplete"}]},
          error=Exception)
    check("non_object_list_entry", [7], error=Exception)
    check("missing_references", {"unexpected": records}, error=Exception)
    check("invalid_json", "{", raw=True, error=json.JSONDecodeError)

print(json.dumps(dict(module=references.__file__, results=results)))
'''


@unittest.skipIf(
    _MISSING_DEPENDENCIES,
    "Optional molecular example dependencies missing: " + ", ".join(_MISSING_DEPENDENCIES),
)
class TestReferenceManifestContainers(unittest.TestCase):
    """Both independently shipped backends must keep the same loader behavior."""

    def _check_backend(self, backend):
        root = Path(__file__).resolve().parents[1]
        backend_root = root / "example" / "molecular_design" / backend
        environment = os.environ.copy()
        environment.pop("PYTHONPATH", None)
        environment.pop("PYTHONHOME", None)
        result = subprocess.run(
            [sys.executable, "-c", _CHECK_MANIFESTS],
            cwd=backend_root,
            env=environment,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual(
            Path(report["module"]).resolve(),
            (backend_root / "qc_mol_design" / "data" / "references.py").resolve(),
        )
        for case in report["results"]:
            with self.subTest(backend=backend, case=case["name"]):
                self.assertTrue(case["passed"], case["detail"])

    def test_cim_sa(self):
        """Validate CIM/SA manifests through its own normal package import."""
        self._check_backend("cim_sa")

    def test_sqa(self):
        """Validate SQA manifests without reusing the other backend's modules."""
        self._check_backend("sqa")


if __name__ == "__main__":
    unittest.main()
