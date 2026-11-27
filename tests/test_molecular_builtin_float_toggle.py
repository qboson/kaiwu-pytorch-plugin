"""Trace the exported fallback only to its first scalar mutation.

All dependencies are inert metadata facades. No optimizer completes, and no
numerical backend, energy search, chemistry, model or SDK runs.
"""

import importlib.util
import json
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

OBSERVATIONS = []


class FirstToggleWritten(BaseException):
    """Stop immediately after observing the first valid scalar write."""


class ForbiddenBoundary(BaseException):
    """Fail closed before an operation outside this scalar-only test."""


def _module(name, **values):
    module = ModuleType(name)
    module.__dict__.update(values)
    return module


class BuiltinFloatToggle(unittest.TestCase):
    """Both source copies must toggle valid float-valued binary states."""

    def _exercise(self, tree, initialization, explicit_sqa=False):
        root = Path(__file__).resolve().parents[1]
        fixture = json.loads(
            (root / "tests/fixtures/molecular_builtin_toggle.json").read_text(
                encoding="utf-8"
            )
        )
        matrix = fixture["matrix"]
        size = fixture["shape"][0]
        self.assertEqual(fixture["shape"], [3, 3])
        self.assertTrue(all(len(row) == size for row in matrix))
        self.assertTrue(all(matrix[i][j] == 0 for i in range(size) for j in range(i)))
        initial = fixture[initialization + "_binary"]
        self.assertEqual(len(initial), size)
        self.assertTrue(all(value in (0, 1) for value in initial))
        index = fixture["flip_index"]
        observation = {
            "tree": tree,
            "initialization": initialization,
            "explicit_sqa": explicit_sqa,
            "environment_keys": [],
            "random_index_requests": [],
            "initial_energy_calls": 0,
            "scalar_reads": [],
            "writes": [],
        }

        def forbidden(*args, **kwargs):
            raise ForbiddenBoundary("Outside the first scalar-mutation contract")

        def environment_get(key, default=None):
            observation["environment_keys"].append(key)
            if key == "QC_QUBO_SOLVER":
                return "sqa" if explicit_sqa else ""
            if key in ("KAIWU_CIM_PROJECT_NO", "KAIWU_CIM_CACHE_DIR"):
                return default
            raise ForbiddenBoundary("Unapproved environment-key access")

        q_metadata = SimpleNamespace(shape=tuple(fixture["shape"]))
        check = self

        class InitialEnergyMetadata:
            """Return a pre-audited JSON scalar without matrix evaluation."""

            def __init__(self, state):
                self.state = state

            def __matmul__(self, other):
                check.assertIs(other, self.state)
                return fixture["initial_energy_metadata"][initialization]

        class FloatState:
            """Indexing returns actual Python floats; xor is not implemented."""

            def __init__(self, values):
                self.values = [float(value) for value in values]

            def __getitem__(self, position):
                value = self.values[position]
                observation["scalar_reads"].append(
                    {"index": position, "value": value, "type": type(value).__name__}
                )
                return value

            def __setitem__(self, position, value):
                check.assertEqual(position, index)
                check.assertIs(type(value), float)
                check.assertIn(value, (0.0, 1.0))
                before = self.values[position]
                self.values[position] = value
                observation["writes"].append(
                    {"index": position, "before": before, "after": value}
                )
                raise FirstToggleWritten()

            def __matmul__(self, other):
                check.assertIs(other, q_metadata)
                observation["initial_energy_calls"] += 1
                check.assertEqual(observation["initial_energy_calls"], 1)
                return InitialEnergyMetadata(self)

        class InitialIntegerMetadata:
            """Describe a valid fixed random initializer without generating it."""

            def astype(self, dtype):
                check.assertIs(dtype, float)
                return FloatState(fixture["random_binary"])

        def array(values, dtype):
            self.assertIs(dtype, float)
            return FloatState(values)

        def randint_array(low, high, count):
            self.assertEqual((low, high, count), (0, 2, size))
            return InitialIntegerMetadata()

        def randint_index(low, high):
            self.assertEqual((low, high), (0, size - 1))
            observation["random_index_requests"].append([low, high])
            return index

        def absolute_metadata(value):
            self.assertIs(value, q_metadata)
            return SimpleNamespace(max=lambda: fixture["max_abs"])

        numpy = _module(
            "numpy",
            ndarray=FloatState,
            float64=float,
            array=array,
            abs=absolute_metadata,
            random=SimpleNamespace(randint=randint_array),
        )
        configuration = _module(
            "qc_mol_design.config",
            CFG=SimpleNamespace(**fixture["config"]),
            ATOM_VALENCY={"C": 4},
        )
        package = _module("qc_mol_design", config=configuration, __path__=[])
        doubles = {
            "numpy": numpy,
            "math": _module("math", exp=forbidden, log=forbidden, ceil=forbidden),
            "random": _module("random", randint=randint_index, random=forbidden),
            "os": _module("os", environ=SimpleNamespace(get=environment_get)),
            "qc_mol_design": package,
            "qc_mol_design.config": configuration,
            "dwave": None,
            "dwave.samplers": None,
            "neal": None,
        }
        namespace_before = {
            name: (name in sys.modules, sys.modules.get(name)) for name in doubles
        }
        path = root / f"example/molecular_design/{tree}/qc_mol_design/optimization/qubo.py"
        spec = importlib.util.spec_from_file_location(f"float_toggle_{tree}", path)
        module = importlib.util.module_from_spec(spec)
        expected = ModuleNotFoundError if explicit_sqa else FirstToggleWritten
        try:
            with patch.dict(sys.modules, doubles):
                spec.loader.exec_module(module)
                with self.assertRaises(expected):
                    try:
                        module.solve_qubo(
                            q_metadata,
                            fixture["num_reads"],
                            fixture["num_sweeps"],
                            initial_states=(
                                [{i: value for i, value in enumerate(initial)}]
                                if initialization == "warm" else None
                            ),
                        )
                    except BaseException as exc:
                        observation["exception_type"] = type(exc).__name__
                        observation["exception_message"] = str(exc)
                        raise
        finally:
            for name, (existed, value) in namespace_before.items():
                self.assertEqual(name in sys.modules, existed)
                if existed:
                    self.assertIs(sys.modules[name], value)
            OBSERVATIONS.append(observation)
        if explicit_sqa:
            self.assertEqual(observation["initial_energy_calls"], 0)
            self.assertEqual(observation["writes"], [])
            self.assertEqual(observation["random_index_requests"], [])
        else:
            self.assertEqual(observation["initial_energy_calls"], 1)
            self.assertEqual(observation["random_index_requests"], [[0, size - 1]])
            self.assertEqual(
                observation["writes"],
                [{"index": index, "before": float(initial[index]),
                  "after": 1.0 - float(initial[index])}],
            )
            self.assertTrue(all(row["type"] == "float" for row in observation["scalar_reads"]))

    def test_cim_warm_first_float_toggle(self):
        self._exercise("cim_sa", "warm")

    def test_sqa_warm_first_float_toggle(self):
        self._exercise("sqa", "warm")

    def test_cim_random_first_float_toggle(self):
        self._exercise("cim_sa", "random")

    def test_sqa_random_first_float_toggle(self):
        self._exercise("sqa", "random")

    def test_cim_explicit_sqa_never_falls_back(self):
        self._exercise("cim_sa", "warm", explicit_sqa=True)

    def test_sqa_explicit_sqa_never_falls_back(self):
        self._exercise("sqa", "warm", explicit_sqa=True)


if __name__ == "__main__":
    unittest.main()
