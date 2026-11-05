"""Property-row alignment in the complete optional t-SNE plotting helper.

Scientific dependencies are inert sinks here: no embedding, chemistry, model,
or image rendering is performed. The actual source module and helper run.
"""

import importlib.util
import json
import sys
import unittest
import warnings
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch


class _Array:
    """Small JSON-array facade for indexing and draw arguments only."""

    def __init__(self, values):
        self.values = values

    def __len__(self):
        return len(self.values)

    @property
    def shape(self):
        if self.values and isinstance(self.values[0], list):
            return len(self.values), len(self.values[0])
        return (len(self.values),)

    def __getitem__(self, key):
        if isinstance(key, tuple):
            rows, column = key
            return _Array([row[column] for row in self.values[rows]])
        if isinstance(key, _Array):
            key = key.values
        if isinstance(key, list):
            return _Array([self.values[index] for index in key])
        result = self.values[key]
        return _Array(result) if isinstance(key, slice) else result


def _module(name, **attributes):
    result = ModuleType(name)
    result.__dict__.update(attributes)
    return result


class TsnePropertyAlignment(unittest.TestCase):
    """Both exported copies must keep valid rows and properties paired."""

    def _check_case(self, tree, case_name):
        root = Path(__file__).resolve().parents[1]
        fixture_path = root / "tests/fixtures/molecular_tsne_alignment.json"
        fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
        case = fixture["cases"][case_name]
        train_rows = fixture["train_rows"]
        generated_rows = fixture["generated_rows"][:case["generated_count"]]
        properties = fixture["properties"][:case["generated_count"]]
        original = json.dumps([train_rows, generated_rows, properties])
        selections = [case["train_indices"], case["generated_indices"]]
        choice_calls, draws, fits, saved = [], [], [], []

        def choice(population, size, replace):
            indices = selections[len(choice_calls)]
            self.assertFalse(replace)
            self.assertEqual(len(indices), size)
            self.assertEqual(len(set(indices)), size)
            self.assertTrue(all(0 <= index < population for index in indices))
            choice_calls.append((population, size, replace))
            return _Array(indices)

        numpy = _module(
            "numpy", ndarray=_Array, array=_Array,
            random=SimpleNamespace(choice=choice),
            vstack=lambda arrays: _Array([row for array in arrays for row in array.values]),
        )

        class TsneSink:
            """Check admission and preserve row markers without fitting a model."""

            def __init__(self, **kwargs):
                self.kwargs = kwargs

            def fit_transform(self, values):
                if not self.kwargs["perplexity"] < values.shape[0]:
                    raise ValueError("perplexity must be less than n_samples")
                if values.shape[1] != 2:
                    raise ValueError("The fixture uses two-dimensional row markers")
                fits.append((self.kwargs, values.values))
                return values

        def scatter(x, y, **kwargs):
            color = kwargs["c"]
            draws.append({"x": x.values, "y": y.values,
                          "colors": color.values if isinstance(color, _Array) else color})
            return object()

        axes = SimpleNamespace(
            scatter=scatter, legend=lambda **kwargs: None,
            set_title=lambda *args: None, set_xlabel=lambda *args: None,
            set_ylabel=lambda *args: None,
        )
        pyplot = _module(
            "matplotlib.pyplot", subplots=lambda **kwargs: (object(), axes),
            colorbar=lambda *args, **kwargs: None, tight_layout=lambda: None,
            savefig=lambda *args, **kwargs: saved.append((args, kwargs)),
            close=lambda: None,
        )
        patches = _module("matplotlib.patches")
        matplotlib = _module("matplotlib", use=lambda backend: None,
                             pyplot=pyplot, patches=patches)
        data = _module("torch.utils.data", DataLoader=object)
        utils = _module("torch.utils", data=data)
        manifold = _module("sklearn.manifold", TSNE=TsneSink)
        doubles = {
            "numpy": numpy, "pandas": _module("pandas"),
            "torch": _module("torch", utils=utils), "torch.utils": utils,
            "torch.utils.data": data, "matplotlib": matplotlib,
            "matplotlib.pyplot": pyplot, "matplotlib.patches": patches,
            "sklearn": _module("sklearn", manifold=manifold),
            "sklearn.manifold": manifold,
        }
        path = root / f"example/molecular_design/{tree}/qc_mol_design/evaluation/visualization.py"
        spec = importlib.util.spec_from_file_location(f"alignment_{tree}", path)
        module = importlib.util.module_from_spec(spec)
        filters_before = list(warnings.filters)
        namespace_before = {name: (name in sys.modules, sys.modules.get(name))
                            for name in doubles}
        with warnings.catch_warnings(), patch.dict(sys.modules, doubles):
            spec.loader.exec_module(module)
            module.plot_tsne(_Array(train_rows), _Array(generated_rows), _Array(properties),
                             max_points=case["max_points"], save_path="controlled-draw.png")
        self.assertEqual(warnings.filters, filters_before)
        for name, (existed, value) in namespace_before.items():
            self.assertEqual(name in sys.modules, existed)
            if existed:
                self.assertIs(sys.modules[name], value)
        self.assertEqual(len(choice_calls), 2)
        self.assertEqual(len(fits), 1)
        self.assertEqual(len(draws), 2)
        self.assertEqual(len(saved), 1)
        selected = case["generated_indices"]
        self.assertEqual(draws[1]["x"], [generated_rows[index][0] for index in selected])
        self.assertEqual(draws[1]["y"], [generated_rows[index][1] for index in selected])
        self.assertEqual(draws[1]["colors"], [properties[index] for index in selected])
        self.assertEqual(json.dumps([train_rows, generated_rows, properties]), original)

    def test_cim_subsample_keeps_property_rows(self):
        self._check_case("cim_sa", "subsample")

    def test_sqa_subsample_keeps_property_rows(self):
        self._check_case("sqa", "subsample")

    def test_cim_full_permutation_keeps_property_rows(self):
        self._check_case("cim_sa", "full_permutation")

    def test_sqa_full_permutation_keeps_property_rows(self):
        self._check_case("sqa", "full_permutation")

    def test_cim_identity_selection_control(self):
        self._check_case("cim_sa", "identity")

    def test_sqa_identity_selection_control(self):
        self._check_case("sqa", "identity")


if __name__ == "__main__":
    unittest.main()
