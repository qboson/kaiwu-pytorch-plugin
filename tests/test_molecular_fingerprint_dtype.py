"""NumPy encoding should follow the molecular fingerprint's parameter dtype."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch


@pytest.mark.parametrize("variant", ["sqa", "cim_sa"])
@pytest.mark.parametrize(("model_dtype", "output_dtype"), [
    (torch.float16, np.float16),
    (torch.bfloat16, np.float32),
    (torch.float64, np.float64),
])
def test_fingerprint_numpy_encoding_respects_model_precision(variant, model_dtype, output_dtype, monkeypatch):
    root = Path.cwd() / "example" / "molecular_design" / variant
    monkeypatch.syspath_prepend(str(root))
    spec = importlib.util.spec_from_file_location(
        f"{variant}_graph_conv", root / "qc_mol_design" / "models" / "graph_conv.py"
    )
    graph_conv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(graph_conv)

    model = graph_conv.NeuralFingerprint().to(device="cpu", dtype=model_dtype)
    features = np.eye(2, graph_conv.CFG.atom_feat_dim, dtype=np.float32)
    adjacency = np.array([[0., 1.], [1., 0.]], dtype=np.float32)
    degrees = np.ones(2, dtype=np.int64)

    actual = model.encode_numpy(features, adjacency, degrees)

    with torch.no_grad():
        expected = model(
            torch.tensor(features[None], dtype=model_dtype),
            torch.tensor(adjacency[None], dtype=model_dtype),
            torch.tensor(degrees[None]),
        )[0]
    if model_dtype == torch.bfloat16:
        expected = expected.float()
    assert actual.dtype == output_dtype
    assert np.isfinite(actual).all()
    np.testing.assert_array_equal(actual, expected.numpy())
