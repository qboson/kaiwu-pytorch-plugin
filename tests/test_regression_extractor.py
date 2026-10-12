"""Feature inputs follow the current floating-point model parameters."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class SmallQVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(3, 2)

    def forward(self, x):
        q = self.encoder(x)
        return x, None, q, torch.sigmoid(q)


@pytest.mark.parametrize("model_dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("as_numpy", [True, False])
def test_transform_casts_numpy_and_tensor_inputs(model_dtype, as_numpy):
    path = Path(__file__).resolve().parents[1] / "example/qvae_mnist/model/feature_extractor.py"
    spec = importlib.util.spec_from_file_location("feature_dtype_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = SmallQVAE()
    extractor = module.FeatureExtractor(model)
    model.to(dtype=model_dtype)  # Conversion after extractor construction must work too.
    data = np.arange(12, dtype=np.float64).reshape(4, 3)
    inputs = data if as_numpy else torch.tensor(data, dtype=torch.float32)
    observed = extractor.transform(inputs)
    expected = model.encoder(torch.as_tensor(data, dtype=model_dtype)).detach().numpy()
    np.testing.assert_allclose(observed, expected, rtol=1e-6)
    assert observed.dtype == expected.dtype


def test_extract_casts_loader_batches_and_preserves_labels():
    path = Path(__file__).resolve().parents[1] / "example/qvae_mnist/model/feature_extractor.py"
    spec = importlib.util.spec_from_file_location("feature_loader_dtype_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = SmallQVAE().double()
    data = torch.arange(15, dtype=torch.float32).reshape(5, 3)
    labels = torch.arange(5)
    extractor = module.FeatureExtractor(model, feature_type="zeta")
    features, actual_labels = extractor.extract(DataLoader(TensorDataset(data, labels), batch_size=2))
    torch.testing.assert_close(features, torch.sigmoid(model.encoder(data.double())))
    torch.testing.assert_close(actual_labels, labels)
    assert not features.requires_grad
