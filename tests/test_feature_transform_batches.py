"""Array-based feature extraction bounds each model forward call."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from kaiwu.torch_plugin import QVAE, BoltzmannMachine

spec = importlib.util.spec_from_file_location('mnist_extractor',
    Path(__file__).resolve().parents[1] / 'example/qvae_mnist/model/feature_extractor.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def model():
    config = SimpleNamespace(num_latent_units=2, loss_type='mse',
                             dist_beta=2., kl_beta=1., weight_decay=0.)
    return QVAE(3, None, config,
        encoder=torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Dropout(.2), torch.nn.Linear(4, 2)),
        decoder=torch.nn.Linear(2, 3), bm=BoltzmannMachine(2, device='cpu'), sampler=object())


@pytest.mark.parametrize('numpy_input', [False, True])
@pytest.mark.parametrize('feature_type', ['q', 'zeta'])
def test_transform_keeps_order_and_bounds_forward(numpy_input, feature_type):
    torch.manual_seed(17)
    net = model()
    x = torch.arange(33, dtype=torch.float32).reshape(11, 3) / 33
    original = x.clone()
    seen = []
    handle = net.register_forward_pre_hook(lambda _, args: seen.append(len(args[0])))
    try:
        extractor = module.FeatureExtractor(net, feature_type, batch_size=4)
        result = extractor.transform(x.numpy() if numpy_input else x)
    finally:
        handle.remove()
    assert seen == [4, 4, 3]
    assert result.shape == (11, 2)
    assert result.dtype == np.float32
    assert np.isfinite(result).all()
    torch.testing.assert_close(x, original)
    assert not net.training
    if feature_type == 'q':
        with torch.no_grad():
            expected = net.encoder(x).numpy()
        np.testing.assert_allclose(result, expected, rtol=1e-6, atol=1e-7)
    else:
        assert ((result >= 0) & (result <= 1)).all()


@pytest.mark.parametrize('size,error', [(0, ValueError), (-1, ValueError), (True, TypeError),
                                       (1.5, TypeError)])
def test_invalid_batch_sizes(size, error):
    with pytest.raises(error, match='batch_size'):
        module.FeatureExtractor(model(), batch_size=size)


def test_empty_input_has_clear_error():
    with pytest.raises(ValueError, match='at least one sample'):
        module.FeatureExtractor(model()).transform(np.empty((0, 3), dtype=np.float32))


def test_default_batch_size_bounds_larger_inputs():
    net = model()
    seen = []
    handle = net.register_forward_pre_hook(lambda _, args: seen.append(len(args[0])))
    try:
        result = module.FeatureExtractor(net).transform(np.zeros((513, 3), dtype=np.float32))
    finally:
        handle.remove()
    assert seen == [256, 256, 1]
    assert result.shape == (513, 2)
