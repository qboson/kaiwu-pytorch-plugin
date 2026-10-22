"""Validation reconstructions retain the likelihood's output semantics."""
import ast
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset
from kaiwu.torch_plugin import BoltzmannMachine, QVAE


@pytest.mark.parametrize('loss_type', ['mse', 'bernoulli'])
def test_validation_returns_likelihood_appropriate_values(loss_type):
    path = Path(__file__).resolve().parents[1] / 'example/qvae_mnist/trainer/model_tuner.py'
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'ModelTuner')
    namespace = {'torch': torch, 'logger': logging.getLogger(__name__)}
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(path), 'exec'), namespace)
    config = SimpleNamespace(type='QVAE', num_latent_units=2, loss_type=loss_type,
                             dist_beta=2., kl_beta=1., weight_decay=0.)
    decoder = torch.nn.Linear(2, 3)
    with torch.no_grad():
        decoder.weight.zero_()
        decoder.bias.copy_(torch.tensor([-2., 0., 2.]))
    model = QVAE(3, None, config, encoder=torch.nn.Linear(3, 2), decoder=decoder,
                 bm=BoltzmannMachine(2, device='cpu'), sampler=object())
    # Replace only external negative-phase sampling with local binary states.
    model.bm.sample = lambda sampler: torch.tensor([[0., 1.], [1., 0.]])
    x = torch.rand(4, 3)
    loader = DataLoader(TensorDataset(x, torch.zeros(4, dtype=torch.long)), batch_size=2)
    tuner = namespace['ModelTuner'](config)
    tuner.register_model(model)
    tuner.register_dataLoaders(loader, loader)
    loss, inputs, reconstruction, _ = tuner.test()
    raw = decoder.bias.detach().expand(2, -1)
    expected = torch.sigmoid(raw) if loss_type == 'bernoulli' else raw
    torch.testing.assert_close(reconstruction, expected)
    torch.testing.assert_close(inputs, x[-2:])
    assert torch.isfinite(loss)
    assert not reconstruction.requires_grad
