"""Representation extraction skips unused sampling and reconstruction."""
import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset
from kaiwu.torch_plugin import BoltzmannMachine

EXAMPLE = Path(__file__).resolve().parents[1] / 'example/qvae_cell'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


models = load('cell_models', EXAMPLE / 'models.py')


@pytest.mark.parametrize('loss_type', ['mse', 'bernoulli'])
@pytest.mark.parametrize('representation', ['q', 'zeta'])
def test_extraction(loss_type, representation):
    # Execute the real Trainer without unrelated legacy SDK sampler imports (#202).
    tree = ast.parse((EXAMPLE / 'trainer.py').read_text())
    trainer_class = next(node for node in tree.body
                         if isinstance(node, ast.ClassDef) and node.name == 'Trainer')
    namespace = {'torch': torch, 'np': np}
    exec(compile(ast.Module(body=[trainer_class], type_ignores=[]),
                 str(EXAMPLE / 'trainer.py'), 'exec'), namespace)
    config = SimpleNamespace(num_latent_units=2, loss_type=loss_type,
                             dist_beta=2., kl_beta=1., weight_decay=0.)
    model = models.CellQVAE(3, None, config,
        encoder=torch.nn.Linear(3, 2), decoder=torch.nn.Linear(4, 3),
        bm=BoltzmannMachine(2, device='cpu'), sampler=object(), n_batches=2)
    model.set_dataset_mean(torch.tensor([.1, .2, .3]))
    x = torch.arange(12, dtype=torch.float32).reshape(4, 3) / 12
    loader = DataLoader(TensorDataset(x, torch.tensor([0, 1, 0, 1])), batch_size=2)
    model.eval()
    with torch.no_grad():
        expected_q = model.encoder(x - model._dataset_mean if loss_type == 'bernoulli' else x)
        torch.manual_seed(47)
        expected = []
        for batch, idx in loader:
            _, _, q, zeta = model(batch, idx)
            expected.append(q if representation == 'q' else zeta)
        expected = torch.cat(expected)
    model.decoder.forward = Mock(side_effect=AssertionError('unused decoder'))
    model.posterior = Mock(wraps=model.posterior)
    trainer = namespace['Trainer'](SimpleNamespace(representation=representation), 'cpu')
    torch.manual_seed(47)
    actual = torch.from_numpy(trainer.extract_representation(model, loader))
    torch.testing.assert_close(actual, expected)
    if representation == 'q':
        torch.testing.assert_close(actual, expected_q)
        model.posterior.assert_not_called()
    else:
        assert model.posterior.call_count == len(loader)
    model.decoder.forward.assert_not_called()
