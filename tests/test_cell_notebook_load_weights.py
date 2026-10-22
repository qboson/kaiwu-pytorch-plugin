"""Execute the actual notebook weight/history cells with local checkpoint bytes."""
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

NOTEBOOK = Path(__file__).resolve().parents[1] / 'example/qvae_cell/train_qvae_cell.ipynb'


def cells():
    notebook = json.loads(NOTEBOOK.read_text())
    weight = next(cell for cell in notebook['cells']
                  if cell['cell_type'] == 'code' and 'weights_path = ' in ''.join(cell['source']))
    history = next(cell for cell in notebook['cells']
                   if cell['cell_type'] == 'code' and 'plot_training_history(history)' in ''.join(cell['source']))
    return ''.join(weight['source']), ''.join(history['source'])


def context(tmp_path, load):
    model = torch.nn.Linear(3, 2)
    return {
        'os': os, 'torch': torch, 'args': SimpleNamespace(output_dir=str(tmp_path), load_weights=load),
        'device_obj': torch.device('cpu'), 'model': model,
        'trainer': SimpleNamespace(train=Mock(side_effect=AssertionError('unexpected training'))),
        'train_loader': object(), 'val_loader': object(),
        'vae_optimizer': object(), 'bm_optimizer': object(),
        'plot_training_history': Mock(), 'plt': SimpleNamespace(show=Mock()),
        'display': Mock(), 'pd': SimpleNamespace(DataFrame=Mock()),
        # Rerunning cells must not display history from a previous training run.
        'history': {'train_loss': [999.]},
    }


def test_load_skips_training_and_stale_history_and_preserves_checkpoint(tmp_path):
    source = torch.nn.Linear(3, 2)
    with torch.no_grad():
        source.weight.fill_(.7)
        source.bias.fill_(-.3)
    path = tmp_path / 'immune_kpp_qvae_best.pth'
    torch.save(source.state_dict(), path)
    before = path.read_bytes()
    namespace = context(tmp_path, True)
    weight_cell, history_cell = cells()
    exec(compile(weight_cell, str(NOTEBOOK), 'exec'), namespace)
    exec(compile(history_cell, str(NOTEBOOK), 'exec'), namespace)
    namespace['trainer'].train.assert_not_called()
    namespace['plot_training_history'].assert_not_called()
    namespace['display'].assert_not_called()
    assert namespace['history'] is None
    assert not namespace['model'].training
    assert path.read_bytes() == before
    x = torch.randn(4, 3)
    torch.testing.assert_close(namespace['model'](x), source(x))


def test_training_branch_passes_history_to_existing_plots(tmp_path):
    namespace = context(tmp_path, False)
    history = {'train_loss': [1., .5]}
    namespace['trainer'].train.side_effect = None
    namespace['trainer'].train.return_value = history
    weight_cell, history_cell = cells()
    exec(compile(weight_cell, str(NOTEBOOK), 'exec'), namespace)
    exec(compile(history_cell, str(NOTEBOOK), 'exec'), namespace)
    namespace['trainer'].train.assert_called_once_with(
        namespace['model'], namespace['train_loader'], namespace['val_loader'],
        namespace['vae_optimizer'], namespace['bm_optimizer'],
        os.path.join(str(tmp_path), 'immune_kpp_qvae_best.pth'))
    assert namespace['history'] is history
    namespace['plot_training_history'].assert_called_once_with(history)
    namespace['plt'].show.assert_called_once()
    namespace['pd'].DataFrame.assert_called_once_with(history)
    namespace['display'].assert_called_once()


@pytest.mark.parametrize('kind', ['missing', 'incompatible'])
def test_bad_checkpoint_does_not_fall_back_to_training(tmp_path, kind):
    namespace = context(tmp_path, True)
    if kind == 'incompatible':
        torch.save(torch.nn.Linear(5, 4).state_dict(), tmp_path / 'immune_kpp_qvae_best.pth')
    weight_cell, _ = cells()
    with pytest.raises(FileNotFoundError if kind == 'missing' else RuntimeError):
        exec(compile(weight_cell, str(NOTEBOOK), 'exec'), namespace)
    namespace['trainer'].train.assert_not_called()
