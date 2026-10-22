"""Verbose DBN training does not retain unused full-weight snapshots."""
import ast
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import torch
from torch.optim import SGD
from torch.utils.data import DataLoader, TensorDataset
from kaiwu.torch_plugin import RestrictedBoltzmannMachine


def trainer_class():
    path = Path(__file__).resolve().parents[1] / 'example/dbn_digits/dbn_trainer.py'
    tree = ast.parse(path.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'DBNTrainer')
    namespace = dict(torch=torch, np=np, SGD=SGD, DataLoader=DataLoader, TensorDataset=TensorDataset)
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['DBNTrainer']


class LocalSolver:
    def solve(self, matrix):
        return np.ones((1, len(matrix)), dtype=np.float32)


def run(verbose, plot):
    cls = trainer_class()
    trainer = cls.__new__(cls)
    trainer.learning_rate_rbm = .01
    trainer.batch_size = 2
    trainer.shuffle = False
    trainer.drop_last = False
    trainer.n_epochs_rbm = 3
    trainer.verbose = verbose
    trainer.plot_img = plot
    trainer.sampler = LocalSolver()
    trainer._print_layer_stats = Mock()
    trainer._evaluate_reconstruction_quality = Mock()
    trainer._visualize_training_progress = Mock()
    torch.manual_seed(17)
    rbm = RestrictedBoltzmannMachine(3, 2, device='cpu')
    x = np.array([[0., 1., 0.], [1., 0., 1.]], dtype=np.float32)
    exports = []
    original_numpy = torch.Tensor.numpy

    def record(tensor, *args, **kwargs):
        exports.append(tuple(tensor.shape))
        return original_numpy(tensor, *args, **kwargs)

    with patch.object(torch.Tensor, 'numpy', record):
        output = trainer._train_rbm_layer(rbm, x, 0)
    return output, rbm.state_dict(), exports, trainer


def test_verbose_keeps_training_results_without_snapshot_exports(capsys):
    quiet, quiet_state, _, _ = run(False, False)
    output, state, exports, trainer = run(True, True)
    np.testing.assert_allclose(output, quiet)
    for key in state:
        torch.testing.assert_close(state[key], quiet_state[key])
    # Solver conversion exports are expected. The trainable V x H weight
    # matrix must not be exported solely to populate an unused snapshot list.
    assert (3, 2) not in exports
    assert trainer._visualize_training_progress.call_count == 3
    assert trainer._evaluate_reconstruction_quality.call_count == 3
    text = capsys.readouterr().out
    assert 'Epoch 3/3' in text
    assert '[DBN] Pre-training finished' in text
