"""Compare the direct path with pool dispatch using real training and legal spins."""
import importlib.util
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch

spec = importlib.util.spec_from_file_location(
    'bm_example', Path(__file__).resolve().parents[1] / 'example/bm_generation/trainer.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class Solver:
    def solve(self, matrix):
        return np.ones((1, len(matrix)), dtype=np.float32)


class Pool:
    def __init__(self, **kwargs):
        self.closed = self.joined = False

    def map(self, fn, args):
        return list(map(fn, args))

    def close(self):
        self.closed = True

    def join(self):
        self.joined = True


def train(processes):
    torch.manual_seed(17)
    saver = Mock()
    trainer = module.Trainer([torch.tensor([[0., 1.], [1., 0.], [1., 1.]])],
        saver, Solver(), num_visible=2, num_hidden=1, num_output=1)
    trainer.train(2, 'unused', num_processes=processes)
    return trainer.bm_net.state_dict(), saver.output_loss.call_args_list


def test_direct_training_skips_pool_shared_storage_and_logging_changes():
    with patch.object(module.mp, 'Pool', side_effect=AssertionError('pool created')), patch.object(
        torch.Tensor, 'share_memory_', side_effect=AssertionError('shared storage')
    ), patch.object(module.kw.common, 'set_log_level', side_effect=AssertionError('global log changed')):
        actual, losses = train(1)
    with patch.object(module.mp, 'Pool', Pool), patch.object(module.kw.common, 'set_log_level'):
        expected, expected_losses = train(2)
    for key in actual:
        torch.testing.assert_close(actual[key], expected[key])
    assert losses == expected_losses


def test_direct_solver_failure_propagates_without_creating_pool():
    trainer = module.Trainer([torch.ones(2, 2)], Mock(), Solver(),
                             num_visible=2, num_hidden=1, num_output=1)
    with patch.object(module.mp, 'Pool', side_effect=AssertionError('pool created')), patch.object(
        trainer.bm_net, 'condition_sample', side_effect=RuntimeError('conditional failed')
    ), pytest.raises(RuntimeError, match='conditional failed'):
        trainer.train(1, 'unused')
