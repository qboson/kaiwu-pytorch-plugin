"""Invalid epoch configurations fail before expensive DPLM workflow setup."""
import ast
import operator
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

PATH = Path(__file__).resolve().parents[1] / 'example/qdiffusion/dplm/workflows/train.py'


def workflow(config):
    tree = ast.parse(PATH.read_text())
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef)
             and node.name in ('main', 'validate_training_epochs')]
    namespace = {
        'operator': operator,
        'build_default_workflow_config': lambda: config,
        'torch': SimpleNamespace(cuda=SimpleNamespace(is_available=Mock(
            side_effect=AssertionError('device checked before validation')))),
        'default_outputs_root': Mock(side_effect=AssertionError('output created')),
        'build_generator_from_config': Mock(side_effect=AssertionError('model loaded')),
    }
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(PATH), 'exec'), namespace)
    return namespace


@pytest.mark.parametrize('epochs,error', [(0, ValueError), (-1, ValueError),
    (1.5, TypeError), ('2', TypeError), (True, TypeError), (False, TypeError)])
def test_main_rejects_bad_epochs_before_setup(epochs, error):
    config = SimpleNamespace(train=SimpleNamespace(epochs=epochs, require_cuda=True))
    namespace = workflow(config)
    with pytest.raises(error, match='train.epochs'):
        namespace['main']()
    namespace['torch'].cuda.is_available.assert_not_called()
    namespace['default_outputs_root'].assert_not_called()
    namespace['build_generator_from_config'].assert_not_called()


@pytest.mark.parametrize('epochs', [1, 20, np.int64(3)])
def test_positive_integral_values_reach_normal_device_setup(epochs):
    config = SimpleNamespace(train=SimpleNamespace(epochs=epochs, require_cuda=True))
    namespace = workflow(config)
    namespace['torch'].cuda.is_available.side_effect = None
    namespace['torch'].cuda.is_available.return_value = False
    with pytest.raises(RuntimeError, match='CUDA is required'):
        namespace['main']()
    assert config.train.epochs == int(epochs)
    namespace['torch'].cuda.is_available.assert_called_once()
