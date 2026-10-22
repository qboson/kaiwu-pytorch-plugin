"""Real-image collection bounds the copy at the concatenation boundary."""
import ast
import operator
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset


def collect():
    path = Path(__file__).resolve().parents[1] / 'example/qvae_mnist/utils/helpers.py'
    tree = ast.parse(path.read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'get_real_images')
    namespace = {'torch': torch, 'operator': operator}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['get_real_images']


@pytest.mark.parametrize('batch_size,requested', [(20, 3), (4, 7), (4, 8), (4, 25), (4, np.int64(3))])
def test_prefix_and_concat_storage_bound(batch_size, requested):
    x = torch.arange(60, dtype=torch.float32).reshape(20, 3)
    loader = DataLoader(TensorDataset(x, torch.arange(20)), batch_size=batch_size)
    calls = []
    original = torch.cat

    def cat(parts, **kwargs):
        calls.append(sum(t.numel() * t.element_size() for t in parts))
        return original(parts, **kwargs)

    with patch.object(torch, 'cat', cat):
        result = collect()(loader, requested)
    torch.testing.assert_close(result, x[:requested])
    assert calls == [min(len(x), requested) * 3 * 4]


def test_stops_before_requesting_another_batch():
    def batches():
        yield torch.ones(5, 3), None
        raise AssertionError('read beyond requested prefix')
    torch.testing.assert_close(collect()(batches(), 2), torch.ones(2, 3))


@pytest.mark.parametrize('value,error', [(0, ValueError), (-1, ValueError), (True, TypeError),
    (False, TypeError), (1.5, TypeError), ('2', TypeError)])
def test_invalid_requests_do_not_iterate_source(value, error):
    def batches():
        raise AssertionError('source consumed')
        yield
    with pytest.raises(error, match='n_images'):
        collect()(batches(), value)


def test_empty_batches_and_empty_sources():
    torch.testing.assert_close(collect()([(torch.empty(0, 3), None), (torch.ones(2, 3), None)], 3),
                               torch.ones(2, 3))
    for batches in ([], [(torch.empty(0, 3), None)]):
        with pytest.raises(ValueError, match='no images'):
            collect()(batches, 3)
