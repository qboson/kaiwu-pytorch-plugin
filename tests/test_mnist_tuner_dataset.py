"""ModelTuner data helpers work with PyTorch DataLoader datasets."""
import ast
import logging
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset, Subset


def tuner():
    path = Path(__file__).resolve().parents[1] / 'example/qvae_mnist/trainer/model_tuner.py'
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'ModelTuner')
    namespace = {'torch': torch, 'logger': logging.getLogger(__name__), 'ValueError': ValueError}
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['ModelTuner']()


@pytest.mark.parametrize('shape', [(3,), (1, 2, 2)])
def test_dimensions_and_means_use_dataset_not_batch_policy(shape):
    values = torch.arange(5 * torch.tensor(shape).prod().item(), dtype=torch.float32).reshape(5, *shape)
    dataset = Subset(TensorDataset(values, torch.arange(5)), [0, 2, 4])
    model_tuner = tuner()
    model_tuner.register_dataLoaders(DataLoader(dataset, batch_size=2, drop_last=True, shuffle=True), None)
    assert model_tuner.get_input_dimension() == [values[0].numel()]
    means = model_tuner.get_train_dataset_mean()
    torch.testing.assert_close(means[0], values[[0, 2, 4]].flatten(1).mean(0, keepdim=True))
    assert len(means) == 1
    assert not means[0].requires_grad


class MultiInput(Dataset):
    def __len__(self):
        return 3

    def __getitem__(self, index):
        if index >= len(self):
            raise IndexError
        return [torch.tensor([index, index + 2.]), torch.full((1, 2, 2), float(index))], index


def test_multi_input_keeps_feature_shapes_separate():
    model_tuner = tuner()
    model_tuner.register_dataLoaders(DataLoader(MultiInput(), batch_size=2), None)
    assert model_tuner.get_input_dimension() == [2, 4]
    left, right = model_tuner.get_train_dataset_mean()
    torch.testing.assert_close(left, torch.tensor([[1., 3.]]))
    torch.testing.assert_close(right, torch.ones(1, 4))


def test_empty_dataset_has_actionable_error():
    model_tuner = tuner()
    model_tuner.register_dataLoaders(DataLoader(TensorDataset(torch.empty(0, 2), torch.empty(0))), None)
    for method in (model_tuner.get_input_dimension, model_tuner.get_train_dataset_mean):
        with pytest.raises(ValueError, match='empty'):
            method()
