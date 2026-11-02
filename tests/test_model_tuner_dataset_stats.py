"""Regression tests for ModelTuner dataset dimension and mean helpers.

Issue #531: ``ModelTuner.get_input_dimension`` called ``get_input_size`` on the
DataLoader, which standard PyTorch loaders do not implement. The mean helper
then failed too and confused vector coordinates with separate model inputs.
"""

import importlib.util
from pathlib import Path
import sys

import pytest
import torch
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset

EXAMPLE_ROOT = Path(__file__).resolve().parents[1] / "example" / "qvae_mnist"
sys.path.insert(0, str(EXAMPLE_ROOT))

_SPEC = importlib.util.spec_from_file_location(
    "model_tuner_under_test", EXAMPLE_ROOT / "trainer" / "model_tuner.py"
)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
ModelTuner = _MODULE.ModelTuner


def make_tuner(loader) -> ModelTuner:
    """Registers a loader on a fresh tuner."""
    tuner = ModelTuner()
    tuner.register_dataLoaders(loader, None)
    return tuner


def test_tensor_dataset_dimension_and_mean_with_drop_last_and_shuffle() -> None:
    """Statistics use the dataset, not the loader's shuffled/dropped batches."""
    features = torch.randn(11, 3, 4)
    targets = torch.zeros(11)
    loader = DataLoader(
        TensorDataset(features, targets),
        batch_size=4,
        shuffle=True,
        drop_last=True,
    )
    tuner = make_tuner(loader)

    assert tuner.get_input_dimension() == [12]

    means = tuner.get_train_dataset_mean()
    expected = features.reshape(11, -1).mean(dim=0, keepdim=True)
    assert len(means) == 1
    assert means[0].shape == (1, 12)
    torch.testing.assert_close(means[0], expected)


def test_image_input_is_flattened_to_its_feature_width() -> None:
    """Image-shaped samples contribute one flattened feature width."""
    features = torch.randn(9, 2, 5, 5)
    loader = DataLoader(TensorDataset(features, torch.zeros(9)), batch_size=3)
    tuner = make_tuner(loader)

    assert tuner.get_input_dimension() == [50]
    means = tuner.get_train_dataset_mean()
    torch.testing.assert_close(
        means[0], features.reshape(9, -1).mean(dim=0, keepdim=True)
    )


def test_subset_means_only_use_selected_samples() -> None:
    """A Subset loader reports statistics for the selected observations."""
    features = torch.randn(10, 4)
    dataset = TensorDataset(features, torch.zeros(10))
    selected = [1, 3, 7]
    loader = DataLoader(Subset(dataset, selected), batch_size=2, drop_last=True)
    tuner = make_tuner(loader)

    assert tuner.get_input_dimension() == [4]
    means = tuner.get_train_dataset_mean()
    torch.testing.assert_close(
        means[0], features[selected].mean(dim=0, keepdim=True)
    )


class MultiInputDataset(Dataset):
    """Dataset with two independently shaped input tensors."""

    def __init__(self) -> None:
        self.first = torch.randn(7, 6)
        self.second = torch.randn(7, 4)
        self.targets = torch.zeros(7)

    def __len__(self) -> int:
        return 7

    def __getitem__(self, index):
        return (self.first[index], self.second[index]), self.targets[index]


def test_multiple_inputs_return_one_width_and_mean_each() -> None:
    """Multiple model inputs are reported separately, not as coordinates."""
    dataset = MultiInputDataset()
    tuner = make_tuner(DataLoader(dataset, batch_size=3, shuffle=True))

    assert tuner.get_input_dimension() == [6, 4]

    means = tuner.get_train_dataset_mean()
    assert [mean.shape for mean in means] == [(1, 6), (1, 4)]
    torch.testing.assert_close(means[0], dataset.first.mean(dim=0, keepdim=True))
    torch.testing.assert_close(means[1], dataset.second.mean(dim=0, keepdim=True))


def test_empty_dataset_raises_a_clear_error() -> None:
    """Empty datasets cannot describe an input width or a mean."""
    loader = DataLoader(TensorDataset(torch.empty(0, 3), torch.empty(0)))
    tuner = make_tuner(loader)

    with pytest.raises(ValueError, match="empty dataset"):
        tuner.get_input_dimension()
    with pytest.raises(ValueError, match="empty dataset"):
        tuner.get_train_dataset_mean()


class LegacyLoader:
    """Loader wrapper exposing only the legacy get_input_size() contract."""

    dataset = None

    @staticmethod
    def get_input_size():
        return 12


def test_legacy_get_input_size_contract_is_still_supported() -> None:
    """The pre-existing wrapper API remains usable for dimension queries."""
    assert make_tuner(LegacyLoader()).get_input_dimension() == [12]
