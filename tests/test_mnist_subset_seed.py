"""Seed subset membership without affecting unrelated PyTorch randomness."""

import importlib.util
from pathlib import Path

import pytest
import torch


class ToyDigits:
    def __init__(self, *, root, train, download, transform):
        self.size = 60 if train else 24

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return torch.tensor([float(index)]), index % 10


@pytest.fixture
def loaders(monkeypatch):
    pytest.importorskip("torchvision", exc_type=ImportError)
    directory = Path(__file__).resolve().parents[1] / "example" / "qvae_mnist" / "utils"
    spec = importlib.util.spec_from_file_location("kpp_mnist_subsets", directory / "loadMNIST.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "MNIST", ToyDigits)
    return module


def test_seeded_subsets_repeat_independent_of_global_random_draws(loaders):
    before = torch.random.get_rng_state()
    first_train, first_test = loaders.loadMNIST(num_evts_train=20, num_evts_test=12, subset_seed=9)
    torch.testing.assert_close(torch.random.get_rng_state(), before)
    torch.rand(500)
    second_train, second_test = loaders.loadMNIST(num_evts_train=20, num_evts_test=12, subset_seed=9)
    torch.testing.assert_close(first_train.dataset.indices, second_train.dataset.indices)
    torch.testing.assert_close(first_test.dataset.indices, second_test.dataset.indices)
    assert len(set(first_test.dataset.indices.tolist())) == 12


def test_test_subset_is_independent_of_training_sample_count(loaders):
    _, first_test = loaders.loadMNIST(num_evts_train=20, num_evts_test=12, subset_seed=9)
    full_train, second_test = loaders.loadMNIST(num_evts_train=60, num_evts_test=12, subset_seed=9)
    assert len(full_train.dataset) == 60
    torch.testing.assert_close(first_test.dataset.indices, second_test.dataset.indices)


def test_class_based_loader_forwards_seed(loaders):
    wrapper = loaders.MNISTDataLoader(num_evts_train=20, num_evts_test=12, subset_seed=12)
    _, wrapped_test = wrapper.load()
    _, direct_test = loaders.loadMNIST(num_evts_train=20, num_evts_test=12, subset_seed=12)
    _, different_seed = loaders.loadMNIST(num_evts_train=20, num_evts_test=12, subset_seed=13)
    torch.testing.assert_close(wrapped_test.dataset.indices, direct_test.dataset.indices)
    assert not torch.equal(different_seed.dataset.indices, direct_test.dataset.indices)
