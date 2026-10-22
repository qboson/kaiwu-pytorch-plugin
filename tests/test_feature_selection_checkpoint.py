"""A saved feature selector must resume its actual mask-update schedule."""
from copy import deepcopy
from itertools import product
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from kaiwu.torch_plugin.maifs import plugin

assert Path(plugin.__file__).resolve() == (
    Path(__file__).resolve().parents[1] / 'src/kaiwu/torch_plugin/maifs/plugin.py'
)


def training_case(dtype=torch.float32, interval=2):
    selector = plugin.FeatureSelectionWrapper(
        nn.Linear(2, 1, bias=False), feature_dim=2, lambda_reg=.1,
        min_selected_features=0, mask_update_epochs=interval,
        solver='local_search', solver_kwargs={'max_iter': 100},
    ).to(dtype=dtype)
    with torch.no_grad():
        selector.model.weight.copy_(torch.tensor([[.8, .7]], dtype=dtype))
    features = torch.tensor([[1., 0.], [0., 1.], [1., 1.], [-1., 0.], [0., -1.]],
                            dtype=dtype)
    labels = 2 * features[:, :1]
    loader = DataLoader(TensorDataset(features, labels), batch_size=5, shuffle=False)
    optimiser = torch.optim.Adam(selector.parameters(), lr=.05)
    return selector, optimiser, loader, features, labels


def record_real_updates(selector, loader, monkeypatch):
    """Observe the public update call while retaining its derivatives and solver."""
    calls = []
    actual_update = selector.update_mask
    def update(*args, **kwargs):
        mask = actual_update(*args, **kwargs)
        calls.append(selector._trained_epochs)
        # For this linear MSE, its Taylor QUBO equals the original loss plus
        # lambda * mask.sum(), up to an irrelevant constant. Enumerate it directly.
        features, labels = loader.dataset.tensors
        masks = np.array(list(product((0, 1), repeat=2)))
        losses = []
        with torch.no_grad():
            for candidate in masks:
                prediction = selector.model(features * torch.tensor(candidate, dtype=features.dtype))
                losses.append(float(nn.MSELoss()(prediction, labels)) + .1 * candidate.sum())
        np.testing.assert_array_equal(mask, masks[np.argmin(losses)])
        return mask
    monkeypatch.setattr(selector, 'update_mask', update)
    return calls


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('interval,trained', [(2, 1), (3, 2), (2, 3)])
def test_saved_adam_and_model_continue_through_real_mask_updates(
        tmp_path, monkeypatch, dtype, interval, trained):
    selector, optimiser, loader, features, _ = training_case(dtype, interval)
    reference_updates = record_real_updates(selector, loader, monkeypatch)
    selector.fit_weights(loader, nn.MSELoss(), optimiser, train_epochs=trained)
    path = tmp_path / 'selector.pt'
    torch.save({'model': selector.state_dict(), 'optimizer': optimiser.state_dict()}, path)
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    resumed, resumed_optimiser, _, _, _ = training_case(dtype, interval)
    resumed.load_state_dict(checkpoint['model'])
    resumed_optimiser.load_state_dict(checkpoint['optimizer'])
    assert resumed.solver == selector.solver
    assert resumed.solver_kwargs == selector.solver_kwargs
    assert resumed.mask_update_epochs == selector.mask_update_epochs
    torch.testing.assert_close(resumed(features), selector(features), rtol=0, atol=0)
    resumed_updates = record_real_updates(resumed, loader, monkeypatch)
    previous_updates = len(reference_updates)
    # The first continuation epoch is a mask update. The following step tests
    # that its selected mask also reaches actual model gradients and Adam state.
    for epoch in range(trained + 1, trained + 3):
        selector.fit_weights(loader, nn.MSELoss(), optimiser)
        resumed.fit_weights(loader, nn.MSELoss(), resumed_optimiser)
        assert resumed._trained_epochs == selector._trained_epochs == epoch
        assert type(resumed._trained_epochs) is int
        torch.testing.assert_close(resumed.mask, selector.mask, rtol=0, atol=0)
        torch.testing.assert_close(resumed.model.state_dict(), selector.model.state_dict(), rtol=0, atol=0)
        torch.testing.assert_close(resumed_optimiser.state_dict(), optimiser.state_dict(), rtol=0, atol=0)
        torch.testing.assert_close(resumed(features), selector(features), rtol=0, atol=0)
    assert resumed_updates == reference_updates[previous_updates:]
    assert resumed_updates[0] == trained + 1


@pytest.mark.parametrize('nested', [False, True])
def test_tensor_state_entries_and_nested_prefix_restore_progress(tmp_path, nested):
    selector, optimiser, loader, features, _ = training_case(torch.float64)
    selector.fit_weights(loader, nn.MSELoss(), optimiser, train_epochs=3)
    model = nn.Sequential(selector) if nested else selector
    checkpoint = {name: value.detach().clone() for name, value in model.state_dict().items()}
    extra_key = '0._extra_state' if nested else '_extra_state'
    assert checkpoint[extra_key].dtype == torch.int64
    assert checkpoint[extra_key].ndim == 0
    assert checkpoint[extra_key].item() == 3
    assert checkpoint[extra_key].device == selector.mask.device
    fresh, _, _, _, _ = training_case(torch.float64)
    destination = nn.Sequential(fresh) if nested else fresh
    path = tmp_path / 'tensor-state.pt'
    torch.save(checkpoint, path)
    destination.load_state_dict(torch.load(path, weights_only=True))
    assert fresh._trained_epochs == 3 and type(fresh._trained_epochs) is int
    torch.testing.assert_close(destination(features), model(features), rtol=0, atol=0)
    # The scalar snapshot has no alias to later runtime counter changes.
    selector.fit_weights(loader, nn.MSELoss(), optimiser)
    assert checkpoint[extra_key].item() == 3


@pytest.mark.parametrize('nested', [False, True])
@pytest.mark.parametrize('already_trained', [False, True])
def test_strict_legacy_checkpoint_load_resets_unknown_progress_without_mutation(
        nested, already_trained):
    source, optimiser, loader, features, _ = training_case()
    source.fit_weights(loader, nn.MSELoss(), optimiser, train_epochs=2)
    saved_model = nn.Sequential(source) if nested else source
    legacy = {name: value.detach().clone() for name, value in saved_model.state_dict().items()
              if not name.endswith('_extra_state')}
    originals = dict(legacy)
    destination, new_optimiser, _, _, _ = training_case()
    if already_trained:
        destination.fit_weights(loader, nn.MSELoss(), new_optimiser, train_epochs=3)
    destination_model = nn.Sequential(destination) if nested else destination
    result = destination_model.load_state_dict(legacy, strict=True)
    assert not result.missing_keys and not result.unexpected_keys
    assert destination._trained_epochs == 0
    assert legacy.keys() == originals.keys()
    assert all(legacy[name] is originals[name] for name in originals)
    torch.testing.assert_close(destination_model(features), saved_model(features), rtol=0, atol=0)


@pytest.mark.parametrize('missing_key', ['mask', 'model.weight'])
def test_legacy_compatibility_preserves_unrelated_strict_missing_keys(missing_key):
    selector, _, _, _, _ = training_case()
    checkpoint = deepcopy(selector.state_dict())
    checkpoint.pop('_extra_state', None)
    del checkpoint[missing_key]
    with pytest.raises(RuntimeError, match='Missing key'):
        selector.load_state_dict(checkpoint, strict=True)


def test_legacy_compatibility_preserves_unrelated_strict_unexpected_keys():
    selector, _, _, _, _ = training_case()
    checkpoint = deepcopy(selector.state_dict())
    checkpoint.pop('_extra_state', None)
    checkpoint['unrelated._extra_state'] = torch.tensor(7)
    with pytest.raises(RuntimeError, match='Unexpected key'):
        selector.load_state_dict(checkpoint, strict=True)
