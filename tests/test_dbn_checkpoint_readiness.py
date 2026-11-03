"""DBN checkpoints preserve readiness without guessing for legacy weights."""
import copy
import io

import numpy as np
import pytest
import torch
from torch import nn

from kaiwu.torch_plugin.dbn import UnsupervisedDBN


def make_model(trained=False):
    model = UnsupervisedDBN([3, 2]).create_rbm_layer(4)
    if trained:
        model.mark_as_trained()
    return model


def samples():
    return np.array([[0., 1., 0., 1.], [1., 0., 1., 0.]], dtype=np.float32)


def assert_same_weights(source, target):
    for left, right in zip(source.parameters(), target.parameters()):
        assert torch.equal(left, right)


@pytest.mark.parametrize('trained', [False, True])
@pytest.mark.parametrize('target_trained', [False, True])
def test_ready_and_unready_checkpoints_restore_exact_state(trained, target_trained):
    source, target = make_model(trained), make_model(target_trained)
    saved = copy.deepcopy(source.state_dict())
    result = target.load_state_dict(saved, strict=True)
    assert not result.missing_keys and not result.unexpected_keys
    assert target._is_trained is trained
    assert_same_weights(source, target)
    assert saved['_extra_state'] == {'version': 1, 'is_trained': trained}
    if trained:
        np.testing.assert_allclose(source(samples()), target(samples()))
    else:
        with pytest.raises(ValueError, match='not trained'):
            target(samples())


@pytest.mark.parametrize('trained', [False, True])
def test_weights_only_serialization_roundtrip(trained):
    source = make_model(trained)
    buffer = io.BytesIO()
    torch.save(source.state_dict(), buffer)
    buffer.seek(0)
    saved = torch.load(buffer, weights_only=True)
    target = make_model()
    target.load_state_dict(saved)
    assert target._is_trained is trained
    assert_same_weights(source, target)


@pytest.mark.parametrize('target_trained', [False, True])
@pytest.mark.parametrize('strict', [False, True])
def test_legacy_weights_load_but_require_explicit_mark(strict, target_trained):
    source, target = make_model(True), make_model(target_trained)
    legacy = source.state_dict()
    legacy.pop('_extra_state', None)
    original_keys = list(legacy)
    result = target.load_state_dict(legacy, strict=strict)
    assert not result.missing_keys and not result.unexpected_keys
    assert not target._is_trained
    assert list(legacy) == original_keys
    assert_same_weights(source, target)
    with pytest.raises(ValueError, match='not trained'):
        target(samples())
    target.mark_as_trained()
    np.testing.assert_allclose(source(samples()), target(samples()))


@pytest.mark.parametrize('legacy', [False, True])
def test_nested_module_loading_handles_prefix(legacy):
    source, target = nn.Module(), nn.Module()
    source.dbn, target.dbn = make_model(True), make_model(True)
    saved = source.state_dict()
    if legacy:
        saved.pop('dbn._extra_state', None)
    result = target.load_state_dict(saved)
    assert not result.missing_keys and not result.unexpected_keys
    assert target.dbn._is_trained is (not legacy)
    assert_same_weights(source.dbn, target.dbn)


@pytest.mark.parametrize('metadata', [None, {}, True,
    {'version': 2, 'is_trained': True},
    {'version': True, 'is_trained': True},
    {'version': 1, 'is_trained': 1},
    {'version': 1, 'is_trained': 'true'},
    {'version': 1},
])
def test_invalid_metadata_cannot_mark_model_ready(metadata):
    target = make_model(True)
    saved = make_model(True).state_dict()
    saved['_extra_state'] = metadata
    with pytest.raises(ValueError, match='DBN checkpoint'):
        target.load_state_dict(saved)
    assert not target._is_trained


@pytest.mark.parametrize('problem', ['missing', 'shape', 'unexpected'])
@pytest.mark.parametrize('nested', [False, True])
def test_failed_weight_load_leaves_model_unready(problem, nested):
    source, target = make_model(True), make_model(True)
    if nested:
        outer_source, outer_target = nn.Module(), nn.Module()
        outer_source.dbn, outer_target.dbn = source, target
    else:
        outer_source, outer_target = source, target
    saved = outer_source.state_dict()
    prefix = 'dbn.' if nested else ''
    key = prefix + 'rbm_layers.0.quadratic_coef'
    if problem == 'missing':
        saved.pop(key)
    elif problem == 'shape':
        saved[key] = torch.zeros(1)
    else:
        saved[prefix + 'unexpected_weight'] = torch.zeros(1)
    with pytest.raises(RuntimeError):
        outer_target.load_state_dict(saved, strict=True)
    assert not target._is_trained


def test_partial_non_strict_load_does_not_mark_ready():
    source, target = make_model(True), make_model(True)
    saved = source.state_dict()
    saved.pop('rbm_layers.0.linear_bias')
    result = target.load_state_dict(saved, strict=False)
    assert result.missing_keys == ['rbm_layers.0.linear_bias']
    assert not target._is_trained


def test_layer_rebuild_resets_loaded_readiness():
    target = make_model()
    target.load_state_dict(make_model(True).state_dict())
    assert target._is_trained
    target.create_rbm_layer(4)
    assert not target._is_trained


def test_unbuilt_state_roundtrip_is_unready():
    source, target = UnsupervisedDBN([3]), UnsupervisedDBN([3])
    target.load_state_dict(source.state_dict())
    assert not target._is_trained and target.rbm_layers is None
    with pytest.raises(ValueError, match='not built'):
        target(samples())


def test_checkpoint_metadata_is_an_independent_snapshot():
    source = make_model(True)
    saved = source.state_dict()
    source.create_rbm_layer(4)
    assert saved['_extra_state']['is_trained'] is True
    source.set_extra_state({'version': 1, 'is_trained': False})
    assert not source._is_trained


def test_legacy_compatibility_does_not_hide_missing_weights():
    saved = make_model(True).state_dict()
    saved.pop('_extra_state')
    saved.pop('rbm_layers.1.linear_bias')
    target = make_model(True)
    with pytest.raises(RuntimeError, match='rbm_layers.1.linear_bias'):
        target.load_state_dict(saved)
    assert not target._is_trained


def test_structure_must_be_created_before_loading_trained_weights():
    target = UnsupervisedDBN([3, 2])
    with pytest.raises(RuntimeError, match='Unexpected key'):
        target.load_state_dict(make_model(True).state_dict())
    assert target.rbm_layers is None and not target._is_trained


def test_non_strict_unexpected_key_keeps_model_unready():
    saved = make_model(True).state_dict()
    saved['unexpected'] = torch.zeros(1)
    target = make_model(True)
    result = target.load_state_dict(saved, strict=False)
    assert result.unexpected_keys == ['unexpected']
    assert not target._is_trained


def test_nested_prefix_does_not_hide_sibling_missing_keys():
    source, target = nn.Module(), nn.Module()
    source.dbn, target.dbn = make_model(True), make_model(True)
    source.other, target.other = nn.Linear(2, 1), nn.Linear(2, 1)
    saved = source.state_dict()
    saved.pop('dbn._extra_state')
    saved.pop('other.bias')
    with pytest.raises(RuntimeError, match='other.bias'):
        target.load_state_dict(saved)
    assert not target.dbn._is_trained


@pytest.mark.parametrize('problem', ['metadata', 'shape', 'missing'])
def test_valid_reload_after_failure_restores_readiness(problem):
    source, target = make_model(True), make_model(True)
    broken = copy.deepcopy(source.state_dict())
    if problem == 'metadata':
        broken['_extra_state']['version'] = 100
    elif problem == 'shape':
        broken['rbm_layers.0.linear_bias'] = torch.zeros(1)
    else:
        broken.pop('rbm_layers.0.linear_bias')
    with pytest.raises((RuntimeError, ValueError)):
        target.load_state_dict(broken)
    assert not target._is_trained
    target.load_state_dict(source.state_dict())
    assert target._is_trained and target._checkpoint_loading is None
    assert_same_weights(source, target)
