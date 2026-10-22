"""Feature-only DPLM encoding skips logits while preserving backbone gradients."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import ast

import pytest
import torch
from torch import nn

from transformers import EsmConfig

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('esm_patch_local',
    ROOT / 'example/qdiffusion/dplm/models/esm_patch.py')
patch_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(patch_module)


def encoder_class():
    path = ROOT / 'example/qdiffusion/dplm/models/common.py'
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'DPLMFeatureEncoder')
    future = ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)
    namespace = {'nn': nn, 'torch': torch}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[future, cls], type_ignores=[])),
                 str(path), 'exec'), namespace)
    return namespace['DPLMFeatureEncoder']


@pytest.mark.parametrize('conditioned', [False, True])
def test_actual_tiny_esm_features_and_gradients_without_lm_projection(conditioned):
    config = EsmConfig(vocab_size=12, hidden_size=8, intermediate_size=16,
        num_hidden_layers=1, num_attention_heads=2, pad_token_id=0, mask_token_id=3,
        position_embedding_type='absolute', token_dropout=False)
    tokenizer = SimpleNamespace(mask_token_id=3, pad_token_id=0, cls_token_id=1,
                                eos_token_id=2, _token_to_id={'X': 4})
    with patch.object(patch_module.AutoTokenizer, 'from_pretrained', return_value=tokenizer):
        net = patch_module._EsmForDPLM(config, dropout=0.)
    net.eval()
    tokens = torch.tensor([[1, 5, 6, 2], [1, 7, 8, 2]])
    mask = tokens.ne(0)
    inputs_embeds = net.get_input_embeddings()(tokens) if conditioned else None
    expected = net(input_ids=tokens, inputs_embeds=inputs_embeds, attention_mask=mask)
    assert expected['logits'].shape == (2, 4, 12)
    parameters = list(net.esm.parameters())
    expected_grad = torch.autograd.grad(expected['last_hidden_state'].square().sum(),
                                        parameters, allow_unused=True)
    feature_encoder = encoder_class()(SimpleNamespace(net=net))
    with patch.object(net.lm_head, 'forward', side_effect=AssertionError('unused LM head')):
        if conditioned:
            actual = feature_encoder.encode_conditioned(tokens, net.get_input_embeddings()(tokens), mask)
        else:
            actual = feature_encoder.encode_tokens(tokens, mask)
    torch.testing.assert_close(actual, expected['last_hidden_state'])
    actual_grad = torch.autograd.grad(actual.square().sum(), parameters, allow_unused=True)
    for actual_value, expected_value in zip(actual_grad, expected_grad):
        if expected_value is None:
            assert actual_value is None
        else:
            torch.testing.assert_close(actual_value, expected_value)
    # Default proposal calls must still execute the LM head.
    with patch.object(net.lm_head, 'forward', wraps=net.lm_head.forward) as head:
        proposal = net(tokens)
    head.assert_called_once()
    torch.testing.assert_close(proposal['logits'], expected['logits'])
