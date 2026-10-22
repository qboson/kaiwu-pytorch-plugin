"""Run the real DPLM scorer with a local encoder, without downloading a backbone."""
import ast
from pathlib import Path

import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]


def load_scorer():
    # The methods do not need the heavyweight transformers/tokenizer imports.
    common = ROOT / 'example/qdiffusion/dplm/models/common.py'
    tree = ast.parse(common.read_text())
    pool = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'masked_mean_pool')
    path = ROOT / 'example/qdiffusion/dplm/models/bm.py'
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
    cls.bases = [ast.Name(id='Module', ctx=ast.Load())]
    cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in (
        '_noisy_features', 'build_conditioned_features')]
    namespace = {'torch': torch, 'Module': nn.Module}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[pool, cls], type_ignores=[])),
                 str(path), 'exec'), namespace)
    return namespace['BMConditionedEnergyModel'], namespace['masked_mean_pool']


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(16, 3)
        self.rows = []

    def encode_tokens(self, tokens, attention_mask):
        self.rows.append(len(tokens))
        hidden = self.embedding(tokens)
        # Attention-dependent outputs ensure masks are part of the reuse key.
        return hidden + (hidden * attention_mask.unsqueeze(-1)).sum(1, keepdim=True)


@pytest.mark.parametrize('training,grad,encoder_training', [
    (False, False, False), (True, False, True), (False, True, False), (False, False, True)])
def test_reuse_values_gradients_and_guards(training, grad, encoder_training):
    cls, pool = load_scorer()
    model = cls()
    model.encoder = Encoder()
    model.train(training)
    model.encoder.train(encoder_training)
    noisy = torch.tensor([[1, 2, 3], [1, 2, 3], [1, 2, 3], [4, 5, 6]])
    mask = torch.tensor([[1, 1, 1], [1, 1, 1], [1, 1, 0], [1, 1, 1]])
    candidates = noisy + 1
    with torch.set_grad_enabled(grad):
        expected = torch.cat([pool(model.encoder.encode_tokens(noisy, mask), mask),
                              pool(model.encoder.encode_tokens(candidates, mask), mask)], -1)
        if grad:
            expected_grad = torch.autograd.grad(expected.sum(), model.parameters())
        model.encoder.rows.clear()
        actual = model.build_conditioned_features(noisy, candidates, mask)
        torch.testing.assert_close(actual, expected)
        assert model.encoder.rows == ([3, 4] if not(training or grad or encoder_training) else [4, 4])
        if grad:
            actual_grad = torch.autograd.grad(actual.sum(), model.parameters())
            for a, b in zip(actual_grad, expected_grad):
                torch.testing.assert_close(a, b)
    # A second call must observe weight changes, with no persistent cache.
    with torch.no_grad():
        model.encoder.embedding.weight.add_(1)
        changed = model.build_conditioned_features(noisy, candidates, mask)
    assert not torch.equal(actual, changed)
