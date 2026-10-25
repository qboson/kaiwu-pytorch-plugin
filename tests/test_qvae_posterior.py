"""Contract test for ``kaiwu.torch_plugin.qvae.QVAE.posterior``."""

from __future__ import annotations

import torch

from kaiwu.torch_plugin import qvae as qvae_module


class _FakeMixture:
    instances: list["_FakeMixture"] = []

    def __init__(self, logits, beta):
        self.logit_mu = logits
        self.beta = beta
        self.calls: list[bool] = []
        _FakeMixture.instances.append(self)

    def reparameterize(self, is_training):
        self.calls.append(is_training)
        return self.logit_mu * 2.0


def _model(training: bool):
    model = object.__new__(qvae_module.QVAE)
    object.__setattr__(model, "training", training)
    return model


def test_builds_a_mixture_with_the_requested_beta(monkeypatch):
    monkeypatch.setattr(qvae_module, "MixtureGeneric", _FakeMixture)
    logits = torch.randn(2, 3)

    posterior, sample = qvae_module.QVAE.posterior(_model(True), logits, 3.5)

    assert isinstance(posterior, _FakeMixture)
    assert posterior.beta == 3.5
    assert posterior.logit_mu is logits
    assert torch.equal(sample, logits * 2.0)
    assert sample.shape == logits.shape


def test_forwards_the_training_flag_to_reparameterize(monkeypatch):
    monkeypatch.setattr(qvae_module, "MixtureGeneric", _FakeMixture)
    logits = torch.randn(4, 2)

    _, _ = qvae_module.QVAE.posterior(_model(True), logits, 1.0)
    _, _ = qvae_module.QVAE.posterior(_model(False), logits, 1.0)

    assert _FakeMixture.instances[-2].calls == [True]
    assert _FakeMixture.instances[-1].calls == [False]
