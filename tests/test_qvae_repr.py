"""Contract test for ``AutoEncoderBase.__repr__`` (inherited by QVAE)."""

from __future__ import annotations

from kaiwu.torch_plugin import qvae as qvae_module


def _bare_model(**attributes):
    model = object.__new__(qvae_module.QVAE)
    for name, value in attributes.items():
        object.__setattr__(model, name, value)
    return model


def test_repr_lists_every_instance_attribute_on_its_own_line():
    model = _bare_model(encoder="enc", decoder="dec", _dataset_mean=None)

    text = qvae_module.QVAE.__repr__(model)

    assert text.splitlines() == [
        "('encoder', 'enc')",
        "('decoder', 'dec')",
        "('_dataset_mean', None)",
    ]


def test_repr_is_empty_for_a_bare_instance():
    model = object.__new__(qvae_module.QVAE)

    assert qvae_module.QVAE.__repr__(model) == ""
