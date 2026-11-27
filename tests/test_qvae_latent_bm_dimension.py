"""QVAE rejects a latent width that the Boltzmann machine cannot consume.

``QVAE`` feeds one Bernoulli state per latent unit into its Boltzmann machine,
so ``config.num_latent_units`` has to equal the number of variables of the
injected machine.  Before the invariant was checked, a mismatch was accepted by
the constructor and only blew up much later inside
``RestrictedBoltzmannMachine.objective()`` as
``RuntimeError: size mismatch, got input (2), mat (2x8), vec (4)``.
"""

from pathlib import Path
from types import SimpleNamespace
import inspect
import logging
import sys

import pytest
import torch
from torch import nn

src_root = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(src_root))

import kaiwu

kaiwu.__path__ = [str(src_root / "kaiwu")] + list(kaiwu.__path__)
for module_name in list(sys.modules):
    if module_name == "kaiwu.torch_plugin" or module_name.startswith("kaiwu.torch_plugin."):
        del sys.modules[module_name]
if hasattr(kaiwu, "torch_plugin"):
    delattr(kaiwu, "torch_plugin")

from kaiwu.torch_plugin.full_boltzmann_machine import BoltzmannMachine
from kaiwu.torch_plugin.qvae import QVAE
from kaiwu.torch_plugin.restricted_boltzmann_machine import RestrictedBoltzmannMachine


QVae_logger = logging.getLogger("kaiwu.torch_plugin.qvae")


INPUT_DIMENSION = 4
MESSAGE = "does not match the latent dimension"


class RecordingSampler:
    """Sampler stub: these tests never sample, they only build the model."""

    def solve(self, ising_matrix):
        """Return two zero states so the interface stays callable."""
        return torch.zeros((2, ising_matrix.shape[0]), dtype=torch.float32)


class OpaqueBoltzmannMachine(nn.Module):
    """A third-party Boltzmann machine that does not publish ``num_nodes``."""

    def forward(self, state):
        """Act as the identity so the module is usable."""
        return state


class RecordingHandler(logging.Handler):
    """Attach to the QVAE logger so the assertions do not depend on the root setup."""

    def __init__(self):
        super().__init__(level=logging.DEBUG)
        self.records = []

    def emit(self, record):
        """Keep the record for the assertion that follows."""
        self.records.append(record)


def make_config(latent_units):
    """Build the minimal config ``QVAE.__init__`` reads."""
    return SimpleNamespace(
        num_latent_units=latent_units,
        loss_type="bernoulli",
        dist_beta=1.0,
        kl_beta=0.0,
        weight_decay=0.0,
    )


def build_qvae(latent_units, bm):
    """Instantiate ``QVAE`` with every component supplied explicitly."""
    return QVAE(
        INPUT_DIMENSION,
        None,
        make_config(latent_units),
        encoder=nn.Linear(INPUT_DIMENSION, latent_units),
        decoder=nn.Linear(latent_units, INPUT_DIMENSION),
        bm=bm,
        sampler=RecordingSampler(),
    )


class FactoryBuiltQVAE(QVAE):
    """QVAE whose Boltzmann machine comes from the subclass factory."""

    def __init__(self, config, num_visible, num_hidden):
        self._num_visible = num_visible
        self._num_hidden = num_hidden
        super().__init__(
            INPUT_DIMENSION,
            None,
            config,
            encoder=nn.Linear(INPUT_DIMENSION, config.num_latent_units),
            decoder=nn.Linear(config.num_latent_units, INPUT_DIMENSION),
            sampler=RecordingSampler(),
        )

    def _create_bm(self):
        return RestrictedBoltzmannMachine(self._num_visible, self._num_hidden)


def test_matching_latent_and_bm_nodes_constructs():
    """A latent width equal to the number of BM variables is accepted."""
    bm = RestrictedBoltzmannMachine(2, 2)
    assert bm.num_nodes == 4

    model = build_qvae(4, bm)

    assert model.bm is bm
    assert model._latent_dimensions == bm.num_nodes  # pylint: disable=protected-access


def test_mismatched_latent_and_bm_nodes_raise():
    """The mismatched pair is rejected with both numbers in the message."""
    bm = RestrictedBoltzmannMachine(2, 2)

    with pytest.raises(ValueError, match=MESSAGE) as error:
        build_qvae(8, bm)

    message = str(error.value)
    assert "(4)" in message
    assert "(8)" in message


def test_mismatch_is_reported_by_the_constructor():
    """The failure happens while building, not inside a later loss call."""
    bm = RestrictedBoltzmannMachine(2, 2)

    with pytest.raises(ValueError, match=MESSAGE):
        build_qvae(8, bm)

    # The mismatched model was never handed out to the caller.
    assert bm.num_nodes == 4


def test_full_boltzmann_machine_num_nodes_is_accepted():
    """``BoltzmannMachine`` exposes ``num_nodes`` directly and is accepted."""
    bm = BoltzmannMachine(8)
    assert bm.num_nodes == 8

    model = build_qvae(8, bm)

    assert model.bm is bm


def test_full_boltzmann_machine_mismatch_raises():
    """The check compares against ``num_nodes``, not against a visible count."""
    bm = BoltzmannMachine(4)

    with pytest.raises(ValueError, match=MESSAGE):
        build_qvae(8, bm)


def test_factory_created_bm_is_validated():
    """The factory path is checked too, not only the injection path."""
    with pytest.raises(ValueError, match=MESSAGE):
        FactoryBuiltQVAE(make_config(8), num_visible=2, num_hidden=2)


def test_factory_created_bm_with_matching_nodes_is_accepted():
    """The factory path keeps working when the invariant holds."""
    model = FactoryBuiltQVAE(make_config(4), num_visible=2, num_hidden=2)

    assert model.bm.num_nodes == 4


def test_non_positive_latent_units_keep_the_original_error():
    """A non-positive latent width still trips the pre-existing assertion."""
    for latent_units in (0, -2):
        with pytest.raises(AssertionError, match="Latent dimension must be >0"):
            QVAE(
                INPUT_DIMENSION,
                None,
                make_config(latent_units),
                encoder=nn.Identity(),
                decoder=nn.Identity(),
                bm=RestrictedBoltzmannMachine(2, 2),
                sampler=RecordingSampler(),
            )


def test_injected_bm_without_num_nodes_is_skipped():
    """Custom Boltzmann machines that expose no ``num_nodes`` still build.

    The check cannot judge such a machine, so it is skipped - but the skip is
    logged, because this is the one case where a typo in ``num_latent_units``
    survives construction.
    """
    handler = RecordingHandler()
    previous_level = QVae_logger.level
    QVae_logger.addHandler(handler)
    QVae_logger.setLevel(logging.DEBUG)
    try:
        model = build_qvae(8, OpaqueBoltzmannMachine())
    finally:
        QVae_logger.removeHandler(handler)
        QVae_logger.setLevel(previous_level)

    assert isinstance(model.bm, OpaqueBoltzmannMachine)
    skipped = [record for record in handler.records if "no num_nodes" in record.getMessage()]
    assert len(skipped) == 1
    assert skipped[0].levelno == logging.DEBUG
    assert "OpaqueBoltzmannMachine" in skipped[0].getMessage()


def test_loss_docstring_documents_the_real_signature():
    """``QVAE.loss`` documents exactly the parameters it accepts."""
    docstring = QVAE.loss.__doc__ or ""
    args_block = docstring.split("Args:", 1)[1].split("Returns:", 1)[0]

    documented = [
        line.strip().split(" ", 1)[0]
        for line in args_block.strip().splitlines()
        if "(torch.Tensor)" in line or "(MixtureGeneric)" in line
    ]
    assert documented == ["x", "recon_x", "posterior"]
    assert list(inspect.signature(QVAE.loss).parameters) == ["self", "x", "recon_x", "posterior"]

    assert "q (torch.Tensor)" not in args_block
    assert "zeta (torch.Tensor)" not in args_block
    assert "zeta" not in docstring
    assert "x (torch.Tensor)" in args_block
    assert "recon_x (torch.Tensor)" in args_block
    assert "posterior (MixtureGeneric)" in args_block
