"""The bundled DBN trainer must build layers for a freshly created DBN."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "example" / "dbn_digits")
)

from dbn_trainer import DBNTrainer  # noqa: E402  pylint: disable=wrong-import-position
from kaiwu.torch_plugin import UnsupervisedDBN  # noqa: E402  pylint: disable=wrong-import-position


class _StubSampler:  # pylint: disable=too-few-public-methods
    def solve(self, ising_matrix):
        return np.ones((1, ising_matrix.shape[0]), dtype=np.float32)


def test_train_builds_layers_for_an_unbuilt_model() -> None:
    trainer = DBNTrainer(
        use_cim=False,
        verbose=False,
        n_epochs_rbm=1,
        batch_size=2,
        shuffle=False,
    )
    trainer.sampler = _StubSampler()

    trained = trainer.train(
        UnsupervisedDBN([3]), np.zeros((4, 4), dtype=np.float32)
    )

    assert trained.num_layers == 1
    transformed = trained.transform(np.zeros((2, 4), dtype=np.float32))
    assert transformed.shape == (2, 3)
