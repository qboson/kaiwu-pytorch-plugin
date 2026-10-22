"""Pin the reconstruction output and error returned by UnsupervisedDBN."""

import numpy as np

from kaiwu.torch_plugin import UnsupervisedDBN


def test_reconstruct_returns_visible_and_per_sample_errors() -> None:
    dbn = UnsupervisedDBN([3, 2]).create_rbm_layer(4).mark_as_trained()
    data = np.random.default_rng(0).standard_normal((3, 4)).astype(np.float32)

    visible, errors = dbn.reconstruct(data, layer_index=0)

    assert visible.shape == (3, 4)
    assert errors.shape == (3,)
    np.testing.assert_allclose(
        errors, np.mean((data - visible) ** 2, axis=1), rtol=1e-6, atol=1e-6
    )


def test_reconstruct_output_is_in_probability_space() -> None:
    dbn = UnsupervisedDBN([3]).create_rbm_layer(4).mark_as_trained()
    visible, _ = dbn.reconstruct(np.zeros((2, 4), dtype=np.float32), layer_index=0)
    assert np.all(visible >= 0.0) and np.all(visible <= 1.0)
