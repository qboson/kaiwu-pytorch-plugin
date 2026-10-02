"""Regression test for ``PipelineTransformer.fit`` on small subsets.

``run_pipeline.py --num-train-samples N`` builds a random subset of the
dataset; for small N some classes can have a single member.  The transformer
always passed ``stratify=y`` to ``train_test_split``, which rejects that:

    ValueError: The least populated class in y has only 1 member, which is too
    few. The minimum number of groups for any class cannot be less than 2.

The checks run in a subprocess with only the example directory on ``sys.path``:
``example/qvae_mnist/utils`` is a namespace package that other examples'
regular ``utils`` packages would otherwise shadow when collected together.
The trainer is stubbed because real QVAE training requires a Kaiwu licence.
"""

import os
import subprocess
import sys
import textwrap

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
QVAE_DIR = os.path.join(REPO_ROOT, "example", "qvae_mnist")

CHILD_SCRIPT = textwrap.dedent(
    """
    import sys
    import types

    import numpy as np

    sys.path.insert(0, {qvae_dir!r})
    sys.path.insert(0, {src_dir!r})

    import downstream.pipeline as pipeline_module
    from downstream.pipeline import PipelineTransformer


    class _StubTrainer:
        def __init__(self, config=None, custom_train_data=None,
                     custom_test_data=None):
            self.config = config
            self.train_data = custom_train_data
            self.test_data = custom_test_data

        def train(self, run_tsne=False, compute_energy=False):
            return "model", [], []


    pipeline_module.Trainer = _StubTrainer

    def _config():
        return types.SimpleNamespace(run_tsne=False, compute_energy=False)


    rng = np.random.RandomState(0)

    # Singleton class: the stratified split is impossible.
    x = rng.rand(30, 784).astype(np.float32)
    y = np.array([0] * 10 + [1] * 10 + [2] * 9 + [3])
    transformer = PipelineTransformer(_config())
    transformer.fit(x, y)
    train_x, _ = transformer.trainer.train_data
    val_x, _ = transformer.trainer.test_data
    assert train_x.shape[0] + val_x.shape[0] == x.shape[0]
    print("SINGLETON_OK")

    # Balanced data keeps the stratified split.
    x2 = rng.rand(40, 784).astype(np.float32)
    y2 = np.array([0, 1] * 20)
    transformer2 = PipelineTransformer(_config())
    transformer2.fit(x2, y2)
    _, train_y = transformer2.trainer.train_data
    _, val_y = transformer2.trainer.test_data
    assert set(np.unique(train_y)) == {{0, 1}}
    assert set(np.unique(val_y)) == {{0, 1}}
    print("STRATIFIED_OK")
    """
).format(qvae_dir=QVAE_DIR, src_dir=os.path.join(REPO_ROOT, "src"))


def test_pipeline_split_survives_singleton_classes():
    result = subprocess.run(
        [sys.executable, "-c", CHILD_SCRIPT],
        capture_output=True,
        text=True,
        timeout=600,
    )
    combined = result.stdout + result.stderr

    assert "SINGLETON_OK" in combined, (
        "PipelineTransformer.fit must not abort on subsets with singleton "
        f"classes; child output:\n{combined}"
    )
    assert "STRATIFIED_OK" in combined, (
        f"balanced data must still use the stratified split; output:\n{combined}"
    )
    assert "ValueError" not in combined
