"""Regression test for the MLP progress log of ``MLPClassifier``.

The periodic progress line printed the current epoch as both the numerator and
the denominator (``Epoch {epoch}/{epoch}``), so a 20-epoch run logged
``Epoch 10/10`` instead of ``Epoch 10/20``.

The check runs in a subprocess: ``downstream`` imports ``utils.helpers``, and
``example/qvae_mnist/utils`` is a namespace package that other examples'
regular ``utils`` packages shadow once several example directories share
``sys.path``.
"""

import os
import subprocess
import sys
import textwrap

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
QVAE_DIR = os.path.join(REPO_ROOT, "example", "qvae_mnist")

CHILD_SCRIPT = textwrap.dedent(
    """
    import logging
    import sys

    import numpy as np
    import torch

    sys.path.insert(0, {qvae_dir!r})
    sys.path.insert(0, {src_dir!r})

    import matplotlib
    matplotlib.use("Agg")

    import downstream.classifier as classifier_module
    from downstream.classifier import MLPClassifier

    messages = []

    class _Collect(logging.Handler):
        def emit(self, record):
            messages.append(record.getMessage())

    logging.getLogger().addHandler(_Collect())

    classifier_module.MLPClassifier._train_mlp_epoch = (
        lambda self, **kwargs: (90.0, 1.0)
    )
    classifier_module.MLPClassifier._eval_mlp_epoch = (
        lambda self, **kwargs: (80.0, 1.2)
    )

    rng = np.random.RandomState(0)
    x = rng.rand(40, 8).astype(np.float32)
    y = rng.randint(0, 3, 40)

    classifier = MLPClassifier(
        input_dim=8,
        hidden_dims=[4],
        output_dim=3,
        epochs_mlp=20,
        batch_size_mlp=8,
        device=torch.device("cpu"),
        save_path={save_dir!r},
    )
    classifier.fit(x, y)

    epoch_lines = [m for m in messages if m.startswith("Epoch ")]
    print("EPOCH_LINES:", epoch_lines)
    if any(m.startswith("Epoch 10/20:") for m in epoch_lines):
        print("LOG_OK")
    """
)


def test_progress_log_reports_the_total_epochs(tmp_path):
    script = CHILD_SCRIPT.format(
        qvae_dir=QVAE_DIR,
        src_dir=os.path.join(REPO_ROOT, "src"),
        save_dir=str(tmp_path),
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=600,
    )
    combined = result.stdout + result.stderr

    assert "LOG_OK" in combined, (
        "a 20-epoch run must log 'Epoch 10/20'; child output:\n" + combined
    )
