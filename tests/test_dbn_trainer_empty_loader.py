"""Regression test for the empty-loader crash in ``DBNTrainer._train_rbm_layer``.

With the documented ``drop_last=True`` option and a dataset smaller than
``batch_size`` the DataLoader yields no batches, so ``avg_loss`` was never
assigned and the post-loop progress print raised ``UnboundLocalError``.  The
trainer must reject that configuration with a clear error instead.
"""

import os
import subprocess
import sys
import textwrap

import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DBN_DIR = os.path.join(REPO_ROOT, "example", "dbn_digits")

CHILD_SCRIPT = textwrap.dedent(
    """
    import sys
    import numpy as np

    sys.path.insert(0, {dbn_dir!r})
    sys.path.insert(0, {src_dir!r})
    from dbn_trainer import DBNPretrainer

    x = np.random.RandomState(0).rand(8, 8).astype(np.float32)
    pretrainer = DBNPretrainer(
        hidden_layers_structure=[8],
        n_epochs_rbm=1,
        batch_size=32,   # larger than the dataset
        drop_last=True,  # documented option
        verbose=True,
    )
    try:
        pretrainer.fit(x)
    except ValueError as exc:
        print("RAISED_VALUE_ERROR: {{}}".format(exc))
    except Exception as exc:
        print("RAISED_OTHER: {{!r}}".format(exc))
    else:
        print("RETURNED_WITHOUT_ERROR")
    """
).format(dbn_dir=DBN_DIR, src_dir=os.path.join(REPO_ROOT, "src"))


def test_empty_training_loader_is_rejected(tmp_path):
    result = subprocess.run(
        [sys.executable, "-c", CHILD_SCRIPT],
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
        timeout=300,
    )
    combined = result.stdout + result.stderr

    assert "RAISED_VALUE_ERROR" in combined, (
        "DBNTrainer must fail fast when drop_last empties the loader; "
        f"child output:\n{combined}"
    )
    assert "UnboundLocalError" not in combined
    assert "no batches" in combined
