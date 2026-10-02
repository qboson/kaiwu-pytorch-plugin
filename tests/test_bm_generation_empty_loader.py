"""Regression test for the bm_generation trainer hang on an empty data source.

``Trainer.train`` nests ``for batch_data in self.data`` inside
``while step < max_steps``.  When the data loader yields no batches the inner
loop body never runs, the step counter never advances, and ``train`` spins
forever.  The trainer must reject such a configuration instead of hanging.

The case is exercised in a subprocess so that a regression cannot wedge the
test session: the child is killed by the timeout if ``train`` fails to return.
"""

import os
import subprocess
import sys
import textwrap

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TRAINER_DIR = os.path.join(REPO_ROOT, "example", "bm_generation")

CHILD_SCRIPT = textwrap.dedent(
    """
    import sys
    import torch

    sys.path.insert(0, {trainer_dir!r})
    from trainer import Trainer

    class DummySaver:
        def save_info(self, *args, **kwargs):
            pass

        def output_loss(self, *args, **kwargs):
            pass

    class DummyWorker:
        pass

    trainer = Trainer(
        data=[],
        saver=DummySaver(),
        worker=DummyWorker(),
        num_visible=4,
        num_hidden=2,
        num_output=2,
    )
    try:
        trainer.train(max_steps=2, save_path="/tmp/bm_empty_loader.pth")
    except ValueError as exc:
        print("RAISED_VALUE_ERROR: {{}}".format(exc))
    except Exception as exc:  # pragma: no cover - diagnostic detail
        print("RAISED_OTHER: {{!r}}".format(exc))
    else:
        print("RETURNED_WITHOUT_ERROR")
    """
).format(trainer_dir=TRAINER_DIR)


def test_train_rejects_an_empty_data_source():
    result = subprocess.run(
        [sys.executable, "-c", CHILD_SCRIPT],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=180,
    )
    combined = result.stdout + result.stderr
    assert "RAISED_VALUE_ERROR" in combined, (
        "Trainer.train() must fail fast when the data source yields no batches; "
        f"child output:\n{combined}"
    )
    assert "RETURNED_WITHOUT_ERROR" not in combined
    assert "No batches" in combined or "no batches" in combined
