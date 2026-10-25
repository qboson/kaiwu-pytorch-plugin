"""Progress-visualisation composition contract for the DBN trainer."""

import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "example", "dbn_digits"))

from dbn_trainer import DBNTrainer  # noqa: E402


def test_all_three_sub_visualisations_run_in_order():
    """Samples, weights and reconstruction panels are produced per batch."""
    events = []
    trainer = DBNTrainer.__new__(DBNTrainer)
    trainer._visualize_generated_samples = lambda rbm, i, e: events.append(("samples", i, e))
    trainer._visualize_weights_gradients = lambda rbm, i, e: events.append(("weights", i, e))
    trainer._visualize_current_reconstruction = lambda rbm, batch, i, e: events.append(
        ("recon", i, e)
    )
    trainer._visualize_training_progress("rbm", batch_idx=1, epoch=2, current_batch="batch")
    assert events == [("samples", 1, 2), ("weights", 1, 2), ("recon", 1, 2)]
