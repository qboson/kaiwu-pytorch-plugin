"""Regression test for ``helpers.plot_calo_images`` terminating the process.

With ``do_gif=False`` the helper saved its figure and then called
``sys.exit()``, so any caller lost its interpreter silently (exit code 0, no
exception, no traceback).  A plotting helper must return control to its caller.
"""

import os
import subprocess
import sys
import textwrap

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
HELPERS_DIR = os.path.join(REPO_ROOT, "example", "qvae_mnist")

CHILD_SCRIPT = textwrap.dedent(
    """
    import sys
    import matplotlib
    matplotlib.use("Agg")
    import torch

    sys.path.insert(0, {helpers_dir!r})
    sys.path.insert(0, {src_dir!r})
    from utils.helpers import plot_calo_images

    x_true = torch.rand(2, 4, 4)
    x_recon = torch.rand(2, 16)
    plot_calo_images(
        x_true, x_recon, n_samples=2, output="/tmp/calo_helper_test.png", do_gif=False
    )
    print("AFTER CALL: still running")
    """
).format(helpers_dir=HELPERS_DIR, src_dir=os.path.join(REPO_ROOT, "src"))


def test_plot_calo_images_returns_to_the_caller():
    result = subprocess.run(
        [sys.executable, "-c", CHILD_SCRIPT],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert "AFTER CALL" in result.stdout, (
        "plot_calo_images must return instead of calling sys.exit(); "
        f"returncode={result.returncode}, output:\n{result.stdout}{result.stderr}"
    )
    assert result.returncode == 0
