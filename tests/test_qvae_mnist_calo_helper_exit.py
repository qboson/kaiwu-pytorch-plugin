# -*- coding: utf-8 -*-
# Copyright (C) 2022-2026 Beijing QBoson Quantum Technology Co., Ltd.
#
# SPDX-License-Identifier: Apache-2.0
"""``plot_calo_images`` must return control to its caller.

The helper used to call ``sys.exit()`` after saving the figure when
``do_gif=False`` (the default), which silently terminated the interpreter with
status 0. A subprocess is used so a regression cannot kill the test run; the
figure helper is loaded by file path to avoid shadowing the standard library
``logging`` module with ``utils/logging.py``.
"""

import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
HELPERS = REPO_ROOT / "example" / "qvae_mnist" / "utils" / "helpers.py"

REQUIRED_MODULES = (
    "matplotlib",
    "torch",
    "torchvision",
    "torchmetrics",
    "imageio",
    "sklearn",
    "PIL",
)

CHILD_SCRIPT = textwrap.dedent(
    """
    import importlib.util
    import sys
    import types

    gif = types.ModuleType("gif")


    def _frame(*args, **kwargs):
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        return lambda fn: fn


    gif.frame = _frame
    sys.modules["gif"] = gif

    spec = importlib.util.spec_from_file_location("kpp_qvae_helpers", r"__HELPERS__")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    import torch

    x_true = torch.rand(3, 16, 16) + 0.1
    x_recon = torch.rand(3, 256) + 0.1
    module.plot_calo_images(
        x_true,
        x_recon,
        n_samples=2,
        output=r"__OUTPUT__",
        do_gif=False,
    )
    print("AFTER_CALL")
    """
)


def test_plot_calo_images_returns_to_caller(tmp_path):
    """The call must not terminate the interpreter."""
    for module in REQUIRED_MODULES:
        pytest.importorskip(module)

    output = tmp_path / "calo.png"
    script = (
        CHILD_SCRIPT.replace("__HELPERS__", str(HELPERS))
        .replace("__OUTPUT__", str(output))
    )
    env = dict(os.environ)
    env["MPLBACKEND"] = "Agg"

    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env=env,
        timeout=600,
    )

    assert result.returncode == 0, result.stderr
    assert "AFTER_CALL" in result.stdout, (
        "plot_calo_images terminated the interpreter before returning"
    )
    assert output.is_file()
