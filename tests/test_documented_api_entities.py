# -*- coding: utf-8 -*-
"""The documented KPP layer entity list must use real public class names."""

from __future__ import annotations

import re
from pathlib import Path

import kaiwu.torch_plugin as kpp
from kaiwu.torch_plugin.gbrbm import GaussianBernoulliRestrictedBoltzmannMachine

DOC = Path(__file__).resolve().parents[1] / "docs/source/getting_started/introduction.md"


def test_kpp_layer_uses_actual_public_class_names() -> None:
    line = next(
        line
        for line in DOC.read_text(encoding="utf-8").splitlines()
        if "AbstractBoltzmannMachine" in line and "QDiffusion" in line
    )
    names = re.findall(r"`([A-Za-z_][A-Za-z0-9_]*)`", line)

    assert "UnsupervisedDBN" in names
    assert "GaussianBernoulliRestrictedBoltzmannMachine" in names
    assert "DBN" not in names
    assert "GBRBM" not in names

    assert hasattr(kpp, "UnsupervisedDBN")
    assert issubclass(GaussianBernoulliRestrictedBoltzmannMachine, object)
