# -*- coding: utf-8 -*-
# Copyright (C) 2022-2026 Beijing QBoson Quantum Technology Co., Ltd.
#
# SPDX-License-Identifier: Apache-2.0
"""CLI entry point for quantum-inspired financial portfolio optimization."""

import importlib
import sys
from pathlib import Path
from typing import List, Optional


def main(argv: Optional[List[str]] = None) -> None:
    """Launch the financial portfolio optimization CLI from any directory."""
    parent_dir = str(Path(__file__).resolve().parent.parent)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    benchmark_mod = importlib.import_module("financial_portfolio.benchmark")
    benchmark_mod.main(argv)


if __name__ == "__main__":
    main()
