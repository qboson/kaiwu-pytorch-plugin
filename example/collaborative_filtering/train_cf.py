# -*- coding: utf-8 -*-
# Copyright (C) 2022-2025 Beijing QBoson Quantum Technology Co., Ltd.
#
# SPDX-License-Identifier: Apache-2.0
"""CLI entry point for training and benchmarking RBM collaborative filtering."""

import importlib
import sys
from pathlib import Path
from typing import List, Optional


def main(argv: Optional[List[str]] = None) -> None:
    """Launch the collaborative filtering benchmark CLI from any directory."""
    parent_dir = str(Path(__file__).resolve().parent.parent)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    benchmark_mod = importlib.import_module("collaborative_filtering.benchmark")
    benchmark_mod.main(argv)


if __name__ == "__main__":
    main()
