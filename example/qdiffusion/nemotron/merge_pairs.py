#!/usr/bin/env python3
"""Merge independently collected train/val pair artifacts safely."""

from __future__ import annotations

import argparse
from pathlib import Path

from .common.pairs import load_pairs, save_pairs


def main() -> None:
    """Merges independently collected pair artifacts into one."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    records = [record for path in args.inputs for record in load_pairs(path)]
    save_pairs(args.output, records)
    print(f"saved {len(records)} validated pairs to {args.output}")


if __name__ == "__main__":
    main()
