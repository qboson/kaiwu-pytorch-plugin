from __future__ import annotations

import argparse
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch

from mts_tokenizer import MTSTokenizer


@dataclass
class MTSDatasetTensors:
    x: torch.Tensor  # (N, 1540) float32 0/1
    sequences: List[str]

    @property
    def mean_x(self) -> np.ndarray:
        # mean in data space, used by QVAE to build train_bias
        return self.x.mean(dim=0).cpu().numpy()


def _load_pickle(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def load_mts_pickles(data_root: Path) -> Tuple[MTSDatasetTensors, MTSDatasetTensors]:
    """Load the exact train/valid split used by Zhao-Group/MTS-VAE.

    Expected files (relative to data_root):
    - MTS/data/tv_sim_split_train.pkl
    - MTS/data/tv_sim_split_valid.pkl

    The pickles are pandas DataFrames with a `sequence` column.
    """
    # The Zenodo `Datasets.zip` currently extracts to `Datasets/MTS/...`.
    # Allow either layout:
    # - <data_root>/MTS/data/...
    # - <data_root>/Datasets/MTS/data/...
    candidates = [
        data_root,
        data_root / "Datasets",
        data_root / "Datasets" / "MTS",  # extra safety
    ]

    train_pkl = None
    valid_pkl = None
    for root in candidates:
        tp = root / "MTS" / "data" / "tv_sim_split_train.pkl"
        vp = root / "MTS" / "data" / "tv_sim_split_valid.pkl"
        if tp.exists() and vp.exists():
            train_pkl = tp
            valid_pkl = vp
            break

    if train_pkl is None or valid_pkl is None:
        # last resort: search a bit deeper
        found_train = list(data_root.glob("**/MTS/data/tv_sim_split_train.pkl"))
        found_valid = list(data_root.glob("**/MTS/data/tv_sim_split_valid.pkl"))
        if found_train and found_valid:
            train_pkl = found_train[0]
            valid_pkl = found_valid[0]

    if train_pkl is None or valid_pkl is None or (not train_pkl.exists()) or (not valid_pkl.exists()):
        raise FileNotFoundError(
            "Missing expected dataset pickles. "
            f"Tried under {data_root} (and common nested layouts). "
            "Did you run download_datasets.py and point --data-root to the extracted folder?"
        )

    train_obj = _load_pickle(train_pkl)
    valid_obj = _load_pickle(valid_pkl)

    # We avoid importing pandas at module import time; but the unpickling itself
    # requires pandas installed if the pickles contain DataFrames.
    try:
        train_seqs = [str(s) for s in train_obj["sequence"].tolist()]
        valid_seqs = [str(s) for s in valid_obj["sequence"].tolist()]
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Failed to read `sequence` column from unpickled objects. "
            "These pickles are expected to be pandas DataFrames. "
            "Install pandas (and its dependencies) and retry."
        ) from e

    tok = MTSTokenizer()
    x_train = tok.batch_one_hot(train_seqs).reshape(len(train_seqs), -1)
    x_valid = tok.batch_one_hot(valid_seqs).reshape(len(valid_seqs), -1)

    return (
        MTSDatasetTensors(x=torch.from_numpy(x_train), sequences=train_seqs),
        MTSDatasetTensors(x=torch.from_numpy(x_valid), sequences=valid_seqs),
    )


def make_loader(x: torch.Tensor, batch_size: int, shuffle: bool) -> torch.utils.data.DataLoader:
    ds = torch.utils.data.TensorDataset(x)
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def main() -> int:
    parser = argparse.ArgumentParser(description="Quick sanity check for MTS pickle loading")
    parser.add_argument("--data-root", type=Path, required=True)
    args = parser.parse_args()

    train, valid = load_mts_pickles(args.data_root)
    print("train", train.x.shape, "valid", valid.x.shape)
    print("mean_x", train.mean_x.shape, float(train.mean_x.min()), float(train.mean_x.max()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
