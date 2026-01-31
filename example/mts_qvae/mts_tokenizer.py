from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import numpy as np


@dataclass(frozen=True)
class MTSTokenizer:
    """Tokenizer/one-hot encoder matching the original MTS-VAE scripts.

    Alphabet and order are taken from Zhao-Group/MTS-VAE `MTS/scripts/train.py`:
    "FIWLVMYCATHGSQRKNEPD$0" (22 tokens)

    Notes:
    - `$` marks cleavage site (appended to each sequence)
    - `0` is padding (right-pad to length 70)
    """

    alphabet: str = "FIWLVMYCATHGSQRKNEPD$0"
    pad_char: str = "0"
    eos_char: str = "$"
    max_len: int = 70

    @property
    def vocab_size(self) -> int:
        return len(self.alphabet)

    @property
    def mapping(self) -> dict[str, int]:
        return {c: i for i, c in enumerate(self.alphabet)}

    @property
    def inv_mapping(self) -> dict[int, str]:
        return {i: c for i, c in enumerate(self.alphabet)}

    def normalize(self, seq: str) -> str:
        seq = seq.strip().upper()
        if not seq:
            raise ValueError("Empty sequence")
        # paper pipeline: append '$' then right-pad with '0' to length 70
        seq = seq + self.eos_char
        if len(seq) > self.max_len:
            raise ValueError(f"Sequence too long after adding eos: {len(seq)} > {self.max_len}")
        return seq.ljust(self.max_len, self.pad_char)

    def one_hot(self, seq: str) -> np.ndarray:
        seq = self.normalize(seq)
        mapping = self.mapping
        idx = [mapping[c] for c in seq]
        return np.eye(self.vocab_size, dtype=np.float32)[idx]

    def batch_one_hot(self, seqs: Iterable[str]) -> np.ndarray:
        mats = [self.one_hot(s) for s in seqs]
        return np.stack(mats, axis=0)

    def decode_argmax(self, probs_70xV: np.ndarray) -> str:
        """Decode (70, vocab) probabilities/logits by argmax, then strip padding/eos.

        Matches Zhao-Group/MTS-VAE `generate.py` logic.
        """
        if probs_70xV.shape != (self.max_len, self.vocab_size):
            raise ValueError(f"Expected {(self.max_len, self.vocab_size)}, got {probs_70xV.shape}")
        inv = self.inv_mapping
        tokens = [inv[int(i)] for i in probs_70xV.argmax(axis=-1)]
        out = "".join(tokens)
        out = out.rstrip(self.pad_char).rstrip(self.eos_char)
        return out

    def is_valid_generated(self, seq: str) -> bool:
        # the original script filters out sequences that contain '$' or '0' anywhere
        return (self.eos_char not in seq) and (self.pad_char not in seq) and (len(seq) > 0)
