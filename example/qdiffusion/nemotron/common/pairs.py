"""Schema, validation, and persistence for same-state outcome pairs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import torch

from .runtime import atomic_torch_save

PAIR_SCHEMA_VERSION = 2
VALID_SPLITS = {"train", "val"}


@dataclass(frozen=True)
class OutcomePair:
    """A correct/wrong candidate pair captured at one identical proposal state.

    Attributes:
        problem_id: Identifier of the source problem.
        split: Dataset split the problem belongs to, ``train`` or ``val``.
        block_index: Diffusion block the state was captured in.
        step_index: Decode step within the block.
        state_hash: SHA-256 hash identifying the shared proposal state.
        noisy_tokens: Noisy token ids of the captured state.
        hidden_states: Frozen proposal hidden states of the captured state.
        noisy_features: Token features of the noisy block.
        positive_tokens: Candidate tokens judged correct.
        negative_tokens: Candidate tokens judged wrong.
        positive_candidate_features: Token features of ``positive_tokens``.
        negative_candidate_features: Token features of ``negative_tokens``.
        transfer_mask: Boolean mask of transfer positions in the candidates.
        positive_logprob: Proposal log-probability of the positive candidate.
        negative_logprob: Proposal log-probability of the negative candidate.
        negative_kind: Strategy used to pick the negative candidate.
    """

    problem_id: str
    split: str
    block_index: int
    step_index: int
    state_hash: str
    noisy_tokens: torch.Tensor
    hidden_states: torch.Tensor
    noisy_features: torch.Tensor
    positive_tokens: torch.Tensor
    negative_tokens: torch.Tensor
    positive_candidate_features: torch.Tensor
    negative_candidate_features: torch.Tensor
    transfer_mask: torch.Tensor
    positive_logprob: float
    negative_logprob: float
    negative_kind: str = "rollout_hard"

    def to_dict(self) -> dict[str, Any]:
        """Returns the dictionary form persisted in pair artifacts.

        Returns:
            Dict holding the schema version plus every field of this pair.
        """
        return {"schema_version": PAIR_SCHEMA_VERSION, **asdict(self)}


REQUIRED_FIELDS = {
    "schema_version",
    "problem_id",
    "split",
    "block_index",
    "step_index",
    "state_hash",
    "noisy_tokens",
    "hidden_states",
    "noisy_features",
    "positive_tokens",
    "negative_tokens",
    "positive_candidate_features",
    "negative_candidate_features",
    "transfer_mask",
    "positive_logprob",
    "negative_logprob",
    "negative_kind",
}


def validate_pair(record: dict[str, Any], *, index: int | None = None) -> None:
    """Validates one pair record against the pair schema.

    Args:
        record: Candidate pair record loaded from an artifact.
        index: Optional artifact position used in error messages.

    Raises:
        ValueError: If required fields, the schema version, the split, the
            state hash, or tensor shapes/dtypes are invalid.
        TypeError: If tensor fields are not ``torch.Tensor`` values.
    """
    label = f"pair {index}" if index is not None else "pair"
    missing = sorted(REQUIRED_FIELDS - set(record))
    if missing:
        raise ValueError(f"{label} is missing fields: {missing}")
    if record["schema_version"] != PAIR_SCHEMA_VERSION:
        raise ValueError(f"{label} has unsupported schema_version")
    if record["split"] not in VALID_SPLITS:
        raise ValueError(f"{label} has invalid split: {record['split']}")
    state_hash = record["state_hash"]
    if (
        not isinstance(state_hash, str)
        or len(state_hash) != 64
        or any(character not in "0123456789abcdef" for character in state_hash)
    ):
        raise ValueError(f"{label} state_hash must be a lowercase SHA-256 digest")
    tensors = {
        key: record[key]
        for key in (
            "hidden_states",
            "noisy_tokens",
            "noisy_features",
            "positive_tokens",
            "negative_tokens",
            "positive_candidate_features",
            "negative_candidate_features",
            "transfer_mask",
        )
    }
    if not all(isinstance(value, torch.Tensor) for value in tensors.values()):
        raise TypeError(f"{label} tensor fields must be torch.Tensor values")
    sequence_length = tensors["positive_tokens"].numel()
    if tensors["positive_tokens"].ndim != 1:
        raise ValueError(f"{label} token tensors must be one-dimensional")
    if sequence_length == 0:
        raise ValueError(f"{label} token tensors must not be empty")
    if tensors["negative_tokens"].shape != tensors["positive_tokens"].shape:
        raise ValueError(f"{label} positive/negative token shapes differ")
    if tensors["noisy_tokens"].shape != tensors["positive_tokens"].shape:
        raise ValueError(f"{label} noisy token shape differs from candidates")
    if tensors["transfer_mask"].shape != tensors["positive_tokens"].shape:
        raise ValueError(f"{label} transfer_mask shape differs from tokens")
    if tensors["transfer_mask"].dtype != torch.bool:
        raise ValueError(f"{label} transfer_mask must be boolean")
    if not bool(tensors["transfer_mask"].any()):
        raise ValueError(f"{label} transfer_mask must select at least one token")
    for key in (
        "hidden_states",
        "noisy_features",
        "positive_candidate_features",
        "negative_candidate_features",
    ):
        if tensors[key].ndim != 2 or tensors[key].shape[0] != sequence_length:
            raise ValueError(f"{label} {key} must have shape [sequence, feature]")
    if (
        tensors["positive_candidate_features"].shape
        != tensors["negative_candidate_features"].shape
    ):
        raise ValueError(f"{label} positive/negative feature shapes differ")
    if tensors["noisy_features"].shape != tensors["positive_candidate_features"].shape:
        raise ValueError(f"{label} noisy/candidate feature shapes differ")
    changed = (
        tensors["positive_tokens"][tensors["transfer_mask"]]
        != tensors["negative_tokens"][tensors["transfer_mask"]]
    )
    if not bool(changed.any()):
        raise ValueError(f"{label} candidates are identical on the transfer mask")
    for key in ("positive_tokens", "negative_tokens"):
        if not bool(
            (
                tensors["noisy_tokens"][tensors["transfer_mask"]]
                != tensors[key][tensors["transfer_mask"]]
            ).all()
        ):
            raise ValueError(f"{label} {key} must differ from noisy_tokens on transfer")


def validate_problem_splits(records: Iterable[dict[str, Any]]) -> None:
    """Ensures no ``problem_id`` appears under two different splits.

    Args:
        records: Pair records to check.

    Raises:
        ValueError: If a problem leaks across splits.
    """
    assignments: dict[str, str] = {}
    for record in records:
        previous = assignments.setdefault(record["problem_id"], record["split"])
        if previous != record["split"]:
            raise ValueError(
                f"problem {record['problem_id']} leaks across {previous}/{record['split']}"
            )


def load_pairs(path: str | Path) -> list[dict[str, Any]]:
    """Loads and fully validates one pair artifact.

    Args:
        path: Path to a ``pairs.pt`` artifact.

    Returns:
        list[dict[str, Any]]: Validated pair records.

    Raises:
        ValueError: If the artifact is empty or fails pair validation.
        TypeError: If a record is not a dictionary.
    """
    records = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(records, list) or not records:
        raise ValueError("pair artifact must contain a non-empty list")
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise TypeError(f"pair {index} must be a dictionary")
        validate_pair(record, index=index)
    validate_problem_splits(records)
    return records


def save_pairs(path: str | Path, records: Iterable[dict[str, Any]]) -> None:
    """Validates pair records and atomically persists them.

    Args:
        path: Destination artifact path.
        records: Pair records to persist.

    Raises:
        ValueError: If ``records`` is empty or any record fails validation.
    """
    materialized = list(records)
    if not materialized:
        raise ValueError("cannot save an empty pair artifact")
    for index, record in enumerate(materialized):
        validate_pair(record, index=index)
    validate_problem_splits(materialized)
    atomic_torch_save(Path(path), materialized)
