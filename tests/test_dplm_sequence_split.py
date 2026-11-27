"""Keep repeated protein sequences together in the DPLM workflow."""

from collections import Counter

import pytest

pytest.importorskip("omegaconf")
pytest.importorskip("transformers")

from example.qdiffusion.dplm.workflows.workflow_helpers import split_train_val_test


def test_duplicate_sequences_do_not_cross_split_boundaries():
    sequences = ["AAA", "CCC", "DDD", "E E E", "aaa", "AAA", "AAA", "AAA", "AAA", "EEE"]
    records = [(f"record-{index}", sequence) for index, sequence in enumerate(sequences)]
    kwargs = {"val_ratio": 0.25, "test_ratio": 0.25, "seed": 42}

    splits = split_train_val_test(records, **kwargs)

    assert splits == split_train_val_test(records, **kwargs)
    assert Counter(record for split in splits for record in split) == Counter(records)
    groups = [{sequence.replace(" ", "").upper() for _, sequence in split} for split in splits]
    assert [len(group) for group in groups] == [2, 1, 1]
    assert len(set.union(*groups)) == sum(map(len, groups))


def test_split_requires_three_distinct_sequences():
    records = [("a", "AAA"), ("a again", "aaa"), ("b", "CCC"), ("b again", "CCC")]
    with pytest.raises(ValueError, match="at least 3 distinct sequences"):
        split_train_val_test(records, val_ratio=0.2, test_ratio=0.2, seed=0)


def test_preserves_seeded_split_for_distinct_sequences():
    records = [(sequence, sequence) for sequence in ["AAA", "CCC", "DDD", "EEE", "FFF"]]
    train, val, test = split_train_val_test(records, val_ratio=0.2, test_ratio=0.2, seed=42)
    assert (train, val, test) == ([records[2], records[4], records[0]], [records[3]], [records[1]])
