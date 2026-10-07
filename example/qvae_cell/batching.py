"""Training batches that preserve observations without singleton tails."""

from torch.utils.data import BatchSampler


class SingletonSafeBatchSampler(BatchSampler):
    """Merge a final singleton into the preceding batch for BatchNorm training.

    The sampler preserves the underlying index order and visits every index
    once. Only the last batch may grow to ``batch_size + 1`` observations.
    """

    def __init__(self, sampler, batch_size):
        super().__init__(sampler, batch_size, drop_last=False)
        if batch_size < 2:
            raise ValueError("Batch normalization requires batch_size of at least 2")
        if len(sampler) < 2:
            raise ValueError("Batch normalization requires at least two training observations")

    def __iter__(self):
        pending = None
        for batch in super().__iter__():
            if pending is not None:
                if len(batch) == 1:
                    yield pending + batch
                    return
                yield pending
            pending = batch
        if pending is not None:
            yield pending

    def __len__(self):
        count = len(self.sampler)
        batches = super().__len__()
        if count > self.batch_size and count % self.batch_size == 1:
            return batches - 1
        return batches
