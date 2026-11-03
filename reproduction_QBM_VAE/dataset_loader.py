import os
import glob
import pickle
import numpy as np
import torch
from torch.utils.data import IterableDataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


class ChunkedDataset(IterableDataset):
    """
    Implements an IterableDataset for efficient loading of sharded .pkl data files.
    Designed to handle large-scale spectral data with limited memory footprint.
    """

    def __init__(self, data_dir, split_name, shuffle=False, max_files=None):
        super(ChunkedDataset, self).__init__()
        self.data_dir = data_dir
        self.split_name = split_name
        self.shuffle = shuffle

        # Locate all data shards
        pattern = os.path.join(data_dir, f"{split_name}_part_*.pkl")
        full_file_list = sorted(glob.glob(pattern))

        if not full_file_list:
            print(f"[WARN] No data shards found in {data_dir} for split '{split_name}'. Check preprocessing.")
            self.file_list = []
        else:
            # File selection strategy
            if max_files is not None and max_files > 0:
                self.file_list = full_file_list[:max_files]
                print(
                    f"[INFO] Fast-mode active. Loading {len(self.file_list)}/{len(full_file_list)} shards for '{split_name}'.")
            else:
                self.file_list = full_file_list
                print(f"[INFO] Full-mode active. Loading all {len(self.file_list)} shards for '{split_name}'.")

    def __iter__(self):
        """Yields batches of (spectrum, sequence) pairs from disk."""
        current_list = list(self.file_list)
        if self.shuffle:
            np.random.shuffle(current_list)

        for file_path in current_list:
            try:
                with open(file_path, 'rb') as f:
                    data_chunk = pickle.load(f)

                # In-memory shuffle for the current chunk
                if self.shuffle:
                    np.random.shuffle(data_chunk)

                for item in data_chunk:
                    # Feature extraction: Sparse matrix to dense tensor
                    if hasattr(item['x'], 'toarray'):
                        x_dense = item['x'].toarray().flatten().astype(np.float32)
                    else:
                        x_dense = item['x'].flatten().astype(np.float32)

                    x_tensor = torch.from_numpy(x_dense)
                    y_tensor = torch.tensor(item['y'], dtype=torch.long)

                    yield x_tensor, y_tensor

            except IOError as e:
                print(f"[ERROR] Failed to read shard {file_path}: {e}")
                continue


def collate_fn_pad(batch):
    """
    Custom collator to handle variable-length peptide sequences.
    Pads sequences with 0 (<PAD>) to the maximum length in the batch.
    """
    xs, ys = zip(*batch)
    xs_stacked = torch.stack(xs)
    ys_padded = pad_sequence(ys, batch_first=True, padding_value=0)
    return xs_stacked, ys_padded


def get_dataloader(data_dir, split_name, batch_size=32, shuffle=False, max_files=None):
    """Factory function to instantiate the DataLoader pipeline."""
    dataset = ChunkedDataset(data_dir, split_name, shuffle=shuffle, max_files=max_files)
    # pin_memory=False for CPU workloads to avoid overhead
    return DataLoader(dataset, batch_size=batch_size, pin_memory=False, collate_fn=collate_fn_pad)