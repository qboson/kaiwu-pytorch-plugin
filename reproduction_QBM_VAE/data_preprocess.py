import os
import pickle
import numpy as np
from tqdm import tqdm
import math

# ================= DATA CONFIGURATION =================
# Physics / Mass Spectrometry Constants
MIN_MZ = 50.0
MAX_MZ = 2500.0
BIN_SIZE = 0.1
# Dimension = (2500 - 50) / 0.1 + 1 ≈ 24501
VECTOR_DIM = int((MAX_MZ - MIN_MZ) / BIN_SIZE) + 1

# Vocabulary (Must match Inference Config)
AA_VOCAB = {
    '<PAD>': 0, '<SOS>': 1, '<EOS>': 2,
    'G': 3, 'A': 4, 'S': 5, 'P': 6, 'V': 7, 'T': 8, 'C': 9, 'L': 10,
    'I': 11, 'N': 12, 'D': 13, 'Q': 14, 'K': 15, 'E': 16, 'M': 17,
    'H': 18, 'F': 19, 'R': 20, 'Y': 21, 'W': 22,
    'M(ox)': 23, 'C(cam)': 24, 'N(deam)': 25, 'Q(deam)': 26
}

# IO Settings
CHUNK_SIZE = 20000  # Number of samples per shard
OUTPUT_DIR = "./processed_data_qbm_chunks"
RAW_DATA_PATH = "./data/raw_data.pkl"  # Point this to your source file


class SpectrumProcessor:
    """
    Handles the discretization and normalization of Mass Spectrometry data.
    """

    @staticmethod
    def bin_spectrum(mz_array, intensity_array):
        """
        Converts raw m/z and intensity arrays into a fixed-dimensional dense vector.
        """
        vector = np.zeros(VECTOR_DIM, dtype=np.float32)

        # Normalize intensity (Base Peak Normalization)
        if len(intensity_array) > 0:
            max_intensity = np.max(intensity_array)
            if max_intensity > 0:
                intensity_array = intensity_array / max_intensity

        # Vectorization / Binning
        for mz, inten in zip(mz_array, intensity_array):
            if mz < MIN_MZ or mz >= MAX_MZ:
                continue

            bin_idx = int((mz - MIN_MZ) / BIN_SIZE)
            if 0 <= bin_idx < VECTOR_DIM:
                # Merge peaks falling into the same bin (Max pooling strategy)
                vector[bin_idx] = max(vector[bin_idx], inten)

        return vector


class SequenceTokenizer:
    """
    Handles encoding of peptide sequences into integer tokens.
    """

    @staticmethod
    def tokenize(sequence):
        """
        Wraps sequence with <SOS> and <EOS> and maps to indices.
        Returns: List[int] or None if validation fails.
        """
        tokens = [AA_VOCAB['<SOS>']]

        # Simple parsing logic (can be extended for complex modifications)
        i = 0
        n = len(sequence)
        while i < n:
            # Check for modifications like M(ox)
            match = False
            for mod_len in [7, 6, 5]:  # Try matching longest keys first
                if i + mod_len <= n:
                    sub = sequence[i: i + mod_len]
                    if sub in AA_VOCAB:
                        tokens.append(AA_VOCAB[sub])
                        i += mod_len
                        match = True
                        break

            if not match:
                # Single amino acid
                aa = sequence[i]
                if aa in AA_VOCAB:
                    tokens.append(AA_VOCAB[aa])
                else:
                    # Unknown AA strategy: Skip or map to <UNK> (here we skip)
                    pass
                i += 1

        tokens.append(AA_VOCAB['<EOS>'])
        return tokens


def save_chunk(data, split, part_idx):
    """Serializes a data shard to disk."""
    if not data:
        return

    filename = os.path.join(OUTPUT_DIR, f"{split}_part_{part_idx}.pkl")
    try:
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
        print(f"[IO] Saved shard: {filename} ({len(data)} samples)")
    except IOError as e:
        print(f"[ERROR] Failed to save shard {filename}: {e}")


def process_pipeline(raw_source):
    """
    Main ETL pipeline: Load -> Transform -> Shard -> Save.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Check raw data existence
    if not os.path.exists(raw_source):
        # Fallback for demonstration if user hasn't configured raw path
        print(f"[WARN] Raw data not found at {raw_source}. Generating synthetic dummy data for verification.")
        # GENERATE DUMMY DATA (Remove this block in production)
        dummy_data = []
        for _ in range(50000):
            mz = np.random.uniform(100, 2000, 50)
            inten = np.random.uniform(0, 1, 50)
            seq = "PEPTIDESEQUENCE"
            dummy_data.append({'m/z array': mz, 'intensity array': inten, 'sequence': seq})
        raw_iterator = dummy_data
    else:
        print(f"[PROC] Loading raw data from {raw_source}...")
        with open(raw_source, 'rb') as f:
            raw_iterator = pickle.load(f)

    print("[PROC] Starting vectorization and tokenization...")

    train_buffer = []
    test_buffer = []

    # Split ratio configuration
    test_ratio = 0.1

    processed_count = 0
    train_chunk_idx = 0
    test_chunk_idx = 0

    for item in tqdm(raw_iterator, desc="Processing"):
        try:
            # Extract fields (Adjust keys based on your raw data schema)
            mz = item.get('m/z array')
            inten = item.get('intensity array')
            seq = item.get('sequence')

            if mz is None or seq is None:
                continue

            # 1. Process Spectrum
            x_vec = SpectrumProcessor.bin_spectrum(mz, inten)

            # 2. Process Sequence
            y_indices = SequenceTokenizer.tokenize(seq)

            # Validation
            if np.sum(x_vec) == 0 or len(y_indices) < 3:
                continue

            sample = {'x': x_vec, 'y': y_indices}

            # Train/Test Split
            if np.random.rand() < test_ratio:
                test_buffer.append(sample)
                if len(test_buffer) >= CHUNK_SIZE:
                    save_chunk(test_buffer, 'test', test_chunk_idx)
                    test_buffer = []
                    test_chunk_idx += 1
            else:
                train_buffer.append(sample)
                if len(train_buffer) >= CHUNK_SIZE:
                    save_chunk(train_buffer, 'train', train_chunk_idx)
                    train_buffer = []
                    train_chunk_idx += 1

            processed_count += 1

        except Exception as e:
            # Fail silently on individual bad samples to keep pipeline running
            continue

    # Flush remaining buffers
    save_chunk(train_buffer, 'train', train_chunk_idx)
    save_chunk(test_buffer, 'test', test_chunk_idx)

    print(f"[DONE] ETL Pipeline complete. Processed {processed_count} valid samples.")


if __name__ == "__main__":
    # Ensure random seed for reproducibility during split
    np.random.seed(42)
    process_pipeline(RAW_DATA_PATH)