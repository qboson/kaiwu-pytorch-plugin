import torch
import os
import numpy as np
from models import PeptideQVAE
from dataset_loader import get_dataloader

# ================= INFERENCE CONFIG =================
# Must match training configuration
INPUT_DIM = 24501
LATENT_DIM = 64
HIDDEN_DIM = 512
VOCAB_SIZE = 30
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IO_CONFIG = {
    'data_dir': "./processed_data_qbm_chunks",
    'model_dir': "./models/Peptide_QVAE_Balanced",
    'dataset_split': 'train',  # Set 'test' or 'train' for evaluation
    'batch_size': 5
}

# Vocabulary Mapping
AA_VOCAB = {
    '<PAD>': 0, '<SOS>': 1, '<EOS>': 2,
    'G': 3, 'A': 4, 'S': 5, 'P': 6, 'V': 7, 'T': 8, 'C': 9, 'L': 10,
    'I': 11, 'N': 12, 'D': 13, 'Q': 14, 'K': 15, 'E': 16, 'M': 17,
    'H': 18, 'F': 19, 'R': 20, 'Y': 21, 'W': 22,
    'M(ox)': 23, 'C(cam)': 24, 'N(deam)': 25, 'Q(deam)': 26
}
IDX_TO_AA = {v: k for k, v in AA_VOCAB.items()}


def resolve_latest_model(model_dir):
    """Utility to fetch the latest checkpoint based on epoch numbering."""
    try:
        models = sorted([f for f in os.listdir(model_dir) if f.endswith('.pth')],
                        key=lambda x: int(x.split('epoch')[1].split('.')[0]))
        return os.path.join(model_dir, models[-1])
    except Exception as e:
        print(f"[ERROR] Could not resolve model from {model_dir}: {e}")
        return None


def decode_sequence(indices):
    """Decodes tensor indices to peptide string, handling special tokens."""
    seq = []
    for idx in indices:
        idx = idx.item()
        if idx == 2: break  # EOS token
        if idx not in [0, 1]:  # Skip PAD, SOS
            seq.append(IDX_TO_AA.get(idx, '?'))
    return ''.join(seq)


def generate_sequence_greedy(model, x, max_len=50):
    """
    Performs greedy decoding strategy for sequence generation.
    """
    model.eval()
    with torch.no_grad():
        # Encode
        logits = model.encoder(x)
        z = (torch.sigmoid(logits) > 0.5).float()  # Hard thresholding

        # Decode
        hidden = model.decoder.latent_to_hidden(z).unsqueeze(0)
        batch_size = x.size(0)
        curr_input = torch.tensor([[1]] * batch_size, device=DEVICE)  # <SOS> token

        generated_seqs = [[] for _ in range(batch_size)]
        finished = [False] * batch_size

        for _ in range(max_len):
            embedded = model.decoder.embedding(curr_input)
            output, hidden = model.decoder.gru(embedded, hidden)
            pred_token = model.decoder.fc_out(output).argmax(dim=-1)
            curr_input = pred_token

            for i in range(batch_size):
                token = pred_token[i].item()
                if token == 2: finished[i] = True
                if not finished[i] and token != 1:
                    generated_seqs[i].append(token)
            if all(finished): break

    return generated_seqs


if __name__ == "__main__":
    latest_ckpt = resolve_latest_model(IO_CONFIG['model_dir'])
    if not latest_ckpt: exit()

    print(f"[INFO] Loading checkpoint: {latest_ckpt}")

    # Data Loader
    test_loader = get_dataloader(
        IO_CONFIG['data_dir'],
        IO_CONFIG['dataset_split'],
        batch_size=IO_CONFIG['batch_size'],
        max_files=1
    )

    # Model Setup
    model = PeptideQVAE(INPUT_DIM, HIDDEN_DIM, LATENT_DIM, VOCAB_SIZE).to(DEVICE)
    model.load_state_dict(torch.load(latest_ckpt, map_location=DEVICE))
    model.eval()

    print("\n[EVAL] Inference Sample Comparison")
    print("-" * 110)
    print(f"{'Ground Truth Sequence':<40} | {'Predicted Sequence':<40} | {'Match Status'}")
    print("-" * 110)

    sample_limit = 10
    processed_count = 0

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(DEVICE)
            pred_indices = generate_sequence_greedy(model, x)

            for i in range(len(y)):
                true_str = decode_sequence(y[i])
                pred_str = decode_sequence(torch.tensor(pred_indices[i]))

                status = "[MATCH]" if true_str == pred_str else "[DIFF]"
                if len(pred_str) == 0: status = "[EMPTY]"

                print(f"{true_str:<40} | {pred_str:<40} | {status}")

                processed_count += 1
                if processed_count >= sample_limit: break
            if processed_count >= sample_limit: break