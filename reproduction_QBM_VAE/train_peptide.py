import os
import torch
import torch.optim as optim
from tqdm import tqdm
import kaiwu
from dataset_loader import get_dataloader
from models import PeptideQVAE

# ================= CONFIGURATION =================
# SDK Credentials
USER_ID = "91850531256946690"
SDK_CODE = "lTj5v0u67gyWsMfXxKAbiJPkkT6w7u"

# Hyperparameters (Balanced Profile for CPU)
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
NUM_EPOCHS = 30
MAX_FILES = 1  # Shard limit for rapid iteration
HIDDEN_DIM = 512
LATENT_DIM = 64

# IO Paths
DATA_DIR = "./processed_data_qbm_chunks"
SAVE_DIR = "./models/Peptide_QVAE_Balanced"

# ================= INITIALIZATION =================
print("[INIT] Initializing Kaiwu Quantum SDK...")
try:
    kaiwu.license.init(user_id=USER_ID, sdk_code=SDK_CODE)
    print("[INFO] License verified successfully.")
except Exception as e:
    print(f"[WARN] License initialization failed: {e}")

os.makedirs(SAVE_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INIT] Runtime device: {device}")

# Pipeline Setup
print(f"[DATA] Loading dataset (Limit: {MAX_FILES} shards)...")
train_loader = get_dataloader(DATA_DIR, 'train', batch_size=BATCH_SIZE, shuffle=True, max_files=MAX_FILES)

print(f"[MODEL] Building architecture (H={HIDDEN_DIM}, L={LATENT_DIM})...")
model = PeptideQVAE(
    input_dim=24501,
    hidden_dim=HIDDEN_DIM,
    latent_dim=LATENT_DIM,
    kl_beta=0.001
).to(device)

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ================= TRAINING LOOP =================
print(f"[TRAIN] Starting training loop for {NUM_EPOCHS} epochs.")

for epoch in range(1, NUM_EPOCHS + 1):
    model.train()
    running_loss = 0.0
    batch_counter = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS}")

    for x, y in pbar:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        # Forward pass
        seq_logits, z, logits_z, energy = model(x, y)

        # Loss computation
        loss, ce, prior = model.compute_loss(seq_logits, y, logits_z, energy)

        # Backpropagation
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        batch_counter += 1

        pbar.set_postfix({
            'L_Total': f"{loss.item():.2f}",
            'L_Recon': f"{ce.item():.2f}"
        })

    epoch_avg_loss = running_loss / max(1, batch_counter)
    print(f"[LOG] Epoch {epoch} completed. Avg Loss: {epoch_avg_loss:.4f}")

    # Checkpointing
    ckpt_path = os.path.join(SAVE_DIR, f"qvae_balanced_epoch{epoch}.pth")
    torch.save(model.state_dict(), ckpt_path)
    print(f"[CKPT] Model state saved to {ckpt_path}")

print("[DONE] Training pipeline finished successfully.")