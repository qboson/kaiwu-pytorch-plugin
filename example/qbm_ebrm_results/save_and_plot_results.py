import os
from pathlib import Path
import sys
import torch
import numpy as np

repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))

from EBRM.src.reward_modeling.ebm_training import ebm_nce_plus as ebm


def build_model(device):
    use_qbm = os.environ.get('USE_QBM', '1').lower() in ('1', 'true', 'yes')
    if use_qbm:
        try:
            from kaiwu.torch_plugin.qbm_adapter import QBMModel
            model = QBMModel(embedding_size=512, num_nodes=64, num_visible=16, device=device)
            print('Using QBMModel')
        except Exception as e:
            print('QBM import failed, falling back to EBM_DNN', e)
            model = ebm.EBM_DNN(embedding_size=512)
    else:
        model = ebm.EBM_DNN(embedding_size=512)
    return model.to(device)


def main():
    out_dir = Path(__file__).parent
    data_path = out_dir / 'rmb_dataset.pt'
    if not data_path.exists():
        print('Dataset not found:', data_path)
        return

    dataset = ebm.RewardEmbeddingDataset(str(data_path))
    val_dataset = ebm.RewardEmbeddingDataset(str(data_path))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model(device)

    batch_size = 32
    epochs = 5
    lr = 1e-5
    M = 64
    std_devs = [0.5]
    lambda_reg = 0.1

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    losses = []
    val_accs = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for embeddings, rewards in dataloader:
            embeddings = embeddings.float().to(device)
            rewards = rewards.float().to(device)
            optimizer.zero_grad()
            loss = ebm.batch_loss_function(model, embeddings, rewards, std_devs, beta=0.1, M=M, l_reg=lambda_reg)
            if torch.isnan(loss) or torch.isinf(loss):
                print('Encountered NaN/Inf loss, aborting epoch')
                break
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / max(1, len(dataloader))
        losses.append(avg_loss)
        acc = ebm.validation(model, val_loader)
        val_accs.append(acc)
        print(f'Epoch {epoch+1}/{epochs} loss={avg_loss:.4f} val_acc={acc}')

    # save model and metrics
    model_file = out_dir / 'model_final.pth'
    torch.save(model.state_dict(), model_file)
    print('Saved model to', model_file)

    np.savez(out_dir / 'training_metrics.npz', losses=np.array(losses), val_accs=np.array(val_accs))

    # try to plot if matplotlib available
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        ax[0].plot(losses, marker='o')
        ax[0].set_title('Train Loss')
        ax[0].set_xlabel('Epoch')
        ax[0].set_ylabel('Loss')
        ax[1].plot(val_accs, marker='o')
        ax[1].set_title('Validation Accuracy')
        ax[1].set_xlabel('Epoch')
        ax[1].set_ylabel('Accuracy')
        plt.tight_layout()
        fig_path = out_dir / 'training_plots5.png'
        fig.savefig(fig_path)
        print('Saved plot to', fig_path)
    except Exception as e:
        print('matplotlib not available or failed to plot:', e)


if __name__ == '__main__':
    main()
