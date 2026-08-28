"""Smoke test runner for QBM + EBRM integration.

This script imports the QBM adapter and runs a single forward
on random embeddings and rewards to verify the module loads.
"""
import os
import sys
import torch

# ensure the repo `src/` is on PYTHONPATH so `kaiwu` package resolves
ROOT_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
sys.path.insert(0, ROOT_SRC)

os.environ.setdefault('USE_QBM', '1')

def main():
    try:
        from kaiwu.torch_plugin.qbm_adapter import QBMModel
    except Exception as e:
        print('QBMModel import failed:', e)
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = QBMModel(embedding_size=512, num_nodes=64, num_visible=16, device=device)
    model.to(device)
    model.eval()

    # small random batch
    B = 4
    embedding = torch.randn(B, 512, device=device)
    reward = torch.randn(B, device=device)

    with torch.no_grad():
        out = model(embedding, reward)

    print('QBMModel forward output shape:', out.shape)
    print('Sample outputs:', out.cpu().numpy())

if __name__ == '__main__':
    main()
