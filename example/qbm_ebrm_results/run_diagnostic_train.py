import os
import torch
import traceback
from pathlib import Path

repo_root = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(repo_root))

# provide a lightweight wandb shim if wandb is not installed
try:
    import wandb
except Exception:
    class _DummyWandb:
        def init(self, *args, **kwargs):
            return None
        def log(self, *args, **kwargs):
            return None
    wandb = _DummyWandb()
    sys.modules['wandb'] = wandb

# provide a minimal transformers.trainer_utils shim if transformers is missing
try:
    import transformers
except Exception:
    import types
    transformers = types.SimpleNamespace()
    trainer_utils = types.SimpleNamespace(EvalPrediction=object)
    transformers.trainer_utils = trainer_utils
    sys.modules['transformers'] = transformers

from EBRM.src.reward_modeling.ebm_training import ebm_nce_plus as ebm


def main():
    data_path = Path(__file__).parent / 'rmb_dataset2.pt'
    val_path = data_path  # reuse as val for diagnostic
    if not data_path.exists():
        print('Dataset not found:', data_path)
        return

    dataset = ebm.RewardEmbeddingDataset(str(data_path))
    val_dataset = ebm.RewardEmbeddingDataset(str(val_path))

    # build model (USE_QBM env honored inside)
    use_qbm = os.environ.get('USE_QBM', '1')
    if use_qbm.lower() in ('1','true','yes'):
        try:
            from kaiwu.torch_plugin.qbm_adapter import QBMModel
            model = QBMModel(embedding_size=512, num_nodes=64, num_visible=16, device=torch.device('cpu'))
            print('Using QBMModel')
        except Exception as e:
            print('QBM import failed, falling back to EBM_DNN', e)
            model = ebm.EBM_DNN(embedding_size=512)
    else:
        model = ebm.EBM_DNN(embedding_size=512)

    model.to(torch.device('cpu'))

    # diagnostics: smaller M, lower lr, stronger regularization
    try:
        torch.autograd.set_detect_anomaly(True)
        acc = ebm.train_ebm(model, dataset, val_dataset, beta=0.1, batch_size=32, epochs=3, learning_rate=1e-5, weight_decay=1e-4, M=64, std_devs=[0.5], lambda_reg=0.1, letter='diag')
        print('Training finished. val acc:', acc)
    except Exception as e:
        print('Training raised exception:')
        traceback.print_exc()


if __name__ == '__main__':
    main()
