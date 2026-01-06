"""Prepare dataset from RMB JSON and run a short QBM-EBRM training smoke run."""
import os
import sys

# ensure repo root and src in path
HERE = os.path.dirname(__file__)
REPO_ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
SRC_ROOT = os.path.join(REPO_ROOT, 'src')
sys.path.insert(0, SRC_ROOT)
sys.path.insert(0, REPO_ROOT)

os.environ.setdefault('USE_QBM', '1')
os.environ.setdefault('WANDB_MODE', 'offline')

import torch
# ensure wandb is available (create a dummy if missing) so imports succeed in minimal env
try:
    import wandb
except Exception:
    import types
    wandb = types.SimpleNamespace(init=lambda *a, **k: None, log=lambda *a, **k: None)
    import sys as _sys
    _sys.modules['wandb'] = wandb

# provide minimal dummy for transformers.trainer_utils.EvalPrediction if transformers not installed
try:
    import transformers
except Exception:
    import types as _types
    import sys as _sys
    transformers_mod = _types.SimpleNamespace()
    trainer_utils = _types.SimpleNamespace(EvalPrediction=object)
    transformers_mod.trainer_utils = trainer_utils
    _sys.modules['transformers'] = transformers_mod
    _sys.modules['transformers.trainer_utils'] = trainer_utils

# provide a minimal scipy.stats stub if scipy is not installed
try:
    import scipy
except Exception:
    import types as _types
    import sys as _sys
    stats_mod = _types.SimpleNamespace()
    _sys.modules['scipy'] = _types.SimpleNamespace(stats=stats_mod)
    _sys.modules['scipy.stats'] = stats_mod

from EBRM.src.reward_modeling.ebm_training import ebm_nce_plus as ebm
from prepare_rmb_dataset import prepare_from_json


def split_dataset(pt_path, train_pt, val_pt, val_frac=0.2):
    data = torch.load(pt_path)
    n = len(data)
    cut = int(n * (1 - val_frac))
    torch.save(data[:cut], train_pt)
    torch.save(data[cut:], val_pt)
    return train_pt, val_pt


def main(json_path=None):
    here = os.path.dirname(__file__)
    pt_out = os.path.join(here, 'rmb_dataset.pt')
    if json_path is None:
        json_path = os.path.join(os.path.abspath(os.path.join(here, '..', '..')), 'RMB-Reward-Model-Benchmark', 'RMB_dataset', 'BoN_set', 'Harmlessness', 'S1.json')
    prepare_from_json(json_path, pt_out)
    train_pt = os.path.join(here, 'rmb_dataset_train.pt')
    val_pt = os.path.join(here, 'rmb_dataset_val.pt')
    split_dataset(pt_out, train_pt, val_pt, val_frac=0.2)

    # load datasets using RewardEmbeddingDataset
    train_ds = ebm.RewardEmbeddingDataset(train_pt)
    val_ds = ebm.RewardEmbeddingDataset(val_pt)

    # create model via USE_QBM env (ebm_nce_plus will read USE_QBM if used as __main__, but we instantiate here)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_qbm = os.environ.get('USE_QBM', '1').lower() in ('1', 'true', 'yes')
    if use_qbm:
        try:
            from kaiwu.torch_plugin.qbm_adapter import QBMModel
            model = QBMModel(embedding_size=512, num_nodes=64, num_visible=16, device=device)
            print('Using QBMModel')
        except Exception as e:
            print('QBM import failed, falling back to EBM_DNN:', e)
            model = ebm.EBM_DNN(embedding_size=512)
    else:
        model = ebm.EBM_DNN(embedding_size=512)

    model.to(device)

    # ensure ebm.module-level `model` exists for functions that reference it
    ebm.model = model

    # run short training
    acc = ebm.train_ebm(model, train_ds, val_ds, beta=0.1, batch_size=8, epochs=1, learning_rate=1e-4, weight_decay=1e-4, M=32, std_devs=[0.5], lambda_reg=0.01, letter='smoke')
    print('Training finished. val acc:', acc)


if __name__ == '__main__':
    main()
