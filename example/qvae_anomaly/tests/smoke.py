"""1 epoch smoke test: verify model forward + backward.

Usage: python tests/smoke.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
from model import Config, build_model  # noqa: E402
from trainer.trainer import set_seed, train_model  # noqa: E402
from utils.logging import get_logger  # noqa: E402

logger = get_logger("smoke")

set_seed(42)
cfg = Config(
    input_dimension=74,
    hidden_dim=32,
    num_var1=16,
    num_var2=16,
    epochs=1,
    batch_size=64,
    lambda_anom=1.0,
)

# Dummy data 1:20
n = 200
X = torch.randn(n, 74)
y = torch.cat([torch.zeros(n - 10), torch.ones(10)]).long()
logger.info("Dataset: dummy (74d, 200 samples, 1:20)")
model = build_model(cfg)
out = train_model(model, cfg, X.numpy(), y.numpy(), X.numpy(), y.numpy(), tag="smoke")
print("SMOKE OK")
