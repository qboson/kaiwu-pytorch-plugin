"""Run Hessian-method QUBO PTQ on LeNet-5 with Kaiwu CIM."""

# pylint: disable=wrong-import-position,duplicate-code

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch


from quant.lenet5_mnist import evaluate, load_model, make_loader
from kaiwu.torch_plugin import quantization


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data"
MODEL_PATH = ROOT / "base_model.pth"
HESSIAN_PATH = ROOT / "base_hessian_matrix_module_layerwise_params.pt"
CIM_SAVE_DIR = ROOT / "kaiwu_cim_results"
KAIWU_PROJECT_NO = "26071422"

BITS = 4
LAYERS = None
BATCH_SIZE = 64
SOLVERS = [
    {
        "name": "cim",
        "params": {
            "project_no": KAIWU_PROJECT_NO,
            "task_name": "qubo_ptq_lenet5_hessian",
            "task_mode": "sample",
            "sample_number": 100,
            "wait": True,
            "interval": 1,
            "bit_width": 14,
            "save_dir": str(CIM_SAVE_DIR),
        },
    }
]

model = load_model(MODEL_PATH)
hessians = torch.load(HESSIAN_PATH, map_location="cpu", weights_only=False)
test_loader = make_loader(DATA_ROOT, train=False, batch_size=BATCH_SIZE)

before_loss, before_accuracy = evaluate(model, test_loader)
print(f"before quantization: loss={before_loss:.6f}, accuracy={before_accuracy:.4%}")

quantized_model, report = quantization(
    model,
    bits=BITS,
    solvers=SOLVERS,
    layers=LAYERS,
).hessian(hessians)

after_loss, after_accuracy = evaluate(quantized_model, test_loader)
print(f"after quantization:  loss={after_loss:.6f}, accuracy={after_accuracy:.4%}")
print(
    f"method={report['method']}, "
    f"bits={report['bits']}, "
    f"layers={[layer['name'] for layer in report['layers']]}"
)
