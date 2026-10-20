# iTransformer integration

Install QDEQ from the repository root (`python -m pip install -e .`). The tested project source is in `experiments/itransformer`, already containing the opt-in launcher and adapter. Keep the dataset and any checkpoints in that project directory, or supply their paths explicitly.

Replace `run.py` in an existing command with `run_qdeq_plugin.py`; retain `--model iTransformerQDEQ` and the original model/solver settings. Use a new model ID and checkpoint directory. Original run.py behavior is untouched.

For another checkout, copy itransformer_qdeq.py and run_qdeq_plugin.py beside its run.py after installing QDEQ. Programmatically: `model = attach_qdeq(Model(configs))`. state_dict keys and checkpoints remain compatible.

## Validated environment

Linux, Python 3.8.20, PyTorch 2.4.1, NumPy 1.24.4, RTX 4090; Kaiwu 1.0.33a0 for the production SA backend. The legacy requirements.txt in the snapshot reflects an older upstream environment, not an exact lock for these results. Additional experiment dependencies include pandas, scikit-learn, matplotlib, requests, einops and reformer-pytorch; some optional attention models have separate dependencies.

## Reproduce the migration check

The formal ETTh1 checkpoint is not distributed. Place it under the model-specific checkpoint directory identified in validation/itransformer.json and obtain ETTh1.csv under dataset/ETT-small. Then, from the repository root:

```bash
python integrations/validate_itransformer.py --project experiments/itransformer --full-eval
```

The script uses the existing trained checkpoint, d_model=256, d_ff=256, 8 heads, seq_len=pred_len=96; Ising basis=8, bits=10, coefficient range=16, fd_eps=0.3, forward budget=40, backward budget=25, eps=1e-3 and SA size_limit=200. The complete input configuration is in the script.

It compares Anderson, Ising FP32 and Ising W8A8 with fixed RNG states and three temporary Adam steps. Full evaluation constructs fresh models from the original checkpoint, so those temporary steps do not affect the reported MSE. Evaluation includes the final partial batch (2785 samples total). Measurements in validation/itransformer.json preserve the original execution results.

The native CLI pipeline was separately validated for one Anderson epoch from the one-layer theta0 checkpoint: 264 training batches and full evaluation, MSE 0.393074631690979 / MAE 0.40563538670539856. This demonstrates operational integration, not a new optimized benchmark.

## Semantics and limits

The original production forward solver and its quantization/dropout behavior are preserved. QDEQ handles the generic solve/reattach/implicit-backward pipeline, with model-specific eval reattachment in this adapter. W8A8 is fake quantization, not native INT8 deployment.

Only single-GPU FP32/fake-quant was checked. DataParallel/DDP, AMP and other datasets are not validated. The original test pipeline drops the last partial batch when eval_batch_size is overridden; use default batch=1 or the dedicated validation script for complete evaluation. No server accounts, SDK credentials, datasets or trained weights are included.
