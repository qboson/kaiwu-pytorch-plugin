QBM + EBRM Short Training Results
=================================

What I ran
- Converted `RMB-Reward-Model-Benchmark/RMB_dataset/BoN_set/Harmlessness/S1.json` into `rmb_dataset.pt` (358 examples) using `prepare_rmb_dataset.py`.
- Ran a diagnostic training and a short training with QBM as the energy scorer.

Outputs
- `rmb_dataset.pt`: dataset used for training (358 examples).
- `model_final.pth`: PyTorch state_dict of the final model saved after training.
- `training_metrics.npz`: numpy archive containing `losses` and `val_accs` arrays per epoch.
- `training_plots.png`: training plots (not generated because `matplotlib` is not installed in the environment). If you install matplotlib, re-run `save_and_plot_results.py` to produce the plot.

Notes
- Training was run offline (WANDB_MODE=offline) to avoid interactive prompts. W&B artifacts are saved under `wandb/`.
- Final validation accuracy from the short run: ~0.50–0.56 depending on epoch; training loss remained relatively large and may benefit from further hyperparameter tuning (smaller M, different learning rate, stronger regularization, or gradient clipping which is already enabled).

How to reproduce
1. Install requirements (optional):
```bash
pip install -r requirements/devel.txt
pip install matplotlib
```
2. Generate dataset (if not already):
```bash
python3 example/qbm_ebrm_results/prepare_rmb_dataset.py
```
3. Run full save+plot training (uses offline W&B):
```bash
WANDB_MODE=offline python3 example/qbm_ebrm_results/save_and_plot_results.py
```

If you want, I can continue hyperparameter tuning and produce plots and a cleaned report.
QBM + EBRM Integration
======================

内容：本目录包含将 QBM（基于仓库中 `BoltzmannMachine`）作为替代能量评分器集成到 EBRM 的示例文件。

主要文件：
- `run_qbm_ebm.py`: 一个轻量化的 smoke-test 脚本，验证 `QBMModel` 可被导入并在随机输入上运行。

使用说明：
1. （可选）设置环境变量以启用 QBM（默认已启用）：
   - `export USE_QBM=1`
2. 运行 smoke test：
   - `python example/qbm_ebrm_results/run_qbm_ebm.py`

注意：完整训练请使用 `EBRM/src/reward_modeling/ebm_training/ebm_nce_plus.py`，该脚本现在在直接作为模块导入时不会自动启动训练（训练入口受 `if __name__ == '__main__'` 保护）。
