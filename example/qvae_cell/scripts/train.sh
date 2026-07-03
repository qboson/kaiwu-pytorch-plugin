#!/bin/bash
python ../train_qvae_cell.py \
  --epochs 100 \
  --sampler-type sa \
  --loss-type mse \
  --feature_type q \
  --lr 1e-4 \
  --rbm-lr 3e-4 \
  --output-dir ./outputs_100ep_sa_mse_split_q_rbm3e-4 \
  --checkpoint-every 20 \
