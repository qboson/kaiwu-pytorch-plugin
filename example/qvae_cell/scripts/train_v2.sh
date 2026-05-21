#!/bin/bash
export OUTPUT_DIR=./outputs_100ep_sa_mse_fullbm_batch_q_binpos
echo $OUTPUT_DIR
python train_qvae_cell.py \
  --epochs 100 \
  --batch-size 512 \
  --early-stopping-patience 10 \
  --sampler-type sa \
  --loss-type mse \
  --representation q \
  --kl-beta 1e-5 \
  --lr 1e-4 \
  --rbm-lr 1e-3 \
  --sa-initial-temperature 1000 \
  --sa-alpha 0.5 \
  --sa-cutoff-temperature 0.001 \
  --sa-iterations-per-t 10 \
  --sa-size-limit 10 \
  --sa-rand-seed 512 \
  --output-dir OUTPUT_DIR \
  --checkpoint-every 20
