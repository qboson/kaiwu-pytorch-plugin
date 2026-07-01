#!/bin/bash
# Example script to run run_pipeline.py with arguments.

python ./run_pipeline.py \
  --model-type MNISTQVAE \
  --name mnist \
  --data-path ./data \
  --batch-size 256 \
  --epochs 10 \
  --lr 8e-4 \
  --bm-lr 8e-4 \
  --sampler-type sa \
  --loss-type bernoulli \
  --feature-type q \
  --mlp-hidden-dims 256 128 \
  --mlp-output-dim 10 \
  --mlp-lr 8e-5 \
  --mlp-epochs 100 \
  --compute_energy

# Optional arguments:
#   --run-tsne
#   --output-dir ./outputs_30ep_sa_mse_split_q_rbm8e-4
#   --use-cuda
