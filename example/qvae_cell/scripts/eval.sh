#!/bin/bash
export OUTPUT_DIR=./outputs_100ep_sa_mse_fullbm_batch_q_binpos
echo $OUTPUT_DIR
python evaluate_clustering.py \
  --h5ad ./$OUTPUT_DIR/immune_kpp_qvae.h5ad \
  --rep-key X_qvae \
  --label-key final_annotation \
  --resolutions 0.05,0.1,0.15,0.2,0.3,0.4
