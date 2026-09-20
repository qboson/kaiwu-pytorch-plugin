#!/bin/bash
set -eu
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_DIR:-$SCRIPT_DIR/../../experiments/itransformer}"
PY="${PYTHON:-python3}"
ENTRYPOINT="${ENTRYPOINT:-run.py}"
case "$ENTRYPOINT" in run.py|run_qdeq_plugin.py) ;; *) echo "Unsupported ENTRYPOINT" >&2; exit 2;; esac
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
mkdir -p qdeq_logs
# Fail before training if the historical initialization or dataset is missing.
[[ -f "./dataset/traffic/traffic.csv" ]] || { echo "Missing required file: ./dataset/traffic/traffic.csv" >&2; exit 2; }
[[ -f "checkpoints/traffic_96_96_theta0_anderson_iTransformer_custom_M_ft96_sl48_ll96_pl512_dm8_nh1_el1_dl512_df1_fctimeF_ebTrue_dtExp_projection_0/checkpoint.pth" ]] || { echo "Missing required file: checkpoints/traffic_96_96_theta0_anderson_iTransformer_custom_M_ft96_sl48_ll96_pl512_dm8_nh1_el1_dl512_df1_fctimeF_ebTrue_dtExp_projection_0/checkpoint.pth" >&2; exit 2; }

ulimit -n 65536

COMMON_TR="--num_workers 10 --root_path ./dataset/traffic/ --data_path traffic.csv --model iTransformerQDEQ --data custom   --features M --seq_len 96 --label_len 48 --pred_len 96   --e_layers 1 --d_layers 1 --enc_in 862 --dec_in 862 --c_out 862 --des Exp   --d_model 512 --n_heads 8 --d_ff 512 --factor 1 --batch_size 16 --learning_rate 0.0013982707059266432   --input_injection_mode add_norm --itr 1 --train_epochs 10 --patience 3"
CKPT_THETA0_TR="checkpoints/traffic_96_96_theta0_anderson_iTransformer_custom_M_ft96_sl48_ll96_pl512_dm8_nh1_el1_dl512_df1_fctimeF_ebTrue_dtExp_projection_0/checkpoint.pth"

run_anderson_tr () {
  local seed=$1
  "$PY" -u "$ENTRYPOINT" --is_training 1 --seed $seed $COMMON_TR     --model_id traffic_96_96_anderson_seed$seed --init_checkpoint $CKPT_THETA0_TR     --deq_variant anderson --anderson_threshold 20 --anderson_m 10 --backward_threshold 30     >> qdeq_logs/traffic_anderson_seed$seed.log 2>&1
}
run_ising_fp32_tr () {
  local seed=$1
  "$PY" -u "$ENTRYPOINT" --is_training 1 --seed $seed $COMMON_TR     --model_id traffic_96_96_ising_fp32_seed$seed --init_checkpoint $CKPT_THETA0_TR     --deq_variant ising_sa --ising_backend sa --ising_variant residual_scaled     --ising_basis 8 --history_size 4 --basis_mode mixed_history --ising_matrix_scale 1.0     --ising_coeff_range 5 --coeff_bits 10 --fd_eps 0.3     --anderson_threshold 40 --anderson_eps 0.001 --backward_threshold 30     --sa_initial_temperature 100.0 --sa_alpha 0.95 --sa_cutoff_temperature 0.1 --sa_iterations_per_t 5 --sa_restarts 4 --sa_size_limit 200     >> qdeq_logs/traffic_ising_fp32_seed$seed.log 2>&1
}
run_ising_int8_tr () {
  local seed=$1
  "$PY" -u "$ENTRYPOINT" --is_training 1 --seed $seed $COMMON_TR     --model_id traffic_96_96_ising_int8_seed$seed --init_checkpoint $CKPT_THETA0_TR     --deq_variant ising_sa --ising_backend sa --ising_variant residual_scaled     --ising_basis 8 --history_size 4 --basis_mode mixed_history --ising_matrix_scale 1.0     --ising_coeff_range 5 --coeff_bits 10 --fd_eps 0.3     --anderson_threshold 40 --anderson_eps 0.001 --backward_threshold 30     --sa_initial_temperature 100.0 --sa_alpha 0.95 --sa_cutoff_temperature 0.1 --sa_iterations_per_t 5 --sa_restarts 4 --sa_size_limit 200     --quant_enabled --quant_bits 8     >> qdeq_logs/traffic_ising_int8_seed$seed.log 2>&1
}

for seed in 2023 42 2024; do
  echo "=== [$(date)] traffic seed=$seed anderson ===" >> qdeq_logs/traffic_multiseed_overview.log
  run_anderson_tr $seed
  echo "=== [$(date)] traffic seed=$seed ising_fp32 ===" >> qdeq_logs/traffic_multiseed_overview.log
  run_ising_fp32_tr $seed
  echo "=== [$(date)] traffic seed=$seed ising_int8 ===" >> qdeq_logs/traffic_multiseed_overview.log
  run_ising_int8_tr $seed
done

echo TRAFFIC_MULTISEED_DONE >> qdeq_logs/sweep_overview.log
