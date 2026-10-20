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
[[ -f "./dataset/ETT-small/ETTh1.csv" ]] || { echo "Missing required file: ./dataset/ETT-small/ETTh1.csv" >&2; exit 2; }
[[ -f "checkpoints/ETTh1_96_96_theta0_realign_iTransformer_ETTh1_M_ft96_sl48_ll96_pl256_dm8_nh1_el1_dl256_df1_fctimeF_ebTrue_dtExp_projection_0/checkpoint.pth" ]] || { echo "Missing required file: checkpoints/ETTh1_96_96_theta0_realign_iTransformer_ETTh1_M_ft96_sl48_ll96_pl256_dm8_nh1_el1_dl256_df1_fctimeF_ebTrue_dtExp_projection_0/checkpoint.pth" >&2; exit 2; }


"$PY" -u "$ENTRYPOINT" --is_training 1 --num_workers 10   --root_path ./dataset/ETT-small/ --data_path ETTh1.csv   --model_id ETTh1_96_96_ising_sa_formal --model iTransformerQDEQ --data ETTh1   --features M --seq_len 96 --label_len 48 --pred_len 96   --e_layers 1 --d_layers 1 --enc_in 7 --dec_in 7 --c_out 7 --des Exp   --d_model 256 --n_heads 8 --d_ff 256 --factor 1 --batch_size 32 --learning_rate 0.0002628831683901524   --init_checkpoint "checkpoints/ETTh1_96_96_theta0_realign_iTransformer_ETTh1_M_ft96_sl48_ll96_pl256_dm8_nh1_el1_dl256_df1_fctimeF_ebTrue_dtExp_projection_0/checkpoint.pth"   --deq_variant ising_sa --ising_backend sa --ising_variant residual_scaled   --ising_basis 8 --history_size 4 --basis_mode mixed_history --ising_matrix_scale 1.0   --ising_coeff_range 16 --coeff_bits 10 --fd_eps 0.3   --anderson_threshold 40 --anderson_eps 0.001 --backward_threshold 25   --sa_initial_temperature 100.0 --sa_alpha 0.95 --sa_cutoff_temperature 0.1 --sa_iterations_per_t 5 --sa_restarts 4 --sa_size_limit 200   --input_injection_mode add_norm --itr 1 --train_epochs 10 --patience 3   > qdeq_logs/ETTh1_96_96_ising_sa_formal.log 2>&1
echo ETTH1_FORMAL_DONE >> qdeq_logs/sweep_overview.log
