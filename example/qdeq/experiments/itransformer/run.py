import argparse
import torch
from experiments.exp_long_term_forecasting import Exp_Long_Term_Forecast
from experiments.exp_long_term_forecasting_partial import Exp_Long_Term_Forecast_Partial
import random
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='iTransformer')

    # basic config
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--seed', type=int, default=2023, help='global random seed (torch/random/numpy) for model init and data shuffling -- overridable for multi-seed repeats; default 2023 matches the original hardcoded behavior')
    parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
    parser.add_argument('--model', type=str, required=True, default='iTransformer',
                        help='model name, options: [iTransformer, iInformer, iReformer, iFlowformer, iFlashformer]')

    # data loader
    parser.add_argument('--data', type=str, required=True, default='custom', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./data/electricity/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='electricity.csv', help='data csv file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length') # no longer needed in inverted Transformers
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')

    # model define
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size') # applicable on arbitrary number of variates in inverted Transformers
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
    parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')

    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    # iTransformer
    parser.add_argument('--exp_name', type=str, required=False, default='MTSF',
                        help='experiemnt name, options:[MTSF, partial_train]')
    parser.add_argument('--channel_independence', type=bool, default=False, help='whether to use channel_independence mechanism')
    parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)
    parser.add_argument('--class_strategy', type=str, default='projection', help='projection/average/cls_token')
    parser.add_argument('--target_root_path', type=str, default='./data/electricity/', help='root path of the data file')
    parser.add_argument('--target_data_path', type=str, default='electricity.csv', help='data file')
    parser.add_argument('--efficient_training', type=bool, default=False, help='whether to use efficient_training (exp_name should be partial train)') # See Figure 8 of our paper for the detail
    parser.add_argument('--use_norm', type=int, default=True, help='use norm and denorm')
    parser.add_argument('--partial_start_index', type=int, default=0, help='the start index of variates for partial training, '
                                                                           'you can select [partial_start_index, min(enc_in + partial_start_index, N)]')


    parser.add_argument('--deq_variant', type=str, default='ising_sa', help="'ising_sa' or 'anderson'")
    parser.add_argument('--deq_orthogonal_reinit', action='store_true', default=False,
                         help='after loading theta0, reinitialize shared-layer Q/K/V/O + conv1/conv2 weights with orthogonal init (keeps theta0 biases/norms/embedding/projector)')
    parser.add_argument('--deq_wnorm', action='store_true', default=False,
                         help='apply weight normalization (dim=0) to shared-layer Q/K/V/O + conv1/conv2 weights, matching locuslab/deq DEQ-Transformer wnorm')
    parser.add_argument('--jac_loss_weight', type=float, default=0.0,
                         help='target weight for Jacobian regularization (Hutchinson tr(J^TJ) estimate at z_star), matching locuslab/deq. 0.0 disables it.')
    parser.add_argument('--jac_pretrain_steps', type=int, default=0,
                         help='global training steps before jac_loss_weight starts ramping up from 0')
    parser.add_argument('--jac_ramp_steps', type=int, default=1000,
                         help='steps over which jac_loss_weight ramps linearly from 0 to its target value, starting after jac_pretrain_steps')
    parser.add_argument('--eval_batch_size', type=int, default=None, help='override test dataloader batch size')
    parser.add_argument('--sa_size_limit', type=int, default=100, help='kaiwu SA population size per restart')
    parser.add_argument('--kaiwu_user_id', type=str, default=None)
    parser.add_argument('--kaiwu_sdk_code', type=str, default=None)
    parser.add_argument('--cim_project_no', type=str, default='YOUR_CIM_PROJECT')
    parser.add_argument('--cim_task_prefix', type=str, default='qdeq_ts_cim')
    parser.add_argument('--cim_priority', type=int, default=10)
    parser.add_argument('--cim_interval', type=float, default=0.01)
    parser.add_argument('--cim_precision_bits', type=int, default=14)
    parser.add_argument('--cim_truncated_precision', type=int, default=18)
    parser.add_argument('--cim_max_retries', type=int, default=20)
    parser.add_argument('--cim_save_dir', type=str, default='cim_tasks')
    # --- Q-DEQ-Ising-SA args ---
    parser.add_argument('--init_checkpoint', type=str, default=None, help='theta0 checkpoint (1-layer explicit pretrain) to warmstart the DEQ shared layer')
    parser.add_argument('--anderson_threshold', type=int, default=20)
    parser.add_argument('--backward_threshold', type=int, default=30)
    parser.add_argument('--anderson_m', type=int, default=6)
    parser.add_argument('--anderson_eps', type=float, default=1e-3)
    parser.add_argument('--anderson_lam', type=float, default=1e-4)
    parser.add_argument('--input_injection_mode', type=str, default='add_norm')
    parser.add_argument('--ising_backend', type=str, default='sa')
    parser.add_argument('--ising_basis', type=int, default=8)
    parser.add_argument('--history_size', type=int, default=4)
    parser.add_argument('--coeff_bits', type=int, default=8)
    parser.add_argument('--ising_coeff_range', type=float, default=2.0, help='half-width R of the quantized per-basis coefficient range [-R,+R]*scale; default 2.0 matches original hardcoded bound')
    parser.add_argument('--ising_two_stage', action='store_true', help='coarse-then-fine two-pass coefficient quantization: solve QUBO once at full ising_coeff_range, then re-solve a second narrower QUBO (same A,b) centered on the result')
    parser.add_argument('--ising_adaptive_range', action='store_true', help='online R calibration: use the first ising_r_calibration_steps DEQ-solve steps to probe the unconstrained local-quadratic optimum (free, reuses the same A,b), lock a coeff_range from that, then use it for the remaining steps of the SAME solve() call. No extra SA calls, same total step budget.')
    parser.add_argument('--ising_r_calibration_steps', type=int, default=3, help='number of initial DEQ-solve steps used to probe/calibrate the adaptive coeff_range')
    parser.add_argument('--ising_r_margin', type=float, default=2.0, help='safety multiplier applied to the calibration-phase max |c*|/scale to get the locked coeff_range')
    parser.add_argument('--ising_r_min', type=float, default=0.5, help='lower clamp for the adaptively locked coeff_range')
    parser.add_argument('--ising_r_max', type=float, default=40.0, help='upper clamp for the adaptively locked coeff_range')
    parser.add_argument('--ising_probe_diag', action='store_true', help='diagnostic: print the unconstrained local-quadratic optimum ratio (|c*|/scale) every step, no extra SA calls, for range estimation')
    parser.add_argument('--ising_dynamic_range', action='store_true', help='continuously-tracking coeff_range: recompute margin*|c*|/scale EVERY step (not just an early calibration phase) and use it as that steps range. Fixes the case where the needed range grows through the solve trajectory instead of staying fixed.')
    parser.add_argument('--ising_dynamic_margin', type=float, default=1.5, help='safety multiplier applied to the per-step |c*|/scale to get that steps dynamic coeff_range')
    parser.add_argument('--ising_trust_region', action='store_true', help='trust-region step damping: shrink the applied SA coefficient when the local JVP-linear model badly mispredicts the actual residual reduction (measured gap between predicted and actual ratio), relax back toward full trust when agreement is good')
    parser.add_argument('--ising_trust_gap_threshold', type=float, default=0.05, help='predicted-vs-actual ratio gap above which the trust scale shrinks')
    parser.add_argument('--ising_trust_shrink', type=float, default=0.7, help='multiplicative shrink factor applied to trust_scale on a bad (large-gap) step')
    parser.add_argument('--ising_trust_grow', type=float, default=1.05, help='multiplicative grow factor applied to trust_scale on a good (small-gap) step, capped at 1.0')
    parser.add_argument('--ising_trust_min', type=float, default=0.1, help='lower clamp for trust_scale')
    parser.add_argument('--fd_eps_scale_ratio', type=float, default=None, help='if set, fd_eps is computed as this ratio times the current scale (r_norm/sqrt(m)) every step, instead of using the fixed --fd_eps constant')
    parser.add_argument('--fd_eps_scale_floor', type=float, default=0.3, help='lower clamp for fd_eps_scale_ratio-based fd_eps, to avoid collapsing into the noise-dominated regime once scale shrinks late in the trajectory')
    parser.add_argument('--ising_central_diff', action='store_true', help='use a central (2-sided) finite difference for the JVP estimate instead of the one-sided forward difference -- O(fd_eps^2) vs O(fd_eps) truncation error, 2x residual_at() cost per basis direction')
    parser.add_argument('--ising_secant_probe', action='store_true', help='probe the JVP at a step size proportional to the actual coefficient magnitude (secant_ratio * ising_coeff_range * scale) instead of a small fixed/scale-relative fd_eps, so the local linear model is fit to the actual trajectory span instead of extrapolated far beyond a tiny local neighborhood')
    parser.add_argument('--ising_secant_ratio', type=float, default=0.5, help='fraction of ising_coeff_range*scale used as the secant probe step when --ising_secant_probe is set')
    parser.add_argument('--diag_max_batches', type=int, default=None, help='diagnostic only: stop each training epoch after this many batches (for fixed-sample-size residual/convergence diagnostics)')
    parser.add_argument('--test_max_batches', type=int, default=None, help='diagnostic only: stop test-set evaluation after this many batches, computing mse/mae over just that subset (for cheap real-hardware CIM inference smoke tests)')
    parser.add_argument('--basis_mode', type=str, default='mixed_history')
    parser.add_argument('--fd_eps', type=float, default=1e-3)
    parser.add_argument('--ising_matrix_scale', type=float, default=1.0)
    parser.add_argument('--sa_initial_temperature', type=float, default=100.0)
    parser.add_argument('--sa_alpha', type=float, default=0.95)
    parser.add_argument('--sa_cutoff_temperature', type=float, default=0.1)
    parser.add_argument('--sa_iterations_per_t', type=int, default=5)
    parser.add_argument('--sa_restarts', type=int, default=4)
    parser.add_argument('--ising_variant', type=str, default='residual_scaled')
    parser.add_argument('--ising_seed', type=int, default=123)
    parser.add_argument('--quant_enabled', action='store_true', default=False)
    parser.add_argument('--quant_bits', type=int, default=8)

    args = parser.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print(args)

    if args.exp_name == 'partial_train': # See Figure 8 of our paper, for the detail
        Exp = Exp_Long_Term_Forecast_Partial
    else: # MTSF: multivariate time series forecasting
        Exp = Exp_Long_Term_Forecast


    if args.is_training:
        for ii in range(args.itr):
            # setting record of experiments
            setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
                args.model_id,
                args.model,
                args.data,
                args.features,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.d_model,
                args.n_heads,
                args.e_layers,
                args.d_layers,
                args.d_ff,
                args.factor,
                args.embed,
                args.distil,
                args.des,
                args.class_strategy, ii)

            exp = Exp(args)  # set experiments
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)

            if args.do_predict:
                print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                exp.predict(setting, True)

            torch.cuda.empty_cache()
    else:
        ii = 0
        setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
            args.model_id,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.d_model,
            args.n_heads,
            args.e_layers,
            args.d_layers,
            args.d_ff,
            args.factor,
            args.embed,
            args.distil,
            args.des,
            args.class_strategy, ii)

        exp = Exp(args)  # set experiments
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        torch.cuda.empty_cache()
