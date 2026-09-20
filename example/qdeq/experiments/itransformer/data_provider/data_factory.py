from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Solar, Dataset_PEMS, \
    Dataset_Pred
from torch.utils.data import DataLoader

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'Solar': Dataset_Solar,
    'PEMS': Dataset_PEMS,
    'custom': Dataset_Custom,
}


def data_provider(args, flag):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1

    if flag == 'test':
        shuffle_flag = False
        # Optional override: some models (e.g. the Q-DEQ-Ising-SA DEQ solver)
        # have per-forward-call cost that is nearly independent of batch
        # size (dominated by a fixed number of Jacobian-vector-product
        # evaluations and SA solver calls, not by tensor batch dimension),
        # so evaluating with the original bsz=1 convention multiplies the
        # number of forward calls needed by the batch size for no benefit.
        # Passing --eval_batch_size lets those models batch the test set
        # like train/val without dropping any samples (drop_last=False).
        # Default behavior (no override) is unchanged: bsz=1, drop_last=True,
        # matching the original Informer/Autoformer-lineage evaluation
        # convention so existing baselines remain reproducible.
        eval_bs_override = getattr(args, 'eval_batch_size', None)
        if eval_bs_override:
            # NOTE: drop_last=True here (not False) -- exp_long_term_forecasting.py's
            # test() does np.array(preds) on the list of per-batch prediction
            # arrays, which requires every batch to have the same shape. A
            # trailing partial batch (e.g. 1 leftover sample) breaks that with
            # "inhomogeneous shape". Dropping it costs at most
            # eval_batch_size-1 samples out of the whole test set (~0.03% for
            # ETTh1's 2785 samples / batch 32) -- negligible vs. crashing.
            drop_last = True
            batch_size = eval_bs_override
        else:
            drop_last = True
            batch_size = 1  # bsz=1 for evaluation
        freq = args.freq
    elif flag == 'pred':
        shuffle_flag = False
        drop_last = False
        batch_size = 1
        freq = args.freq
        Data = Dataset_Pred
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size  # bsz for train and valid
        freq = args.freq

    data_set = Data(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=freq,
    )
    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
    return data_set, data_loader
