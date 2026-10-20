"""GPU regression on the real project, checkpoint and ETTh1 data."""
import argparse
import contextlib
import io
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser()
parser.add_argument("--project", type=Path, required=True)
parser.add_argument("--full-eval", action="store_true")
args = parser.parse_args()
sys.path.insert(0, str(args.project))
from model.iTransformerQDEQ import Model
from data_provider.data_loader import Dataset_ETT_hour
from itransformer_qdeq import attach_qdeq


def seed():
    random.seed(2023)
    np.random.seed(2023)
    torch.manual_seed(2023)
    torch.cuda.manual_seed_all(2023)


def config(variant, quant):
    return SimpleNamespace(seq_len=96, pred_len=96, output_attention=False, use_norm=True,
        d_model=256, d_ff=256, n_heads=8, embed="timeF", freq="h", dropout=0.1,
        class_strategy="projection", factor=1, activation="gelu", deq_variant=variant,
        quant_enabled=quant, quant_bits=8, anderson_threshold=40, backward_threshold=25,
        anderson_eps=1e-3, anderson_m=6, anderson_lam=1e-4,
        ising_basis=8, history_size=4, coeff_bits=10, fd_eps=0.3, ising_coeff_range=16,
        sa_size_limit=200, input_injection_mode="add_norm", ising_backend="sa")


def batch_to_device(batch):
    return [t.float().cuda() for t in batch]


def call(model, batch):
    x, y, xm, ym = batch
    return model(x, xm, None, ym)


seed()
torch.set_num_threads(4)
ckpts = sorted(args.project.glob("checkpoints/ETTh1_96_96_ising_sa_formal_*/checkpoint.pth"))
assert len(ckpts) == 1, ckpts
state = torch.load(ckpts[0], map_location="cpu", weights_only=True)
data = Dataset_ETT_hour(str(args.project/"dataset/ETT-small"), flag="test",
                        size=[96,48,96], features="M", data_path="ETTh1.csv", timeenc=1)
loader = DataLoader(data, batch_size=32 if args.full_eval else 2, shuffle=False, drop_last=False)
batch = batch_to_device(next(iter(loader)))
records = []
for variant, quant in [("anderson", False), ("ising_sa", False), ("ising_sa", True)]:
    old = Model(config(variant, quant)).cuda()
    new = Model(config(variant, quant)).cuda()
    old.load_state_dict(state, strict=True)
    new.load_state_dict(state, strict=True)
    attach_qdeq(new)
    assert old.state_dict().keys() == new.state_dict().keys()
    old.eval(); new.eval()
    with torch.no_grad(), contextlib.redirect_stdout(io.StringIO()):
        seed(); a = call(old,batch)
        seed(); b = call(new,batch)
    torch.testing.assert_close(a,b,atol=1e-6,rtol=1e-5)
    eval_diff = (a-b).abs().max().item()
    opts = [torch.optim.Adam(m.parameters(), lr=1e-5) for m in (old,new)]
    step_records = []
    for step in range(3):
        results = []
        for model,opt in zip((old,new),opts):
            model.train(); opt.zero_grad(set_to_none=True); seed()
            with contextlib.redirect_stdout(io.StringIO()):
                pred = call(model,batch)
                loss = (pred-batch[1][:,-96:,:]).square().mean()
                loss.backward()
            grads = {k:p.grad.detach().clone() for k,p in model.named_parameters() if p.grad is not None}
            assert all(torch.isfinite(g).all() for g in grads.values())
            results.append((pred.detach(), loss.item(), grads))
        a, b = results
        torch.testing.assert_close(a[0],b[0],atol=2e-5,rtol=2e-4)
        assert a[2].keys() == b[2].keys()
        ga,gb = [],[]
        for k in a[2]:
            torch.testing.assert_close(a[2][k],b[2][k],atol=2e-5,rtol=2e-3,msg=k)
            ga.append(a[2][k].flatten()); gb.append(b[2][k].flatten())
        ga,gb = torch.cat(ga),torch.cat(gb)
        step_records.append(dict(step=step, old_loss=a[1], plugin_loss=b[1],
            gradient_relative=((ga-gb).norm()/ga.norm()).item()))
        for opt in opts: opt.step()
    record = dict(variant=variant, quant=quant, eval_max_abs=eval_diff, training=step_records)
    records.append(record)
    print(json.dumps(record),flush=True)
    del old,new,opts
if args.full_eval:
    old=Model(config("ising_sa",True)).cuda(); new=Model(config("ising_sa",True)).cuda()
    old.load_state_dict(state,strict=True); new.load_state_dict(state,strict=True); attach_qdeq(new)
    old.eval();new.eval()
    sums=[0.,0.]; count=0; max_diff=0.; samples=0
    with torch.no_grad():
        for i, raw in enumerate(loader):
            batch=batch_to_device(raw)
            with contextlib.redirect_stdout(io.StringIO()):
                seed();a=call(old,batch)
                seed();b=call(new,batch)
            torch.testing.assert_close(a,b,atol=1e-6,rtol=1e-5)
            target=batch[1][:,-96:,:]
            for j,p in enumerate((a,b)): sums[j]+=(p-target).double().square().sum().item()
            count+=target.numel();samples+=target.shape[0]
            max_diff=max(max_diff,(a-b).abs().max().item())
            if i%10==0: print(json.dumps(dict(eval_batch=i,samples=samples)),flush=True)
    assert samples==len(data)
    records.append(dict(full_eval_samples=samples,old_mse=sums[0]/count,plugin_mse=sums[1]/count,max_abs=max_diff))
    print(json.dumps(records[-1]),flush=True)
Path("itransformer_validation.json").write_text(json.dumps(dict(checkpoint=str(ckpts[0]),torch=torch.__version__,results=records),indent=2))
print("PASS: checkpoint compatibility, GPU forward/backward and optimizer steps",flush=True)
