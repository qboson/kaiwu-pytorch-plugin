"""Local migration regression, using a shared non-radial target.

Compare wrapper equivalence separately from backward solver accuracy.
Old deq_sequence and QDEQ both return f(z*); old straight_through returns z*.
The historical squared-norm-loss verdict (including Anderson false convergence)
is withdrawn. This small CPU check does NOT certify remote iTransformer/QAT.

Run from any directory: python -X utf8 tests/qdeq_migration_regression.py
"""
from __future__ import annotations

import json
import argparse
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT.parent))

import torch
from nmt_deq_ising.models.normalized_decoder import NormalizedMatchingDecoder
from qdeq.core import QDEQ
from qdeq.ising_sa import IsingSolverConfig
from qdeq.solvers import anderson_solver, ising_sa_solver

ISING_KWARGS = dict(threshold=10, eps=1e-3, basis=4, history_size=4,
                    coeff_bits=4, solver_variant="v5_dynamic",
                    solver_backend="cpu_numpy", sa_restarts=6, sa_steps=40, seed=123)


def build_decoder(anchor_alpha=0.0):
    torch.manual_seed(0)
    return NormalizedMatchingDecoder(dict(dim=16, num_heads=2, mlp_hidden_mult=2,
        alpha=0.5, mode="deq_ising", anchor_alpha=anchor_alpha, ising=dict(ISING_KWARGS)))


def make_inputs(seed):
    g = torch.Generator().manual_seed(seed)
    ha, hb = (torch.randn(2, 3, 16, generator=g) for _ in range(2))
    return ha, hb, torch.ones(2, 3, dtype=torch.bool), torch.ones(2, 3, dtype=torch.bool), torch.zeros(2, 3, 3), torch.zeros(2, 3, 3)


def direct_solver(f, z0, **kwargs):
    # Only a small-test reference: construct the affine backward map columnwise.
    # Solve in float64 to reduce linear-solve error; VJPs retain model precision.
    rhs = f(torch.zeros_like(z0)).detach()
    n = z0.numel()
    columns = []
    for i in range(n):
        unit = torch.zeros_like(z0)
        unit.reshape(-1)[i] = 1
        columns.append((f(unit) - rhs).reshape(-1).double())
    matrix = torch.eye(n, dtype=torch.float64) - torch.stack(columns, dim=1)
    y = torch.linalg.solve(matrix, rhs.reshape(-1).double()).to(z0).reshape_as(z0)
    return {"result": y, "nstep": n, "condition_number": torch.linalg.cond(matrix).item()}


def measured(solver):
    def solve(f, z0, **kwargs):
        result = dict(solver(f, z0, **kwargs))
        y = result["result"]
        fy = f(y)
        result["true_abs"] = (fy - y).norm().item()
        result["true_rel"] = result["true_abs"] / max(fy.norm().item(), 1e-12)
        return result
    return solve


def run(decoder, inputs, target, route):
    ha, hb, ma, mb, aa, ab = inputs
    ha, hb = (t.detach().clone().requires_grad_() for t in (ha, hb))
    decoder.zero_grad(set_to_none=True)
    z0 = torch.cat([ha, hb], dim=1)
    f = lambda z: decoder._apply_block(z, 3, ma, mb, aa, ab, anchor=z0)
    torch.manual_seed(999)
    if route.startswith("old"):
        decoder.deq_solver.config.reattach_mode = "straight_through" if route == "old_st" else "deq_sequence"
        a, b, info = decoder(ha, hb, ma, mb, aa, ab)
        out = torch.cat([a, b], dim=1)
        raw = info["result"]
    else:
        back = {"ising": ising_sa_solver(IsingSolverConfig(**ISING_KWARGS)),
                "anderson": anderson_solver, "direct": direct_solver}[route]
        deq = QDEQ(f_solver=ising_sa_solver(IsingSolverConfig(**ISING_KWARGS)),
                   b_solver=measured(back), b_thres=10 if route == "ising" else 50)
        out, info = deq(f, z0)
        raw = info["z_star"]
    (out * target).sum().backward()
    grads = {name: None if p.grad is None else p.grad.detach().clone()
             for name, p in decoder.named_parameters()}
    grads.update(input_a=ha.grad, input_b=hb.grad)
    with torch.no_grad():
        step = f(raw)
        forward_rel = ((step - raw).norm() / raw.norm().clamp_min(1e-12)).item()
    return dict(out=out.detach(), raw=raw.detach(), grads=grads,
                forward_rel=forward_rel, info=info)


def compare(a, b, *, assert_close=False):
    assert a.keys() == b.keys()
    av, bv, per_parameter = [], [], {}
    for name in a:
        assert (a[name] is None) == (b[name] is None), name
        if a[name] is None:
            continue
        assert torch.isfinite(a[name]).all() and torch.isfinite(b[name]).all(), name
        if assert_close:
            torch.testing.assert_close(a[name], b[name], atol=2e-5, rtol=2e-4, msg=name)
        error = (a[name] - b[name]).norm().item()
        norm = b[name].norm().item()
        per_parameter[name] = {"abs": error, "reference_norm": norm,
                               "rel": error / norm if norm > 1e-6 else None}
        av.append(a[name].reshape(-1))
        bv.append(b[name].reshape(-1))
    av, bv = torch.cat(av), torch.cat(bv)
    return {"abs": (av-bv).norm().item(), "rel": ((av-bv).norm()/bv.norm().clamp_min(1e-12)).item(),
            "reference_norm": bv.norm().item(), "per_parameter": per_parameter}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Write complete metrics and source hashes to JSON")
    args = parser.parse_args()
    records = []
    for seed, anchor in [(42, 0.0), (43, 0.0), (42, 0.2)]:
        decoder, inputs = build_decoder(anchor), make_inputs(seed)
        target = torch.randn(2, 6, 16, generator=torch.Generator().manual_seed(7))
        runs = {route: run(decoder, inputs, target, route)
                for route in ["old_st", "old_aligned", "ising", "anderson", "direct"]}
        for r in runs.values():
            torch.testing.assert_close(r["raw"], runs["ising"]["raw"], atol=0, rtol=0)
            assert torch.isfinite(r["out"]).all()
        torch.testing.assert_close(runs["old_st"]["out"], runs["ising"]["raw"], atol=0, rtol=0)
        torch.testing.assert_close(runs["old_aligned"]["out"], runs["ising"]["out"], atol=0, rtol=0)
        equivalent = compare(runs["old_aligned"]["grads"], runs["ising"]["grads"], assert_close=True)
        # Linear target makes incoming gradients independent of the output stage.
        compare(runs["old_st"]["grads"], runs["ising"]["grads"], assert_close=True)
        accuracy = {name: compare(runs[name]["grads"], runs["direct"]["grads"])
                    for name in ["ising", "anderson"]}
        assert accuracy["anderson"]["reference_norm"] > 1e-3
        assert accuracy["anderson"]["rel"] < 0.01, accuracy
        assert runs["direct"]["info"]["backward_true_rel"] < 1e-5
        assert runs["direct"]["info"]["backward_condition_number"] < 1e4
        assert runs["anderson"]["info"]["backward_true_rel"] < 0.002
        if anchor > 0:
            assert runs["anderson"]["grads"]["input_a"].norm() > 0
        record = dict(seed=seed, anchor_alpha=anchor, wrapper_error=equivalent,
                      forward_rel=runs["ising"]["forward_rel"], gradient_error=accuracy,
                      backward_residual={k: runs[k]["info"]["backward_true_rel"] for k in accuracy},
                      condition_number=runs["direct"]["info"]["backward_condition_number"])
        records.append(record)
        summary = {**record, "wrapper_error": {k: v for k, v in equivalent.items() if k != "per_parameter"},
                   "gradient_error": {name: {k: v for k, v in metrics.items() if k != "per_parameter"}
                                      for name, metrics in accuracy.items()}}
        print(json.dumps(summary), flush=True)
    if args.output:
        source_paths = [Path(__file__).resolve(), ROOT / "src/nmt_deq_ising/models/normalized_decoder.py",
                        ROOT / "src/nmt_deq_ising/solvers/deq_ising.py"]
        source_paths += sorted((ROOT.parent / "qdeq").glob("*.py"))
        hashes = {str(p.relative_to(ROOT.parent)): hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in source_paths}
        args.output.write_text(json.dumps({"torch": torch.__version__, "device": "cpu",
                                          "source_sha256": hashes, "cases": records}, indent=2), encoding="utf-8")
    print("PASS: local wrapper equivalence and Anderson reference checks. Ising accuracy is reported separately; remote migration remains untested.")


if __name__ == "__main__":
    main()
