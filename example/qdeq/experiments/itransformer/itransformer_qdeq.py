"""Opt-in QDEQ integration for the existing iTransformerQDEQ model.

Keeps state_dict keys, tuned production forward solvers, quantization and
train/eval semantics. Import after adding the original iTransformer root to sys.path.
"""
from contextlib import contextmanager
from functools import partial
from types import MethodType

import torch
from qdeq import QDEQ, anderson_solver
from model.deq_ising_ts import (EncoderLayerIsingDEQ, EncoderLayerAndersonDEQ,
                               original_anderson, set_quant_enabled, jac_loss_estimate)


def attach_qdeq(model):
    owner = model.deq
    if hasattr(owner, "qdeq_plugin"):
        return model
    is_ising = isinstance(owner, EncoderLayerIsingDEQ)
    if not is_ising and not isinstance(owner, EncoderLayerAndersonDEQ):
        raise TypeError("Expected the project's Ising or Anderson DEQ layer")

    @contextmanager
    def solve_context():
        states = [(m, m.training) for m in owner.layer.modules()]
        if is_ising:
            if owner.quant_enabled_flag:
                set_quant_enabled(owner.layer, False)
            owner.layer.eval()
        try:
            yield
        except BaseException:
            if is_ising and owner.quant_enabled_flag:
                set_quant_enabled(owner.layer, True)
            raise
        finally:
            for module, state in states:
                module.training = state

    @contextmanager
    def reattach_context():
        if is_ising and owner.quant_enabled_flag:
            set_quant_enabled(owner.layer, True)
        yield

    def forward_solver(f, z0, **kwargs):
        if is_ising:
            return owner.ising_solver.solve(f, z0, tag="qdeq_plugin", **kwargs)
        return original_anderson(f, z0, m=owner.anderson_m, lam=owner.anderson_lam, **kwargs)

    owner.qdeq_plugin = QDEQ(
        f_solver=forward_solver,
        b_solver=partial(anderson_solver, m=owner.anderson_m, lam=owner.anderson_lam),
        f_thres=owner.anderson_threshold, b_thres=owner.backward_threshold,
        f_eps=owner.anderson_eps, b_eps=owner.anderson_eps,
        solve_ctx=solve_context, reattach_ctx=reattach_context)
    owner.qdeq_plugin.train(owner.training)

    def forward(self, x):
        f = lambda z: self.explicit_once(z, x)
        out, info = self.qdeq_plugin(f, x)
        if not self.training:
            # The original model performs a reattach even at inference. QDEQ's
            # generic eval API deliberately returns raw z*, so preserve that
            # model-specific step here without altering QDEQ's eval contract.
            with reattach_context():
                out = f(out)
        self.last_jac_loss = None
        if self.training and getattr(self, "jac_loss_weight", 0.0) > 0:
            self.last_jac_loss = jac_loss_estimate(info["new_z"], info["z_star"], vecs=1)
        self.last_solver_info = info
        return out

    owner.forward = MethodType(forward, owner)
    return model
