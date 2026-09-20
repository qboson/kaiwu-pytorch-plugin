"""Run the original iTransformer CLI with opt-in QDEQ integration.

Place beside run.py, itransformer_qdeq.py and the qdeq package. All arguments
are passed unchanged to run.py. Select --model iTransformerQDEQ as usual.
"""
import runpy
from pathlib import Path
from types import SimpleNamespace
from experiments import exp_basic

from model import iTransformerQDEQ
from itransformer_qdeq import attach_qdeq

OriginalModel = iTransformerQDEQ.Model


class PluginModel(OriginalModel):
    def __init__(self, configs):
        super().__init__(configs)
        attach_qdeq(self)
        print("[QDEQ plugin] enabled; production forward solver and eval semantics preserved")


def main():
    # Keep the original module.Model binding: its constructor uses
    # super(Model, self), so replacing that global would break initialization.
    exp_basic.iTransformerQDEQ = SimpleNamespace(Model=PluginModel)
    runpy.run_path(str(Path(__file__).resolve().with_name("run.py")), run_name="__main__")


if __name__ == "__main__":
    main()
