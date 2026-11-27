"""Check CPU random-stream replay through the DPLM runtime seed helper."""

import json
from pathlib import Path
import subprocess
import sys
import unittest


_MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "example" / "qdiffusion" / "dplm" / "utils" / "runtime.py"
)
_DRAW_SCRIPT = """
import importlib.util
import json
import sys
import torch

spec = importlib.util.spec_from_file_location("_dplm_seed_contract_child", sys.argv[1])
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)
runtime.seed_torch(int(sys.argv[2]))
first = torch.rand(32, device="cpu")
runtime.seed_torch(int(sys.argv[3]))
second = torch.rand(32, device="cpu")
print(json.dumps([first.tolist(), second.tolist()]))
"""


def _draws_after_seeding(first_seed, second_seed):
    # A child keeps every backend's seeding effects out of the parent process.
    result = subprocess.run(
        [sys.executable, "-I", "-B", "-c", _DRAW_SCRIPT, str(_MODULE_PATH),
         str(first_seed), str(second_seed)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "Seed-helper child failed with exit {}:\n{}".format(
                result.returncode, result.stderr
            )
        )
    return json.loads(result.stdout)


class TestRuntimeSeed(unittest.TestCase):
    def test_same_seed_replays_cpu_draws(self):
        first, replay = _draws_after_seeding(1234, 1234)
        self.assertEqual(first, replay)

    def test_distinct_seeds_produce_different_cpu_draws(self):
        first, different = _draws_after_seeding(1234, 5678)
        self.assertNotEqual(first, different)


if __name__ == "__main__":
    unittest.main()
