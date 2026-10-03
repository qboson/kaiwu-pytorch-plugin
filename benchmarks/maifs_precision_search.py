"""Small pinned-SDK precision search timing, without a sampler or network."""
import json
from pathlib import Path
import statistics
import sys
import time

import kaiwu
import numpy as np

# An optional source root permits measuring the original code independently.
source = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1]
kaiwu.__path__.insert(0, str(source.resolve() / "src/kaiwu"))
from kaiwu.torch_plugin.maifs import qubo  # pylint: disable=wrong-import-position

matrix = np.triu(np.random.default_rng(8).normal(size=(4, 4)), 1)
samples = []
for repeat in range(21):
    explorer = qubo.PrecisionSplitExplorer(
        target_precision=3, min_precision=3, max_precision=12, max_bits=11,
    )
    start = time.perf_counter()
    plan = explorer.search(matrix)
    elapsed = time.perf_counter() - start
    if repeat:
        samples.append(elapsed)
print(json.dumps({"source": qubo.__file__, "source_precision": plan.source_precision,
                  "split_size": plan.split_size, "attempts": len(plan.history),
                  "median_ms": statistics.median(samples) * 1000}, indent=2))
