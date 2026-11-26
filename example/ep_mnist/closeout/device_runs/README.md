# CPU/CUDA execution evidence

The archived CUDA EP run completed three epochs with test accuracy 89.06%.
Its model, gradients and SGD used CUDA; Kaiwu simulated annealing remained on CPU.
The accompanying BP CPU and CUDA runs are separate three-epoch baselines.
These are single-seed, single-run observations, not statistical speedup estimates.

The unified `train_mnist.py` now accepts `--device cpu`, `--device cuda`, or
`--device cuda:N`. It initializes parameters on CPU before migration, disables
TF32 and uses one PyTorch CPU thread by default (`--threads`). Use the matching
PyTorch CUDA installation for GPU execution; the package itself does not force
a particular CUDA wheel.

Each training, evaluation and test phase reports total wall time, SA wall time,
SA call count and non-SA time. The latter is total minus SA, and still includes
matrix construction, transfers, gradients and Python overhead. CUDA phase
boundaries are synchronized. Evaluation includes train and validation accuracy.

From the repository root, after installing the package and example dependencies:

```sh
python example/ep_mnist/train_mnist.py --data-dir data --device cpu --threads 1 --epochs 3 --seed 0 --split-seed 0 --output-dir runs/ep-cpu
python example/ep_mnist/train_mnist.py --data-dir data --device cuda --threads 1 --epochs 3 --seed 0 --split-seed 0 --output-dir runs/ep-cuda
python example/ep_mnist/train_mnist.py --data-dir data --device cpu --evaluate runs/ep-cuda/checkpoint.pt --output-dir runs/ep-evaluation
```

Set OMP_NUM_THREADS=1 and MKL_NUM_THREADS=1 before running to reproduce the
recorded thread configuration. Checkpoints support evaluation, not training resume.
Device choice for evaluation comes from the command line, independently of the
saved model configuration. Existing output directories are never overwritten.

`provenance.json` stores original local artifact hashes. Published report copies
omit process IDs and the local output path. Executed source snapshots are retained
in `../source_snapshots/`; they are archival evidence, not the current entrypoint.
The historical manifest's training-source hash refers to
`train_mnist_before_device.py`. Existing historical results remain unchanged.
The historical dispatch-script hash resolves to `run_experiments_before_device.py`;
the current dispatcher additionally forwards the EP device option. Both mappings
are explicit in `provenance.json` under `historical_source_mapping`.

The unified-entrypoint CPU run completed at 2026-11-26 03:33:41 Asia/Shanghai,
with 90.30% test accuracy. Its executed sources are retained as
`train_mnist_cpu_full.py` and `device_timing_cpu_full.py`. The final entrypoint
changes only helper import isolation after this run; the full run was not repeated.
Historical CPU results did not record non-SA time and remain separate.
The completed CUDA experiment used the archived harness, not the unified entrypoint.

## Completed observations

All times below are minutes summed across three epochs plus final testing.
Evaluation includes training and validation accuracy. GPU totals use the
device report's phase boundaries consistently for both total and SA time.

| Run | Test accuracy | Phase total | SA | Non-SA |
| --- | ---: | ---: | ---: | ---: |
| ep-full-cpu | 90.30% | 353.004 | 341.725 | 11.279 |
| ep-full-cuda | 89.06% | 453.329 | 386.931 | 66.398 |
| bp-full-cpu | 96.20% | 0.201 | 0.000 | 0.201 |
| bp-full-cuda | 96.23% | 0.564 | 0.000 | 0.564 |

See `comparison.csv` and `comparison.json` for separate training, evaluation
and test totals in seconds. These runs used different execution times and
the GPU run used an archived harness; they do not establish a controlled
CPU/GPU speedup. Non-SA time includes transfers, matrix construction, Python
and optimizer work, not only GPU kernels. Both EP runs made 520000 SA calls.
CPU wall time was about 5h53m11s including startup/output, distinct from the
sum of measured phases. Published evidence excludes checkpoints and failed attempts.
`published_sha256` verifies sanitized published JSON; `runs` retains original
artifact hashes, which may differ because local identifiers were removed.

## PR-head-based candidate

Current code validation is `validation.json`, based directly on PR #1247 head
`ed53937`. `validation_main_candidate.json` is preserved historical validation
from the earlier main-integrated candidate. The PR baseline pins SDK 1.3.1;
the historical SDK 1.4 license gaps do not expand this increment's dependency scope.
Experimental results retain the executed source snapshots, not the current
candidate's commit attribution. No skill repository is bundled or required at runtime.
