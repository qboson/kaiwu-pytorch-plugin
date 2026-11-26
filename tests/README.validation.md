# PR #1247 incremental candidate validation

Baseline: `ed53937dbf3f06bff0b15202d07e8456b80df192`, the remote PR head
checked on 2026-11-26. Candidate branch: `local/ep-pr1247-increment`, uncommitted.
No main merge, dependency change or unrelated model change is included.
The baseline pins Kaiwu 1.3.1, PyTorch 2.7.0 and NumPy 2.2.6 (Python 3.10).

| Suite | PR baseline | Incremental candidate |
| --- | --- | --- |
| CPU | 84 passed, 4 skipped | 105 passed, 6 skipped |
| CUDA | 78 passed, 2 skipped, 8 failed | 101 passed, 2 skipped, 8 failed |

All eight CUDA failure IDs and messages match the PR baseline. They concern
existing BM/RBM/QVAE/energy-model tests; no whole-repository CUDA pass is claimed.
CPU skips CUDA checks and native SDK statistics; CUDA skips native SDK statistics.
No tests were deselected. Strict English/Chinese Sphinx builds and source pylint
results are recorded in `../example/ep_mnist/closeout/device_runs/validation.json`.

## Reproduce

Install this checkout and the example dependencies into the intended Python 3.10
environment, and confirm `kaiwu.torch_plugin.__file__` belongs to this checkout.

```sh
python -m pytest tests -q -rs
python -m pylint src/
python -m sphinx -W --keep-going -b html -D language=en docs runs/docs-en
python -m sphinx -W --keep-going -b html -D language=zh_CN docs runs/docs-zh
```

Use the matching CPU or CUDA PyTorch wheel. Initialize the SDK license locally
for real-SA tests. Skills can assist environment setup but are not runtime
dependencies and do not supply credentials. This candidate does not expand the
SDK pin or claim SDK 1.4.x support.

## Experiment attribution

CPU full-data test accuracy is 90.30%; CUDA model / CPU SA is 89.06%, both
single-seed three-epoch results. Total/SA/non-SA timings and exact executed
sources are retained in `example/ep_mnist/closeout/device_runs/` and
`source_snapshots/`. They are earlier runs, not new training on this candidate.
The current entrypoint's import-isolation change is covered by regression tests.
The historical dispatcher and training hashes resolve to their named snapshots.

`validation_main_candidate.json` preserves the earlier main-integrated candidate
checks, including its SDK 1.4 license limitations. Those results and counts are
not this candidate's acceptance criteria. Local Windows checks do not claim
remote Linux CI success, CIM hardware validation or multi-seed accuracy.
