# EP candidate validation

This cleanup is based on PR #1247 commit `e108eb3`. It retains Kaiwu 1.3.1,
PyTorch 2.7.0 and NumPy 2.2.6 with Python 3.10. Model code and the training CLI
are unchanged; the batch experiment dispatcher and its six tests were removed.

| Local suite | Before cleanup | After cleanup |
| --- | --- | --- |
| CPU | 105 passed, 6 skipped | 99 passed, 6 skipped |
| CUDA | 101 passed, 2 skipped, 8 failed | 95 passed, 2 skipped, 8 failed |

All eight CUDA failure IDs and messages match the previous candidate and original
PR baseline. They concern BM/RBM/QVAE/energy-model tests, not new EP regressions.
No failing test was removed. CPU skips CUDA checks and native SDK statistics;
CUDA skips native SDK statistics. Whole-repository CUDA success is not claimed.

Strict English/Chinese Sphinx builds pass; source pylint is 10/10. Preserved
reference JSON hashes match the earlier commit. Curve CSV values match the CPU
EP/BP summaries; phase timing identities were checked. Full training was not rerun.

After installing this checkout and the example dependencies, confirm
`kaiwu.torch_plugin.__file__` points to the intended checkout, then run:

```sh
python -m pytest tests -q -rs
python -m pylint src/
python -m sphinx -W --keep-going -b html -D language=en docs runs/docs-en
python -m sphinx -W --keep-going -b html -D language=zh_CN docs runs/docs-zh
```

Initialize the SDK license locally for real-SA tests. Skills can assist setup
but are not runtime dependencies or credential providers. Local Windows checks
do not establish remote Linux CI success, SDK 1.4 compatibility or CIM validation.
Historical single-seed experiments and executed source links are described in
[reference results](../example/ep_mnist/reference_results/README.md).
