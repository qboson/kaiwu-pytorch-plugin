# EP final-submission validation

This submission preserves the historical full-MNIST results and separately
checks the final code on small inputs. It does not claim a new full-data run,
a multi-seed full-data result, CIM hardware execution, or CUDA validation.

## Environment and tests

Use Python 3.10, Kaiwu 1.3.1, PyTorch 2.7.0, NumPy 2.2.6 and pytest 8.3.5.
Install the checkout and the MNIST example dependencies before testing:

```bash
python -m pip install -r requirements/devel.txt
python -m pip install -e .
python -m pip install -r example/ep_mnist/requirements.txt
python -c "import kaiwu.torch_plugin as p; print(p.__file__)"
python -m pytest tests -q -rs
python -m pylint src/ --disable=C0114,C0116
```

Confirm the printed module belongs to this checkout. Verification used the
existing SDK environment with this checkout explicitly prepended to the
`kaiwu` namespace; no other plugin checkout supplied the model code.
Final regression: **103 passed, 4 skipped**. The `91dd171` baseline was
**84 passed, 4 skipped**. Both runs skip the same two CUDA device tests and
two native SDK statistics-context tests. Those public tests are unchanged;
none was deselected. Source pylint reports **10.00/10**.

EP tests enumerate linear and convolutional states, compare the actual SDK
Hamiltonian with the original energy in both auxiliary-spin gauges, verify
gradients and candidate decoding, and exercise real Kaiwu annealing.
Synthetic MNIST tests cover saved-config restoration, validation-only data
access, invalid arguments, and output protection. Dispatch tests check
dry-run behavior and propagate training/report failures. `torchvision` is
an optional example dependency; install it to run the synthetic CLI tests.

## Real-SDK smoke checks

Run from the repository root with local MNIST data and fresh output paths:

```bash
python example/ep_mnist/train_mnist.py --data-dir data --output-dir runs/ep-smoke --validation-only --epochs 1 --seed 0 --split-seed 0 --limit-train 200 --limit-eval 100
python example/ep_mnist/train_mnist.py --data-dir data --evaluate runs/ep-smoke/checkpoint.pt --output-dir runs/ep-smoke-evaluation
python example/ep_mnist/train_mnist.py --data-dir data --output-dir runs/ep-conv-smoke --validation-only --epochs 1 --seed 0 --split-seed 0 --conv --channels 1 --limit-train 2 --limit-eval 2 --batch-size 2 --sa-iterations 5
python example/ep_mnist/train_mnist.py --data-dir data --evaluate runs/ep-conv-smoke/checkpoint.pt --output-dir runs/ep-conv-smoke-evaluation
```

Set `OMP_NUM_THREADS=1` and `MKL_NUM_THREADS=1` before launching Python to
match the recorded checks. Linear smoke training/validation accuracy was
0.16/0.12; convolutional smoke accuracy was 0.50/0.00 on two images each.
Restored training and validation metrics matched exactly in both runs.
The validation harness rejected any test-split load, and no test image was
accessed. These checks validate execution and restoration, not accuracy
on the full dataset. Checkpoints support evaluation, not training resume.

## Historical artifacts and documentation

The 33 hashes in the historical manifest (32 artifacts and the training
entrypoint) remain unchanged. The final entrypoint is byte-identical to its
recorded hash; historical provenance still says `93a536b` plus local changes.
Rebuilding all three CSVs from the raw JSON reproduces their published bytes.
PNG/PDF figures are checked for data and layout, without requiring identical
rendered bytes across different plotting environments.

```bash
python example/ep_mnist/closeout/summarize.py example/ep_mnist/closeout --output-dir runs/ep-figures
python -m sphinx -W --keep-going -b html -D language=en docs runs/docs-en
python -m sphinx -W --keep-going -b html -D language=zh_CN docs runs/docs-zh
```

New smoke evidence and source hashes are recorded separately in
`example/ep_mnist/closeout/final_validation.json`. They do not replace the
historical environment, results, or provenance manifest. GitHub CI status
is reported separately from these local checks.
