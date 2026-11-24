# EP closeout validation

Run the commands from the plugin repository root with the checkout available to the selected Python interpreter. Use a fresh writable directory for each report. The closeout checks cover the CPU regression suite and the two CUDA device tests. The two native SDK statistics tests are outside this acceptance scope.

## CPU regression

Use Python 3.10 with Kaiwu 1.3.1 and PyTorch 2.7.0 CPU, the recorded experiment environment. From an environment with this checkout installed, run:

```powershell
python -B -m pytest tests -q -rs --strict-markers -p no:cacheprovider --deselect=tests/test_usage_stats.py::TestNativeCallerContext --basetemp=runs/verify-cpu-tmp --junitxml=runs/verify-cpu.xml
```

The CPU suite skips the two cases that require CUDA. Check those cases in the CUDA environment below.

## Real CUDA

Use a CUDA-enabled PyTorch interpreter with NumPy and pytest. The verified environment used an RTX 4060 Laptop GPU, driver 566.24, Python 3.13 and PyTorch 2.11.0+cu128. Both cases use a deterministic sampler to isolate device movement and scoring.

```powershell
python -B tools/verify_test_environment.py cuda --output-dir runs/verify-cuda
```

The verifier imports the plugin from this checkout, executes a GPU tensor operation and requires both cases to pass with zero skips. It records the environment, source hashes and JUnit report. The device checks cover a parent moving its BM from CUDA to CPU and producing finite CPU scores after the move.
