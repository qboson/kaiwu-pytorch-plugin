# EP merge-candidate validation

Validated locally on 2026-11-27 after merging PR #1247 head `5396704` with
upstream `main` at `393a5f1`. Merge commit `e26bf07` identifies the runtime
source tested in the historical comparison below. Commit `0178090` then updated
the closeout documentation. The subsequent CI repair changes test setup and
workflow coverage, not the plugin runtime or retained training results.
The upstream 0.4.0 version, dependency ranges and six default CI combinations
are retained.

## CI repair and acceptance

The second attempt of [upstream run 45954647458](https://github.com/qboson/kaiwu-pytorch-plugin/actions/runs/45954647458)
tested `0178090`: each of six jobs reported 216 passed, 2 failed and 25 skipped.
Both failures were real SDK solver tests attempting interactive license input.
The nine EP MNIST entrypoint tests were skipped because torchvision was absent.

The repaired workflow has two complementary execution paths:

- Upstream PR jobs select `-m 'not kaiwu_license'`. These jobs do not receive
  fork credentials. Their success alone does not verify the real solver.
- A push to `feat/ep-energy-qubo` runs the complete six-combination matrix in
  the fork, including both real solver tests. `KAIWU_REQUIRE_LICENSE=1` requires
  explicit `KAIWU_USER_ID` and `KAIWU_SDK_CODE` repository Actions secrets.
  Initialization uses `kaiwu.license.init` inside the test process. Missing or
  rejected credentials fail the run; interactive prompts are prohibited.

Configure the two secrets in the fork's **Settings > Secrets and variables >
Actions > New repository secret**, not in Variables. Values come from the Kaiwu
platform SDK page. Never commit credentials or license files. The branch push
trigger works without updating the fork's older default-branch workflow.
Re-run a failed **push** run after correcting credentials. A manual dispatch on
`feat/ep-energy-qubo` must set `run_licensed_tests=true`; its default is the
ordinary, unlicensed mode and does not count as licensed acceptance evidence.
For example: `gh workflow run ci.yml --repo yuke2027/kaiwu-pytorch-plugin
--ref feat/ep-energy-qubo -f scope=default -f run_licensed_tests=true` (one line).

The Python 3.10 / Kaiwu 1.4.4 / PyTorch 2.7.0 Linux lint combination installs
CPU torchvision 0.22.0 plus the EP example requirements. `pip check`, import
checks and version assertions run after installation. Other matrix combinations
retain their existing dependencies; optional-example skips outside this baseline
remain visible with `-rs`.

Each job uploads JUnit and environment metadata even when tests fail. The result
checker requires both real solver test IDs to pass in every licensed job and all
nine EP MNIST test IDs to pass in the example baseline. Missing, skipped,
duplicated, failed and errored acceptance cases fail verification. Solver energy
and trajectory assertions are unchanged. Artifact metadata records the tested
SHA, versions and validation mode, never secret values.

Before merging, require all six ordinary PR jobs and all six licensed fork jobs
to pass. The fork run's head SHA must equal the PR head SHA; the ordinary PR run
may test GitHub's synthetic merge commit. Link both runs in the PR and retain
their reports. A new PR head requires fresh licensed evidence. Maintainer review
and upstream required checks still apply; a green fork run does not bypass them.

Local repair checks on Windows / Python 3.10.11 / Kaiwu 1.3.1 / pytest 8.3.5:
46 targeted tests passed, including both real solver cases and all nine EP MNIST
cases; the CPU full suite passed 256 tests, skipped 16 CUDA-only cases, and
deselected the same two SDK statistics cases excluded from closeout acceptance.
The 22 new regression cases cover license prerequisites, safe error reporting,
and rejection of incomplete JUnit evidence. The targeted and full-suite JUnit
reports pass the 11-case solver-plus-MNIST acceptance checker. These local results
do not replace the pending Linux / SDK 1.4.x / Python 3.12 and 3.14 CI rerun.
The CUDA acceptance subset plus the new CI regression cases passed all 52 tests
without skips. The unlicensed targeted run passed 44 tests and deselected exactly
the two licensed cases, even when credentials were required but absent. Conversely,
selecting the two licensed cases with missing credentials produced the expected
two setup errors without prompting. Actionlint 1.7.12 and `pip check` passed.
Installing pytest 9.1.1 locally was blocked by a PyPI TLS connection error; its
behavior remains part of the remote CI verification rather than a local claim.

## Historical environment and results

Windows, Python 3.10.11, Kaiwu 1.3.1, NumPy 2.2.6 and pytest 8.3.5 were used.
CPU: PyTorch 2.7.0 and torchvision 0.22.0. CUDA: PyTorch 2.7.0+cu126,
torchvision 0.22.0+cu126 and an NVIDIA GeForce RTX 4060 Laptop GPU.
Separate local virtual environments reused existing dependency installations;
each contained its own SDK package and a wheel built from the intended checkout.
Installed plugin source hashes were checked against that checkout. Both candidate
environments passed `pip check`.

Example dependencies were installed before the final runs: imageio 2.37.0,
torchmetrics 1.9.0, seaborn 0.13.2, pandas 2.2.3, tqdm 4.67.1 and
lightning-utilities 0.15.3. No final-suite skip is caused by a missing dependency.

| Checkout / suite | Passed | Failed | Skipped | Deselected |
| --- | ---: | ---: | ---: | ---: |
| Candidate, CPU full suite | 234 | 0 | 16 | 2 |
| Candidate, CUDA full suite | 241 | 9 | 0 | 2 |
| PR head `5396704`, CUDA full suite | 95 | 8 | 0 | 2 |
| Upstream `393a5f1`, CUDA full suite | 213 | 9 | 0 | 2 |
| Candidate, EP and device-migration CUDA checks | 30 | 0 | 0 | 0 |

The 16 CPU skips require CUDA: two BM migration checks, two EP parity checks,
four MNIST device-placement checks and eight Q-Diffusion resampling checks.
All execute in the CUDA full suite; the existing MNIST assertion described below
fails there. The two deselections are native SDK caller-context statistics tests,
which are outside this closeout acceptance scope. No test was deleted or changed
to suppress a failure.
The full suites also report one non-failing Matplotlib warning because an existing
QVAE example calls `show()` with the non-interactive Agg backend.

## CUDA baseline comparison

The candidate and upstream have the same nine failing test IDs and identical
JUnit failure messages. Eight also match the original PR head. The additional
upstream MNIST test was not present in that PR head and was skipped in the first
candidate run until its example dependencies were installed. It is not an EP
regression. Whole-repository CUDA success is **not** claimed.

| Test file / class | Failing methods | Observed error |
| --- | --- | --- |
| `test_bm.py::TestBoltzmannMachine` | `test_condition_sample`, `test_gibbs_sample` | CPU/CUDA tensor mismatch |
| `test_bm.py::TestBoltzmannMachine` | `test_device_compatibility` | `cuda:0` versus `cuda` device assertion |
| `test_energy_model.py::TestEnergyModel` | `test_score_visible_logits_aggregates_sampler_solutions`, `test_score_visible_logits_shape_and_stats` | CPU/CUDA tensor mismatch |
| `test_qvae.py::TestQVAE` | `test_energy_uses_centered_input_and_optional_loss_type`, `test_mse_forward_and_loss` | CPU/CUDA tensor mismatch |
| `test_rbm.py::TestRestrictedBoltzmannMachine` | `test_get_ising_matrix` | CPU/CUDA tensor mismatch |
| `test_mnist_device_placement.py::TestTrainerModelPlacement` | `test_bm_matches_trainer_when_use_cuda_flag_is_true` | `cuda:0` versus `cuda` device assertion; also fails on upstream |

These pre-existing failures remain visible for maintainer review. They were not
suppressed or expanded into unrelated model changes in this closeout.

## Reproduce the checks

Install the intended checkout and the optional example dependencies into the
selected environment. Verify `kaiwu.torch_plugin.__file__` and compare the installed
source with the checkout before testing; running pytest in the repository alone
does not guarantee that the intended package is imported. Initialize the SDK
license locally for real-SA tests. Run from the repository root with a fresh
writable output directory:

```sh
python -m pip check
python -I -m pytest tests -q -rs --strict-markers -p no:cacheprovider \
  --deselect=tests/test_usage_stats.py::TestKppCallerContext::test_nested_context_does_not_corrupt \
  --deselect=tests/test_usage_stats.py::TestKppCallerContext::test_custom_source_applied \
  --basetemp=runs/verify-tmp --junitxml=runs/verify.xml
```

Run that command separately in CPU and CUDA environments. For the 30-case CUDA
acceptance subset, require a real GPU operation first, then run:

```sh
python -c "import torch; assert torch.cuda.is_available(); assert torch.tensor([1., 2.], device='cuda').square().sum().item() == 5."
python -I -m pytest -q -rs \
  tests/test_equilibrium_propagation.py tests/test_ep_cuda.py \
  tests/test_ep_device_timing.py tests/test_ep_mnist.py \
  tests/test_bm_device_sync.py::TestDeviceSyncAcrossAccelerators
```

All 30 cases must pass with no skips. They cover exact EP states and gradients,
linear/convolutional CPU-CUDA parity, real-SA checks, synthetic MNIST checkpoint
evaluation, timing contracts and BM device migration. Synthetic smoke tests do
not rerun full MNIST training.

## Documentation and retained experiments

Pylint **4.1.4**, as pinned upstream, scored **10.00/10**. English and Chinese
Sphinx 8.1.3 strict builds completed with zero warnings:

```sh
python -m pylint src/
python -m sphinx -W --keep-going -b html -D language=en docs runs/docs-en
python -m sphinx -W --keep-going -b html -D language=zh_CN docs runs/docs-zh
```

The merged tutorial hierarchy retains architecture and case-study entries, with
EP appearing once in the practice-tutorial toctree. Chinese translations and
rendered navigation were checked. Three upstream `qvae_anomaly` literal includes
referenced source files absent from this revision; the tutorial now states that
its implementation is unavailable in this checkout and directs readers to the
runnable QVAE MNIST example, without attempting those includes.

All five retained result-file hashes and four archived executed-source hashes
match provenance. Regenerated curve CSV rows match the published data; the figure
was inspected. Accuracy and phase timing identities were rechecked: CPU EP
90.30%, CUDA EP 89.06%, CPU BP 96.20%. These remain historical single-seed,
three-epoch runs; full training was not repeated. See
[reference results](../example/ep_mnist/reference_results/README.md).

The local handoff retains commands, versions, source hashes, JUnit XML, failure
comparisons, build logs and artifact checks under its `evidence/` directory,
outside the plugin repository. Relevant folders: `candidate-complete-cpu`,
`candidate-complete-cuda`, `baseline-pr-cuda`, `baseline-main-cuda`,
`candidate-ep-cuda`, `candidate-docs`, `candidate-lint` and `artifacts`.
The file `baseline-comparison.json` records the failure comparison.

Local Windows/Kaiwu 1.3.1 results do not establish Linux CI, Python 3.12/3.14,
SDK 1.4.x or CIM success. The upstream CI matrix and review requirements remain
unchanged and must be evaluated on the eventual pushed commit before merging.
