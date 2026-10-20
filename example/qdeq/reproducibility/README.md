# Historical experiment evidence and reproduction status

Audited 2026-09-10. This directory adds traceable evidence to the plugin release; it is **not yet a complete reproduction package for every paper result**. No new benchmark training was run for this audit.

## What is verified

`runs_96.csv` contains 27 historical runs: Weather, ECL and Traffic, each with Anderson, Ising FP32 and Ising W8A8, at seeds 2023, 42 and 2024. Every MSE/MAE pair matches both the server's aggregate result file and its corresponding complete individual log. The source filename, setting string, line number, log size and SHA-256 are recorded in the CSV, `recipe_manifest.json` and `log_evidence.json`. Raw logs are not redistributed.

From the repository root:

```bash
python reproducibility/summarize.py
```

This checks unique and complete seed coverage, checks agreement with the extracted log metrics, and recomputes means and **sample** standard deviations (`ddof=1`). All 18 mean/SD cells for these nine groups agree with Tables 2 and 3 of the supplied manuscript at four decimal places. This recomputes historical summaries; it does not rerun training or independently validate predictions.

`source_manifest.json` compares the 34 bundled iTransformer files against the current server snapshot. After formatting differences and replacement of private CIM project identifiers with `YOUR_CIM_PROJECT`, the files agree by normalized text or Python AST. The server snapshot is not an immutable record of the source used at every historical run. The public snapshot therefore remains a provenance limitation, even where recorded metrics agree.

## Recipes and prerequisites

Four historical scripts are provided in `recipes/`: Weather, ECL and Traffic multi-seed runs, and the ETTh1 formal FP32 run used as the starting point for plugin validation. ETTh1's script is not evidence for its full three-seed paper row.

The scripts preserve the historical model and solver arguments. Portability changes are limited to project/interpreter/GPU selection, an optional plugin entry point, fail-fast shell behavior and checks for required data/initial checkpoints. Use a fresh checkout/output location: the historical model IDs and checkpoint paths are retained.

1. Install the plugin from the repository root with `python -m pip install -e .`.
2. Use Python 3.8.20 and the directly observed package versions in `requirements-observed.txt` as the reference environment. This is not a transitive dependency lock or a guarantee that other platforms reproduce the same floating-point results. Obtain the licensed Kaiwu 1.0.33a0 SDK separately for the historical SA backend.
3. Obtain the required datasets and theta0 initialization checkpoints listed in `assets.json` and `recipe_manifest.json`. Their current server hashes are recorded; the files themselves are **not distributed**. Exact initial checkpoints or validated recipes for generating them are still required before these scripts form an end-to-end reproduction path.
4. On Linux, once these prerequisites are available, run from the repository root:

```bash
bash reproducibility/recipes/weather_multiseed.sh
bash reproducibility/recipes/ecl_multiseed.sh
bash reproducibility/recipes/traffic_multiseed_standalone.sh
bash reproducibility/recipes/etth1_formal.sh
```

Scripts select `run.py` by default, which is the historical implementation. For a new experiment using the generic plugin, set `ENTRYPOINT=run_qdeq_plugin.py`. The ETTh1 adapter has been validated as described in [the integration guide](../integrations/README.md); the other three datasets have not been rerun through the plugin. Do not relabel their historical measurements as plugin measurements.

Optional environment variables: `PROJECT_DIR` (an iTransformer checkout), `PYTHON` (interpreter executable), `CUDA_VISIBLE_DEVICES` (default 0), and `ENTRYPOINT` (`run.py` or `run_qdeq_plugin.py`). Each script fails before training when a dataset or theta0 checkpoint is absent. Shell syntax and missing-asset checks were validated; the adapted recipes have not been rerun to convergence.

## Configuration facts that matter

- These four recipes use eight basis directions and **10-bit solver coefficients**. `quant_bits=8` refers separately to weight/activation fake quantization. Weather uses a dynamic coefficient range; ECL and Traffic use range 5, while ETTh1 uses range 16.
- The historical iTransformer SA backend is preserved. QDEQ's standalone default Ising implementation is not substituted in these recipes.
- W8A8 uses floating-point fake quantization and straight-through gradients, not measured native INT8 kernel execution.
- The original test loader defaults to batch size 1. Overriding the evaluation batch size may drop the final partial batch. Record sample counts when comparing new results. The separate plugin equivalence check explicitly includes all 2,785 ETTh1 samples.
- Historical setting names have misleading abbreviated field labels inherited from `run.py`. Use script arguments for architecture and sequence lengths rather than interpreting these labels by position.

## Remaining coverage gaps

| Paper component | Current status |
| --- | --- |
| Weather/ECL/Traffic DEQ groups, horizon 96 | 27 run metrics and log hashes reconciled; recipes included; initial checkpoints not distributed |
| ETTh1/ETTh2 three-seed groups | Complete per-seed mapping not established in this audit; ETTh1 formal recipe alone is insufficient |
| Explicit iTransformer baseline groups | Complete baseline multi-seed recipes and source-to-row mapping still needed |
| Horizons 192 and 336 | Final selected runs, configs and figure-source data still need reconciliation; tuning runs must not be treated as final results |
| Parameter/storage measurements | Profiling/export scripts and the exact storage-accounting boundary still need packaging |
| CIM comparison | Exact checkpoint, evaluation subset, task settings and outputs still need reconciliation; no hardware tasks launched |
| WILLOW graph matching | Full experiment implementation/configuration/results still need packaging; `migration/` is only a regression fixture |
| End-to-end independent reproduction | Datasets, theta0/final checkpoints or validated regeneration recipes, and immutable historical code provenance remain incomplete |

The release `v0.1.0` remains the original plugin/integration release. These evidence additions are on the repository branch after that tag; pin the corresponding newer commit when citing them. No manuscript files are edited by this audit.
