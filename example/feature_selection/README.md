**Language Versions**: [中文](README_ZH.md) | [English](README.md)

### Quantum Feature Selection on Neural Networks

This example shows how `FeatureSelectionWrapper` selects the informative features of a model's input while training it, using the Kaiwu CIM solver on the quantum side. The datasets are synthetic with known signal features, so you can check the selected features against the ground truth.

Two entry scripts:

* `linear_regression_solvers.py`: a linear-regression dataset with a handful of true signal features, run through **three solvers** — `local_search`, `sa`, and `kaiwu_cim` — so you can compare the selected feature sets without any quantum access;
* `neural_network_kaiwu_cim.py`: the same idea on `TinyCNN`, `SimpleRNN`, and `SimpleLSTM` backbones with the `kaiwu_cim` solver.

Run:

```bash
# runs the local_search solver; the sa and kaiwu_cim solvers are
# skipped automatically when no license is configured
python example/feature_selection/linear_regression_solvers.py

# sa / kaiwu_cim paths — both go through the Kaiwu SDK and need a license
export LICENSE_USER_ID="<your-user-id>"
export LICENSE_SDK_CODE="<your-sdk-code>"
export KAIWU_PROJECT_NO="<your-project-no>"   # edit KAIWU_PROJECT_NO in the scripts, or leave the default
python example/feature_selection/linear_regression_solvers.py
python example/feature_selection/neural_network_kaiwu_cim.py
```

Each run prints, per model: the final loss, the accuracy (classification) or loss (regression), the true `signal_features`, and the `selected_features` found by the wrapper. On the synthetic data the selected set should recover most of the signal features.

**Files**:

| File | Purpose |
| --- | --- |
| `feature_selection_datasets.py` | Synthetic datasets with known signal features (CNN image tensors, sequences, linear regression) |
| `feature_selection_models.py` | `TinyCNN`, `SimpleRNN`, `SimpleLSTM` backbones |
| `linear_regression_solvers.py` | Feature selection on linear regression with `local_search` / `sa` / `kaiwu_cim` |
| `neural_network_kaiwu_cim.py` | Feature selection on CNN / RNN / LSTM with `kaiwu_cim` |
| `kaiwu_license.py` | Initializes the Kaiwu license from `LICENSE_USER_ID` / `LICENSE_SDK_CODE` |

**Dependencies**: none beyond the package requirements. The `sa` and `kaiwu_cim` solvers both go through the Kaiwu SDK and require a Kaiwu license ([installation guide](https://kaiwu-pytorch-plugin.readthedocs.io/en/latest/source/getting_started/installation.html), "Kaiwu SDK Configuration & License"); the `local_search` and `sa` solvers in `linear_regression_solvers.py` run without one.
