# QDEQ

A reusable PyTorch Deep Equilibrium (DEQ) plugin with pluggable forward and backward solvers. The default is **Ising/simulated annealing forward + Anderson implicit backward**.

QDEQ separates model architecture, fixed-point solving, implicit differentiation and quantization orchestration. It does not implement native INT8 kernels: W8A8 in the integration refers to fake quantization with straight-through gradients.

## Install

```bash
git clone https://github.com/Renata1027/QDEQ.git
cd QDEQ
python -m pip install -e .
```

Python 3.8+; PyTorch and NumPy. The core plugin does not require Kaiwu or a GPU. The bundled production iTransformer Ising backend requires the separately obtained Kaiwu SDK; Anderson runs without it.

## Minimal example

```python
import torch
from qdeq import QDEQ

x = torch.randn(2, 3, 4, requires_grad=True)
deq = QDEQ()
z, info = deq(lambda z, x: 0.3 * z + x, torch.zeros_like(x), x)
z.square().mean().backward()
print(z.shape, x.grad.norm(), info["backward_nstep"])
```

`QDEQ(f_solver=None, b_solver=None, f_thres=None, b_thres=None, stop_mode="rel", solve_ctx=..., reattach_ctx=..., *, f_eps=None, b_eps=None)` accepts a callable `func(z, *args)` returning a tensor of the same shape. Solvers return a dictionary containing `result` and `nstep`. Omitted thresholds/tolerances preserve solver defaults. Anderson supports a batch axis followed by arbitrary feature axes.

Training solves without retaining the iteration graph, evaluates `new_z = func(z_star, *args)`, and solves `(I - J.T)y = grad` during backward. Numerical accuracy depends on both solves and the consistency of the map; implicit differentiation is not an unconditional guarantee of exact gradients. Higher-order differentiation through the implicit backward is unsupported.

If the callable or its bound owner implements `set_quant(bool)`, training disables it for solve and enables it for reattachment. `QDEQ.eval()` leaves quantization untouched and returns the raw fixed-point approximation. The wrapped model's mode and QDEQ's mode are independent. Disable/fix stochastic operations inside the fixed-point map; `no_grad()` alone does not disable dropout.

Diagnostics use `info["new_z"]` and `info["z_star"]` before backward. See `qdeq/diagnostics.py`. The Ising adapter retains its original stopping rule and does not implement `stop_mode="abs"`.

## Tests and validation

```bash
python -m unittest discover -s qdeq/tests -t . -v
python migration/qdeq_migration_regression.py --output validation/local_migration.json
```

- 23 unit tests, including nonlinear nonsymmetric gradient checks against unrolling.
- Three local decoder migration cases: aligned wrappers agree; aggregate gradient errors versus a direct local linear-system reference are below 0.3%.
- Real iTransformer/ETTh1 checkpoint on RTX 4090: Anderson, Ising FP32 and Ising W8A8 match the original outputs, gradients and three Adam updates.
- Complete 2,785-sample W8A8 inference comparison: predictions match exactly; both MSE values are **0.39415498213669153**.
- Original training pipeline with the plugin: one Anderson epoch (264 batches), checkpoint save/reload and complete test evaluation completed; MSE **0.393074631690979**, MAE **0.40563538670539856**.

These are integration-equivalence and pipeline checks, not multi-seed benchmark claims. Checkpoints, datasets and proprietary SDK binaries are not distributed. The checkpoint used for the full comparison was trained in FP32; fake quantization was enabled for the comparison. Full training recipes and limitations are described in [the integration guide](integrations/README.md). Machine-readable measurements are in `validation/`.

## Paper experiment reproducibility

The [reproducibility guide](reproducibility/README.md) records the current coverage and missing materials. It includes 27 historical Weather/ECL/Traffic runs checked against individual logs, a standard-library summary script, four portable historical recipes, observed dependency versions, and source/data/checkpoint hashes. Their nine DEQ groups agree with the manuscript's horizon-96 mean and sample-standard-deviation values. **This is not yet a complete end-to-end reproduction package for all paper tables and figures.** Initial checkpoints are not distributed; baseline, ETTh1/ETTh2 multi-seed, longer-horizon, CIM, storage and WILLOW coverage remains incomplete.

## Repository layout

- `qdeq/`: generic plugin and tests.
- `migration/`: regression and the minimal graph-matching decoder/solver fixture.
- `integrations/`: opt-in iTransformer adapter, CLI launcher and validation script.
- `experiments/itransformer/`: source snapshot from the actual validated project, with its original license.
- `validation/`: recorded local and remote measurements.
- `reproducibility/`: historical run evidence, recipes, provenance and explicit coverage gaps.

The iTransformer adapter preserves its tuned production forward solver and delegates implicit backward to QDEQ. It does not silently replace the production Ising algorithm with QDEQ's default implementation. Original train/eval and quantization behavior is preserved in the adapter.

## Citation

For a reproducible paper reference, cite the repository and pin release **v0.1.0** or its commit. GitHub's “Cite this repository” uses `CITATION.cff`. This repository is not itself a DOI or an archived paper dataset.

## License and attribution

MIT; see [LICENSE](LICENSE) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Upstream copyright notices are retained. Kaiwu is optional external software and is not covered by this license.
