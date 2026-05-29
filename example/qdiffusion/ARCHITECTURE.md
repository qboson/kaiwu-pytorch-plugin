# `example/qdiffusion` Architecture

This directory wires one generic `Q-Diffusion` core to one concrete experiment
stack:

- proposal side: DPLM
- energy side: conditioned RBM reranker

## Main Path

1. `simple/*.py` or `dplm/*.py` calls `build_dplm_qdiffusion(...)`
2. `dplm/dplm_builder.py` loads:
   - one DPLM proposal backbone
   - one DPLM feature encoder for the energy branch
3. `dplm/dplm_modeling.py` builds:
   - `DPLMFeatureEncoder`
   - `RBMConditionedEnergyModel`
   - `RBMConditionedEnergyAdapter`
4. the builder passes those parts into `kaiwu.torch_plugin.QDiffusion`
5. `Q-Diffusion` uses:
   - proposal logits for candidate generation
   - `energy_adapter.score_conditioned(...)` for candidate reranking

## Training Flow

`dplm/train_workflow.py` follows this chain:

1. read FASTA records
2. split train/validation/test sets
3. build one DPLM-proposal + RBM-reranker generator
4. tokenize sequences into `targets`
5. call `generator.objective({"targets": targets})`
6. inside `Q-Diffusion.objective(...)`:
   - corrupt clean targets into noisy tokens
   - run the DPLM proposal model
   - sample candidate reconstructions
   - score candidates with the RBM reranker
   - return `energy_objective`
7. optimize `energy_objective.mean()`
8. save compact checkpoints for the energy side

## Generation Flow

`Q-Diffusion.generate(...)` keeps the same outer API, but the reranking step now
uses the RBM path:

1. initialize a masked decode state
2. run the DPLM proposal model
3. sample candidate reconstructions
4. call `energy_adapter.score_conditioned(...)`
5. rerank candidates with RBM energies
6. update the decode state and continue

## Checkpoints

Example checkpoints store the guided energy branch explicitly:

- `energy_encoder`
- `feature_projector`
- `energy_rbm`
- `energy_head`
- `vocab_proj`

`energy_head` and `vocab_proj` remain in the checkpoint so the generic
`Q-Diffusion` module stays compatible with both the scorer-hook path and the
fallback hidden-state path.
