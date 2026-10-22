# Q-Diffusion for Protein Sequence Generation

> **Status**: Coming soon — placeholder page; will be expanded into a full tutorial.

Discrete diffusion generation for proteins with the generic `Q-Diffusion` core (`src/kaiwu/torch_plugin/qdiffusion.py`) and a DPLM backbone: load a UniProt proteome FASTA, train with the `objective(...)` interface, and generate sequences with `generate(...)`.

**Examples**: `example/qdiffusion/simple/simple_train_example.py` · `example/qdiffusion/simple/simple_generate_example.py`

**DPLM adaptation**: `example/qdiffusion/dplm/` · **Data**: UniProt proteome UP000005640

## Proposal sampling controls

Set `QDiffusionConfig.proposal_noise_scale=0.0` to sample from the original
proposal logits without Gumbel perturbation. This setting skips Gumbel random
draws and supports half-precision logits during training.

`proposal_temperature=0.0` selects tokens greedily. A positive temperature
continues to draw categorical samples, including when the Gumbel noise scale
is zero. The single-candidate and multiple-candidate sampling paths use the
same temperature rules.
