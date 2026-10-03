# Q-Diffusion for Protein Sequence Generation

> **Status**: Coming soon — placeholder page; will be expanded into a full tutorial.

Discrete diffusion generation for proteins with the generic `Q-Diffusion` core (`src/kaiwu/torch_plugin/qdiffusion.py`) and a DPLM backbone: load a UniProt proteome FASTA, train with the `objective(...)` interface, and generate sequences with `generate(...)`.

**Examples**: `example/qdiffusion/simple/simple_train_example.py` · `example/qdiffusion/simple/simple_generate_example.py`

**DPLM adaptation**: `example/qdiffusion/dplm/` · **Data**: UniProt proteome UP000005640

## Generation and gradient recording

`QDiffusion.generate(...)` runs the full decoding loop under `torch.no_grad()`.
This also applies when the proposal backbone is trainable and when
`return_state=True`: returned scores have no autograd history. Model modes and
parameter `requires_grad` flags are preserved; call `model.eval()` explicitly
when inference should disable training-only behavior such as dropout.

Training through `forward(...)` or `objective(...)` keeps its existing gradient
behavior. For a custom decoding loop that needs caller-controlled gradient
recording, use `initialize_state(...)` followed by `step(...)` directly.
