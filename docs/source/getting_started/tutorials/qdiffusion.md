# Q-Diffusion for Protein Sequence Generation

> **Status**: Coming soon — placeholder page; will be expanded into a full tutorial.

Discrete diffusion generation for proteins with the generic `Q-Diffusion` core (`src/kaiwu/torch_plugin/qdiffusion.py`) and a DPLM backbone: load a UniProt proteome FASTA, train with the `objective(...)` interface, and generate sequences with `generate(...)`.

**Examples**: `example/qdiffusion/simple/simple_train_example.py` · `example/qdiffusion/simple/simple_generate_example.py`

**DPLM adaptation**: `example/qdiffusion/dplm/` · **Data**: UniProt proteome UP000005640


## Aligned quality inputs

The DPLM quality helpers compare records by position and require equal, nonzero record counts. Missing or extra records raise `ValueError` rather than silently scoring a shared prefix. Headers may differ between reference and generated sets; callers must align order before evaluation.
