# Q-Diffusion for Protein Sequence Generation

> **Status**: Coming soon — placeholder page; will be expanded into a full tutorial.

Discrete diffusion generation for proteins with the generic `Q-Diffusion` core (`src/kaiwu/torch_plugin/qdiffusion.py`) and a DPLM backbone: load a UniProt proteome FASTA, train with the `objective(...)` interface, and generate sequences with `generate(...)`.

**Examples**: `example/qdiffusion/simple/simple_train_example.py` · `example/qdiffusion/simple/simple_generate_example.py`

**DPLM adaptation**: `example/qdiffusion/dplm/` · **Data**: UniProt proteome UP000005640

## Candidate sequence context

Training negatives and generation candidates preserve the input sequence's
`BOS`, `EOS`, and `PAD` positions. Generation also preserves positions marked
`True` in `partial_masks`. Energy reranking receives these complete
reconstructions, with padding excluded from its attention mask.

Proposal sampling excludes special token ids from editable positions. The
training objective still returns the original, unfiltered proposal `logits`.
Repetition resampling only masks editable positions and keeps the fixed context
intact when calling the proposal model again.
