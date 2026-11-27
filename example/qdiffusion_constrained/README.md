# Constrained Q-Diffusion Generation

This example shows how `QDiffusion.generate(..., partial_masks=...)` fixes part
of a sequence while the remaining positions are decoded.

`partial_masks` is a boolean tensor with the same shape as the input tokens.
Every position marked `True` keeps its input token for the whole decode loop;
only the other non-special positions are proposed, reranked, and resampled.
Special tokens (BOS, EOS, PAD) and the mask token are never emitted at editable
positions.

## Contents

- `constrained_generate.py`: toy proposal (`PreferSevenProposal`), toy energy
  model (`PreferManySevens`), and a `main()` that runs one constrained
  generation.

## Run

```bash
python example/qdiffusion_constrained/constrained_generate.py
```

Expected output shape: a six-token sequence whose positions 2 and 3 still hold
the input tokens `5` and `6`, while the editable positions contain content
tokens such as `7`.

## Dependencies

Only `torch` and the installed `kaiwu-torch-plugin`. No checkpoint, dataset
download, or Kaiwu license is required.

## Contract checked by the example

- fixed positions are preserved exactly;
- editable positions never receive BOS/EOS/PAD/mask ids;
- the energy model is consulted through `score_conditioned` for reranking.

The same invariants are pinned by `tests/test_qdiffusion_constrained_example.py`.
