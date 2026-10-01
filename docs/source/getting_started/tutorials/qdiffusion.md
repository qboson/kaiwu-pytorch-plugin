# Q-Diffusion for Protein Sequence Generation

> **Status**: Coming soon — placeholder page; will be expanded into a full tutorial.

Discrete diffusion generation for proteins with the generic `Q-Diffusion` core (`src/kaiwu/torch_plugin/qdiffusion.py`) and a DPLM backbone: load a UniProt proteome FASTA, train with the `objective(...)` interface, and generate sequences with `generate(...)`.

**Examples**: `example/qdiffusion/simple/simple_train_example.py` · `example/qdiffusion/simple/simple_generate_example.py`

**DPLM adaptation**: `example/qdiffusion/dplm/` · **Data**: UniProt proteome UP000005640

## DPLM sequence features and mixed precision

The conditioned BM reranker encodes noisy and candidate sequences, then uses
`masked_mean_pool` to reduce each sequence to one feature vector. Float16 and
BF16 token states are accumulated in Float32 before the mean is cast back to the
original dtype. This prevents Float16 sums from overflowing when the final mean
is representable and avoids separately rounding BF16 sums and token counts.
Float32 and Float64 states retain their existing accumulation precision.

Nonfinite states at zero-weight padding positions are removed before multiplication.
An entirely padded sequence produces zero features;
nonfinite values at active positions still propagate. Numeric mask weights and
the existing denominator clamp to at least one are preserved. Gradients pass
through active states and are zero at excluded padding positions.

For example, this CPU calculation returns a finite Float16 sequence feature:

```python
import torch
from example.qdiffusion.dplm.models.common import masked_mean_pool

hidden = torch.full((1, 1000, 2), 1000., dtype=torch.float16)
mask = torch.ones(1, 1000, dtype=torch.bool)
pooled = masked_mean_pool(hidden, mask)
assert pooled.dtype == hidden.dtype
assert torch.equal(pooled, torch.full((1, 2), 1000., dtype=torch.float16))
```

Run from a source checkout with the DPLM example dependencies installed. This
pooling example does not load pretrained weights or require a solver license.
Float32 accumulation reduces these intermediate errors; the returned feature
still has the rounding and representable range of its original dtype.
