# Q-Diffusion for Protein Sequence Generation

> **Status**: Coming soon — placeholder page; will be expanded into a full tutorial.

Discrete diffusion generation for proteins with the generic `Q-Diffusion` core (`src/kaiwu/torch_plugin/qdiffusion.py`) and a DPLM backbone: load a UniProt proteome FASTA, train with the `objective(...)` interface, and generate sequences with `generate(...)`.

**Examples**: `example/qdiffusion/simple/simple_train_example.py` · `example/qdiffusion/simple/simple_generate_example.py`

**DPLM adaptation**: `example/qdiffusion/dplm/` · **Data**: UniProt proteome UP000005640

## Decoding history and memory

`model.generate(tokens, max_steps=500)` returns the final token tensor and
does not retain intermediate token snapshots. Use `return_state=True` when
you need the complete state, including the initial tokens and one snapshot
per decoding step in `state["history"]`. These snapshots are taken before
skeptical remasking, as in the step-by-step decoding API.

For an external loop, history recording is enabled by default. To keep the
current scores, masks and tokens without retaining snapshots:

```python
state = model.initialize_state(tokens, max_steps=500, record_history=False)
for _ in range(state["max_steps"]):
    state = model.step(state)
final_tokens = state["output_tokens"]
assert state["history"] is None
```

Keeping history requires `(max_steps + 1) * batch_size * sequence_length`
token elements on the tokens' device, plus Python list storage. For int64
tokens with batch size 64, sequence length 1024 and 500 steps, the token
snapshots alone occupy about 250.5 MiB. Disabling history removes that
retained storage; proposal logits, scores and energy-model working memory
are separate costs. History recording does not change sampling or random
number consumption.
