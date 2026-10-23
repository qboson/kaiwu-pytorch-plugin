# Q-Diffusion for Protein Sequence Generation

> **Status**: Coming soon — placeholder page; will be expanded into a full tutorial.

Discrete diffusion generation for proteins with the generic `Q-Diffusion` core (`src/kaiwu/torch_plugin/qdiffusion.py`) and a DPLM backbone: load a UniProt proteome FASTA, train with the `objective(...)` interface, and generate sequences with `generate(...)`.

**Examples**: `example/qdiffusion/simple/simple_train_example.py` · `example/qdiffusion/simple/simple_generate_example.py`

**DPLM adaptation**: `example/qdiffusion/dplm/` · **Data**: UniProt proteome UP000005640

## Gradients through decode scores

With a trainable proposal (`freeze_proposal=False`), a custom loop using
`initialize_state(...)` and `step(...)` can backpropagate through the
selected proposal log-probability scores in `state["output_scores"]`.
This also works when repetition resampling is enabled. Decode updates
preserve the token tensors saved by the proposal for backward.

For example, after constructing `model` with a trainable proposal:

```python
state = model.initialize_state(tokens, max_steps=2)
for _ in range(state["max_steps"]):
    state = model.step(state)
scores = state["output_scores"]
loss = -scores[torch.isfinite(scores)].sum()
loss.backward()
```

Token IDs and the categorical sampling, reranking and remasking decisions
remain discrete. This score gradient does not differentiate those decisions
or provide a full gradient through sequence generation. Intermediate masked
scores can be negative infinity; select finite scores when forming a loss.
Calls under `torch.no_grad()` keep their existing inference behavior.
