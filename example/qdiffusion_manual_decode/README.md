# Manual Q-Diffusion Decoding

`QDiffusion.generate` runs the whole decode loop internally. The same loop can
be driven from outside with `initialize_state` / `step`: each `step` call runs
one proposal/rerank/remask round, appends one frame to `state["history"]`, and
returns a new state, leaving the previous one untouched.

This example drives the loop manually, stops as soon as no mask token remains,
and checks that the manual result matches `generate` under the same seed.

## Contents

- `manual_decode.py`: `PreferSevenProposal`, `PreferManySevens`,
  `build_generator()`, `masked_positions()`, and a `main()` with the loop.

## Run

```bash
python example/qdiffusion_manual_decode/manual_decode.py
```

Expected output:

```
initial masked positions: 2
step 1: masked=1 history=2
step 2: masked=1 history=3
step 3: masked=1 history=4
step 4: masked=0 history=5
final tokens: [[0, 7, 5, 6, 7, 1]]
manual loop matches generate: True
manual decode OK
```

## Dependencies

Only `torch` and the installed `kaiwu-torch-plugin`. No checkpoint, dataset, or
Kaiwu license is required.

## Contract checked by the example

- `initialize_state` starts at step 0 with a one-frame history;
- each `step` increments `step`, appends one history frame, and returns a new
  state without mutating the input state;
- positions fixed by `partial_masks` keep their input tokens;
- enough steps resolve every mask token;
- the manual loop equals `generate` when both start from the same seed.

The same invariants are pinned by
`tests/test_qdiffusion_manual_decode_example.py`.
