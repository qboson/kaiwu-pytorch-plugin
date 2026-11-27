# Feature Selection on a Non-Default Feature Axis

`FeatureSelectionWrapper` masks one axis of its input. The axis is chosen by
`input_feature_axis`, which defaults to `-1` for the usual `(batch, features)`
layout. Feature-first tensors such as `(features, batch)` need
`input_feature_axis=0`.

This example builds both layouts, installs a fixed mask that keeps features
`0, 2, 4`, and checks the masking contract.

## Contents

- `axis_masking.py`: `FeatureFirstModel`, a wrapper factory, the mask
  installation helper, and a `main()` that runs both layouts.

## Run

```bash
python example/feature_selection_axis/axis_masking.py
```

Expected output:

```
feature-first mask: [1, 0, 1, 0, 1, 0]
zeroed feature rows: [1, 3, 5]
forward output shape: (1, 8)
mismatched feature axis rejected: input feature axis has size 5, expected 6
batch-first (axis -1) zeroed feature columns: [1, 3, 5]
axis contract OK
```

## Dependencies

Only `torch` and the installed `kaiwu-torch-plugin`. No dataset, checkpoint, or
Kaiwu license is required.

## Contract checked by the example

- with `input_feature_axis=0`, exactly the unselected feature rows are zeroed;
- `forward` passes the masked tensor to the wrapped model;
- a feature axis whose size differs from `feature_dim` raises `ValueError`;
- the default `input_feature_axis=-1` zeroes feature columns instead.

The same invariants are pinned by `tests/test_feature_selection_axis_example.py`.
