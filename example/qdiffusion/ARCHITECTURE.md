# `QDiffusion` Architecture

## Overview

`QDiffusion` is now a single public module at
`src/kaiwu/torch_plugin/qdiffusion.py`.

It combines:

- proposal-side token prediction
- energy-side candidate scoring
- DPLM backbone loading
- iterative skeptical remasking decode

## Public Entry Points

- `from kaiwu.torch_plugin import QDiffusion`
- `from kaiwu.torch_plugin import QDiffusionConfig`
- explicit `QDiffusion(...)` construction with token metadata and an energy adapter
- `build_dplm_qdiffusion(...)` inside `example/qdiffusion/dplm/dplm_factory.py`

## Example Boundary

Everything in `example/qdiffusion/` is workflow or demo code:

- `simple/` for minimal runnable demos
- `dplm/` for DPLM-specific adapters and larger workflows

- data reading
- training loops
- rerun/evaluation scripts
- architecture notes and figures

The examples should treat `QDiffusion` as installed library code.
