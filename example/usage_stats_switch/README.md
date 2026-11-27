# Toggling KPP Usage Statistics

KPP reports anonymous usage statistics only from its own sampler call path. The
global switch is read with `is_usage_stats_enabled()`, changed at runtime with
`enable_usage_stats()` / `disable_usage_stats()`, or set before the process
starts through the `KPP_STATS_ENABLED` environment variable (`true`, `1`,
`yes`, `on` enable it; any other value disables it). A code-level setting takes
precedence over the environment variable.

This example prints the current state, toggles it, and shows the environment
variable in child processes.

## Contents

- `usage_stats_switch.py`: `state()`, `child_state()`, and a `main()` that
  walks through the runtime and environment-variable cases.

## Run

```bash
python example/usage_stats_switch/usage_stats_switch.py
```

Expected output:

```
current: enabled
after disable: disabled
after enable: enabled
child with KPP_STATS_ENABLED=true: enabled
child with KPP_STATS_ENABLED=false: disabled
usage-statistics switch OK
```

## Dependencies

Only the installed `kaiwu-torch-plugin` (which brings `torch`). No sampler,
dataset, checkpoint, or Kaiwu license is required.

## Contract checked by the example

- statistics are enabled by default when the environment variable is unset;
- `KPP_STATS_ENABLED=false` disables them in a fresh process;
- `KPP_STATS_ENABLED` accepts `true`, `1`, `yes`, and `on`;
- `enable_usage_stats()` / `disable_usage_stats()` override the environment at
  runtime.

The same invariants are pinned by
`tests/test_usage_stats_switch_example.py`.
