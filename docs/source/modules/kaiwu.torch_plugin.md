# 模块手册

## Module contents

```{eval-rst}
.. automodule:: kaiwu.torch_plugin
   :members:
   :undoc-members:
   :show-inheritance:
```

## MAIFS CIM Checkpoint Records

`FeatureSelectionWrapper` accepts CIM checkpoint options through
`solver_kwargs`; the same options are accepted by `maifs.qubo.solve_qubo`.
For example, retain the records of a feature-selection job with:

```python
from torch import nn
from kaiwu.torch_plugin import FeatureSelectionWrapper

selector = FeatureSelectionWrapper(
    nn.Linear(4, 1),
    feature_dim=4,
    solver="kaiwu_cim",
    solver_kwargs={"save_dir": "checkpoints", "cleanup_records": False},
)
```

`save_dir` is a base directory. Each invocation writes its records to a unique
child directory, so existing files and records from other jobs remain intact.
When `save_dir` is omitted, the base is the current SDK checkpoint directory,
or `feature_selection_kaiwu_cim` in the system temporary directory if the SDK
has no checkpoint directory configured.

With the default `cleanup_records=True`, a successful solve, restoration and
spin validation remove only the current job directory. Set
`cleanup_records=False` to retain that directory. If optimizer construction,
solving, restoration or validation fails, including a restored solution with
invalid spins, the job's records remain available for inspection under either
setting. The prior SDK `CheckpointManager.save_dir` setting is restored after
success or failure.
