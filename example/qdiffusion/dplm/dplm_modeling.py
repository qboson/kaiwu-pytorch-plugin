"""Compatibility exports for the DPLM example modeling stack.

The concrete implementation now lives under ``models/``:

- ``models/backbone.py``: DPLM configs, checkpoint loading, token spec helpers
- ``models/energy.py``: feature encoders plus RBM/BM reranker backends
- ``models/esm_patch.py``: private ESM runtime patches used by the backbone
"""

try:
    from .models.backbone import (
        DPLMBackbone,
        DPLMConfig,
        DPLMEnergyAdapter,
        DPLMLoRAConfig,
        DPLMNetConfig,
        build_dplm_token_spec,
        get_net,
        get_net_class,
        load_yaml_config,
    )
    from .models.energy import (
        BMConditionedEnergyAdapter,
        BMConditionedEnergyModel,
        DPLMFeatureEncoder,
        RBMConditionedEnergyAdapter,
        RBMConditionedEnergyModel,
    )
    from .models.esm_patch import _EsmForDPLM
except ImportError:  # pragma: no cover - direct script-path compatibility
    from models.backbone import (
        DPLMBackbone,
        DPLMConfig,
        DPLMEnergyAdapter,
        DPLMLoRAConfig,
        DPLMNetConfig,
        build_dplm_token_spec,
        get_net,
        get_net_class,
        load_yaml_config,
    )
    from models.energy import (
        BMConditionedEnergyAdapter,
        BMConditionedEnergyModel,
        DPLMFeatureEncoder,
        RBMConditionedEnergyAdapter,
        RBMConditionedEnergyModel,
    )
    from models.esm_patch import _EsmForDPLM

__all__ = [
    "BMConditionedEnergyAdapter",
    "BMConditionedEnergyModel",
    "DPLMBackbone",
    "DPLMConfig",
    "DPLMEnergyAdapter",
    "DPLMFeatureEncoder",
    "DPLMLoRAConfig",
    "DPLMNetConfig",
    "RBMConditionedEnergyAdapter",
    "RBMConditionedEnergyModel",
    "_EsmForDPLM",
    "build_dplm_token_spec",
    "get_net",
    "get_net_class",
    "load_yaml_config",
]
