"""Every documented public name must be exported and importable."""

import kaiwu.torch_plugin as plugin

EXPECTED = {
    "RestrictedBoltzmannMachine",
    "BoltzmannMachine",
    "EnergyModel",
    "QVAE",
    "UnsupervisedDBN",
    "QDEQ",
    "QDEQConfig",
    "QDiffusion",
    "QDiffusionConfig",
    "FeatureSelectionWrapper",
    "QuadraticLinearSolver",
    "enable_usage_stats",
    "disable_usage_stats",
    "is_usage_stats_enabled",
}


def test_expected_names_are_exported():
    assert EXPECTED <= set(plugin.__all__)


def test_every_exported_name_exists_once():
    assert len(plugin.__all__) == len(set(plugin.__all__))
    for name in plugin.__all__:
        assert hasattr(plugin, name), name
