# Modified: defer the full QVAE pipeline until explicitly requested.
"""
Downstream tasks using QVAE latent features.

This module contains classifiers and feature extraction utilities
that operate on the latent representations learned by QVAE models.
"""

from .classifier import MLPClassifier
# from .feature_extractor import FeatureExtractor

__all__ = [
    "MLPClassifier",
    # "FeatureExtractor",
    "get_full_pipeline"
]


def __getattr__(name):
    if name == "get_full_pipeline":
        from .pipeline import get_full_pipeline

        globals()[name] = get_full_pipeline
        return get_full_pipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__))

# Optional: if you have more downstream tasks (e.g., clustering, regression),
# import them here and add to __all__