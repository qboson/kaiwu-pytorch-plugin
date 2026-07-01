"""
Downstream tasks using QVAE latent features.

This module contains classifiers and feature extraction utilities
that operate on the latent representations learned by QVAE models.
"""

from .classifier import MLPClassifier
# from .feature_extractor import FeatureExtractor
from .pipeline import get_full_pipeline

__all__ = [
    "MLPClassifier",
    # "FeatureExtractor",
    "get_full_pipeline"
]

# Optional: if you have more downstream tasks (e.g., clustering, regression),
# import them here and add to __all__