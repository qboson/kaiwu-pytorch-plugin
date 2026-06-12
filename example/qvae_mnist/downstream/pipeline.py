# -*- coding: utf-8 -*-
"""
Pipeline module for QVAE training and feature extraction with MLP classifier.

This module provides a sklearn Pipeline that trains a QVAE, extracts features,
and trains an MLPClassifier on the extracted features.
"""

from datetime import datetime
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from model import Config
from trainer import Trainer
from downstream import MLPClassifier
# from .feature_extractor import FeatureExtractor

def get_full_pipeline(config):
    """
    Return a sklearn Pipeline that trains a QVAE + extracts features,
    then trains an MLPClassifier on the extracted features.

    Args:
        config (Config): Configuration object containing all parameters.

    Returns:
        Pipeline: Scikit-learn pipeline with QVAE transformer and classifier.
    """
    # Create output directory if not set
    if config.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config.output_dir = f"./output/pipeline_{timestamp}"

    # Create the transformer
    qvae_transformer = PipelineTransformer(config)

    # Create MLPClassifier (using config.classifier_kwargs if present)
    classifier_kwargs = getattr(config, 'classifier_kwargs', {})
    clf = MLPClassifier(save_path=config.output_dir, **classifier_kwargs)
    return Pipeline([('qvae', qvae_transformer), ('classifier', clf)])

class PipelineTransformer:
    """
    A transformer that trains a QVAE and extracts features using the trained model.

    Args:
        config (Config): Configuration object containing all parameters.
    """

    def __init__(self, config):
        self.config = config
        self.trainer = None
        self.extractor = None
        self.model = None

    def fit(self, X, y=None):
        """
        Train the QVAE model.

        Args:
            X (numpy.ndarray): Input features (images).
            y (numpy.ndarray, optional): Labels (ignored, for sklearn compatibility).

        Returns:
            self: The fitted transformer.
        """
        X_train, X_val = train_test_split(X, test_size=0.2, random_state=42)
        y_train = np.zeros(len(X_train), dtype=int)
        y_val = np.zeros(len(X_val), dtype=int)

        # Create Trainer with config - now passing config directly
        self.trainer = Trainer(
            config=self.config,
            custom_train_data=(X_train, y_train),
            custom_test_data=(X_val, y_val)
        )
        # Train QVAE
        self.model, _, _ = self.trainer.train(run_tsne=self.config.run_tsne)
        return self

    def transform(self, X):
        """
        Extract latent features from input data using trained QVAE.

        Args:
            X (numpy.ndarray): Input data.

        Returns:
            numpy.ndarray: Extracted features.
        """
        if self.trainer is None:
            raise ValueError("Model not trained. Call fit() first.")

        # Convert X to a DataLoader with dummy labels
        x_tensor = torch.FloatTensor(X)
        dummy_labels = torch.zeros(len(X), dtype=torch.long)
        dataset = TensorDataset(x_tensor, dummy_labels)
        loader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=False)

        # Extract features
        features, _ = self.trainer.extract_features(loader, feature_type=self.config.feature_type)
        return features.numpy()
