import operator

import numpy as np
import torch

class FeatureExtractor:
    def __init__(self, qvae_model, feature_type='q', batch_size=256):
        """
        Args:
            qvae_model: 训练好的 QVAE 模型
            feature_type: 'q' 或 'zeta'
            batch_size: transform 每次前向的最大样本数（正整数）。
        """
        if isinstance(batch_size, bool):
            raise TypeError("batch_size must be a positive integer")
        try:
            batch_size = operator.index(batch_size)
        except TypeError as exc:
            raise TypeError("batch_size must be a positive integer") from exc
        if batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        self.batch_size = batch_size
        self.model = qvae_model
        self.feature_type = feature_type
        self.device = next(qvae_model.parameters()).device

    def extract(self, dataloader, return_labels=True):
        """提取特征和标签"""
        self.model.eval()
        features, labels = [], []
        with torch.no_grad():
            for data, label in dataloader:
                data = data.to(self.device)
                _, _, q, zeta = self.model(data)  # 返回四个值
                if self.feature_type == 'q':
                    feat = q.cpu()
                elif self.feature_type == 'zeta':
                    feat = zeta.cpu()
                else:
                    raise ValueError("feature_type must be 'q' or 'zeta'")
                features.append(feat)
                if return_labels:
                    labels.append(label)
        features = torch.cat(features, dim=0)
        if return_labels:
            labels = torch.cat(labels, dim=0)
            return features, labels
        return features

    def transform(self, X):
        """
        X: numpy array of shape (n_samples, input_dim) 或 torch Tensor
        返回: numpy array of shape (n_samples, latent_dim)

        按 batch_size 分批前向并保留行序。zeta 仍为随机采样；更改批大小
        可能改变固定随机种子下的采样序列。调用者持有的输入存储不在此处缩减。
        """
        self.model.eval()
        if isinstance(X, np.ndarray):
            X_tensor = torch.FloatTensor(X)
        else:
            X_tensor = X

        features = []
        with torch.no_grad():
            for start in range(0, len(X_tensor), self.batch_size):
                batch = X_tensor[start:start + self.batch_size].to(self.device)
                _, _, q, zeta = self.model(batch)
                feat = q if self.feature_type == 'q' else zeta
                features.append(feat.cpu().numpy())
        if not features:
            raise ValueError("X must contain at least one sample")
        return np.concatenate(features, axis=0)
