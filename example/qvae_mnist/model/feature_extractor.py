# Modified: batch array/tensor feature extraction before device transfer.
import numpy as np
import torch

class FeatureExtractor:
    def __init__(self, qvae_model, feature_type='q', batch_size=256):
        """
        Args:
            qvae_model: 训练好的 QVAE 模型
            feature_type: 'q' 或 'zeta'
            batch_size: Positive integer maximum samples per transform forward.
        """
        if (isinstance(batch_size, (bool, np.bool_))
                or not isinstance(batch_size, (int, np.integer))
                or batch_size <= 0):
            raise ValueError("batch_size must be a positive integer")
        self.model = qvae_model
        self.feature_type = feature_type
        self.batch_size = int(batch_size)
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

        Inputs are sliced in order before transfer and forward. Features are
        collected on CPU; total input/output storage still scales with samples.
        Caller-owned GPU input storage is not released. Changing batch_size can
        change fixed-seed zeta samples by changing random-number draw order.
        """
        if len(X) == 0:
            raise ValueError("X must contain at least one sample")
        self.model.eval()
        features = []
        with torch.no_grad():
            for start in range(0, len(X), self.batch_size):
                batch = X[start:start + self.batch_size]
                if isinstance(batch, np.ndarray):
                    X_tensor = torch.FloatTensor(batch).to(self.device)
                else:
                    X_tensor = batch.to(self.device)
                # QVAE forward 返回: (recon_x, posterior, q, zeta)
                _, _, q, zeta = self.model(X_tensor)
                feat = q if self.feature_type == 'q' else zeta
                features.append(feat.cpu())
        return torch.cat(features, dim=0).numpy()
