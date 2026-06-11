import numpy as np
import torch

class FeatureExtractor:
    def __init__(self, qvae_model, feature_type='q'):
        """
        Args:
            qvae_model: 训练好的 QVAE 模型
            feature_type: 'q' 或 'zeta'
        """
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
        """
        self.model.eval()
        if isinstance(X, np.ndarray):
            X_tensor = torch.FloatTensor(X).to(self.device)
        else:
            X_tensor = X.to(self.device)

        with torch.no_grad():
            # QVAE forward 返回: (recon_x, posterior, q, zeta)
            _, _, q, zeta = self.model(X_tensor)
            if self.feature_type == 'q':
                feat = q.cpu().numpy()
            else:
                feat = zeta.cpu().numpy()
        return feat