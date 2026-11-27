"""构建模型的参数和工具"""

import torch
from torch.utils.data import Dataset


class CSVDataset(Dataset):
    """
    数据集

    Args:
        data: 已加载的二维样本数据，形状为 (num_samples, num_features)。
            每行对应一个样本，每列对应一个特征。
    """

    def __init__(self, data):
        self.data = data

    def __len__(self):
        """
        返回数据集的大小
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        根据索引获取数据
        Args:
            idx: 索引

        Return:
            torch.Tensor: 数据
        """
        return torch.tensor(self.data[idx], dtype=torch.float32)
