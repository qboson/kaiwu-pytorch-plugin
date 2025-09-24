"""构建模型的参数和工具"""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

class CSVDataset(Dataset):
    # def __init__(self, csv_file, n_rows):
    #     """
    #     初始化函数
    #     Args:
    #       csv_file: CSV 文件路径
    #       n_rows: 读取的行数
    #       chunksize: 每次读取的块大小
    #     """
    #     self.data = []
    #     # 读入文件的前100列, 存储的文件行列是反的
    #     self.data = pd.read_csv(csv_file,
    #                             usecols=range(n_rows)).to_numpy().T
    def __init__(self, data):
        """
        初始化函数
        :param data: 输入数据，形状为 (numcol, numrow)
        """
        self.data = data

    def __len__(self):
        """
        返回数据集的大小
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        根据索引获取数据
        :param idx: 索引
        :return: 数据
        """
        return torch.tensor(self.data[idx], dtype=torch.float32)
