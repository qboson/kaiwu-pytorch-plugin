"""用于保存信息"""

import os
from datetime import datetime
import torch


class Saver:
    """用于保存信息"""

    def __init__(self, log_path="./log"):
        """初始化"""
        self.log_path = log_path
        if not os.path.exists(log_path):
            os.makedirs(log_path)

    def save_info(self, model, save_path, output_i, time):
        """将模型对象保存为 ``rbm_model<训练步数>.pth``，并打印耗时和保存路径。

        Args:
            model (torch.nn.Module): 要保存的模型对象。
            save_path (str or os.PathLike): 模型文件的保存目录。
            output_i (int): 用于文件名的训练步数。
            time (float): 已用时间，单位为秒。
        """
        save_path = os.path.join(save_path, f"rbm_model{output_i}.pth")
        print(f"time: {time}, save_path: {save_path}")
        torch.save(model, save_path)

    def output_loss(self, output_i, kl_div, ncl, func):
        """打印当前损失，并向日志目录下的 ``loss.txt`` 追加一条记录。

        Args:
            output_i (int): 当前训练步数。
            kl_div (float): 训练器记录的 KL 损失项。
            ncl (float): 将两个状态张量的 ``[:, -self.num_output:]`` 切片置零后
                计算的损失项。
            func (float): 训练器计算的总损失。
        """
        print(f"step:{output_i}, kl_div:{kl_div}, ncl:{ncl}, cost:{func}")
        with open(os.path.join(self.log_path, "loss.txt"), "a", encoding="utf8") as f:
            f.write(f"step:{output_i}, kl_div:{kl_div}, ncl:{ncl}, cost:{func}\n")
