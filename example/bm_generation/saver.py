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
        """保存模型等信息

        Args:
            model (Model): 构造模型相关参数
            bias_path (str): 一次项系数的路径
            interation_path (str): 二次项系数的路径
            t_run (float): 计算运行时间
        """
        save_path = os.path.join(save_path,f'rbm_model{output_i}.pth')
        print(f"time: {time}, save_path: {save_path}")
        torch.save(model, save_path)

    def output_loss(self, output_i, kl_div, ncl, func):
        """输出loss

        Args:
            model (Model): 构造模型相关参数
            pr_vi (float): graph_out_hidden相关概率
            pr_v (float): graph_hidden相关概率
            output_i (int): 计算步数
        """
        print(f"step:{output_i}, kl_div:{kl_div}, ncl:{ncl}, cost:{func}")
        with open(os.path.join(self.log_path, 'loss.txt'), 'a', encoding="utf8") as f:
            f.write(f"step:{output_i}, kl_div:{kl_div}, ncl:{ncl}, cost:{func}\n")
