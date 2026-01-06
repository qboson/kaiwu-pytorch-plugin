# Copyright (C) 2022-2025 Beijing QBoson Quantum Technology Co., Ltd.
#
# SPDX-License-Identifier: Apache-2.0

import time
import math
import torch
import torch.multiprocessing as mp
from torch.optim.lr_scheduler import LambdaLR

import kaiwu as kw
from kaiwu.torch_plugin import BoltzmannMachine
import matplotlib.pyplot as plt


class CosineScheduleWithWarmup(LambdaLR):
    """带有warmup的余弦退火学习率调度器"""

    def __init__(
        self,
        optimizer,
        num_warmup_steps,
        num_training_steps,
        num_cycles=0.5,
        last_epoch=-1,
    ):
        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps
        self.num_cycles = num_cycles
        super(CosineScheduleWithWarmup, self).__init__(
            optimizer, self.lr_lambda, last_epoch
        )

    def lr_lambda(self, current_step):
        """带有warmup的cosine schedule学习率生成器"""
        if current_step < self.num_warmup_steps:
            return float(current_step) / float(max(1, self.num_warmup_steps))
        descent_steps = float(max(1, self.num_training_steps - self.num_warmup_steps))
        progress = float(current_step - self.num_warmup_steps) / descent_steps
        return max(
            0.0,
            0.5 * (1.0 + math.cos(math.pi * float(self.num_cycles) * 2.0 * progress)),
        )


def process_solve_graph_worker(params):
    """子进程执行的具体采样逻辑。

    Args:
        params (tuple): 包含共享模型、采样器、可见层数据分片、输出维度、起始索引的元组。

    Returns:
        tuple: (state_v_chunk, state_vi_chunk) 采样后的张量结果。
    """
    bm_shared, sampler, s_visible_chunk, num_output = params
    kw.common.set_log_level("ERROR")
    # 在不计算梯度的情况下进行采样
    with torch.no_grad():
        # 正相位采样 (Condition Sample)
        state_v_chunk = bm_shared.condition_sample(sampler, s_visible_chunk).detach()
        # 输入条件采样
        state_vi_chunk = bm_shared.condition_sample(
            sampler, s_visible_chunk[:, :-num_output]
        ).detach()

    return state_v_chunk, state_vi_chunk


class Trainer:
    """玻尔兹曼机模型训练器。

    该类负责协调玻尔兹曼机模型的训练过程，包括数据分批处理、多进程并发采样、
    损失函数计算以及模型参数更新。

    Attributes:
        data (DataLoader or list): 训练数据集。
        saver (Saver): 用于保存模型权重和训练日志的对象。
        worker (Sampler): 采样求解器对象（如 QA 采样器或模拟退火采样器）。
        num_visible (int): 可见层节点数量。
        num_hidden (int): 隐藏层节点数量。
        num_output (int): 输出层节点数量（通常是可见层的一部分）。
        bm_net (BoltzmannMachine): 玻尔兹曼机网络模型。
        cost_param (dict): 包含 loss 惩罚系数 alpha 和 beta 的字典。
        learning_parameters (dict): 包含学习率、权重衰减和动量的字典。
    """

    def __init__(
        self, data, saver, worker, num_visible=100, num_hidden=10, num_output=10
    ):
        """初始化 Trainer。

        Args:
            data (list): 训练数据列表，每个元素应为 torch.Tensor。
            saver (object): 具备 save_info 和 output_loss 方法的持久化对象。
            worker (object): 执行采样的后端 worker。
            num_visible (int): 可见层维数。默认为 100。
            num_hidden (int): 隐藏层维数。默认为 10。
            num_output (int): 输出层维数。默认为 10。
        """
        self.data = data
        self.saver = saver
        self.worker = worker
        self.num_visible = num_visible
        self.num_hidden = num_hidden
        self.num_output = num_output

        self.learning_parameters = {
            "learning_rate": 0.001,
            "weight_decay_rate": 0.0,
            "momentum_rate": 0.0,
        }
        self.bm_net = BoltzmannMachine(num_nodes=self.num_visible + self.num_hidden)

        # 核心：将模型参数移动到共享内存中，供多进程直接访问
        if self.bm_net.device.type != "cpu":
            self.bm_net_cpu = BoltzmannMachine(
                num_nodes=self.num_visible + self.num_hidden
            ).to(torch.device("cpu"))
        else:
            self.bm_net_cpu = self.bm_net
        self.bm_net_cpu.share_memory()

        self.cost_param = {"alpha": 0.5, "beta": 0.5}

    def set_cost_parameter(self, alpha, beta):
        """设置损失函数的惩罚系数。

        Args:
            alpha (float): KL 散度的权重系数。
            beta (float): 其他惩罚项（如 NCL）的权重系数。
        """
        self.cost_param = {"alpha": alpha, "beta": beta}

    def _sync_gpu_to_cpu(self):
        """将主模型的权重原地拷贝到 CPU 共享副本中。
        在需要大量并行采样且同时需要使用gpu时候可以参考这种用法
        """
        if self.bm_net.device.type == "cpu":
            return
        with torch.no_grad():
            for cpu_p, gpu_p in zip(
                self.bm_net_cpu.parameters(), self.bm_net.parameters()
            ):
                cpu_p.copy_(gpu_p)

    def set_learning_parameters(self, learning_rate, weight_decay_rate, momentum_rate):
        """设置训练相关的超参数。

        Args:
            learning_rate (float): 学习率。
            weight_decay_rate (float): 权重衰减率 (L2 正则化)。
            momentum_rate (float): 动量系数。
        """
        self.learning_parameters = {
            "learning_rate": learning_rate,
            "weight_decay_rate": weight_decay_rate,
            "momentum_rate": momentum_rate,
        }

    def train(self, max_steps, save_path, num_processes=1):
        """执行模型训练主循环。

        通过多进程并行采样并结合 Adam 优化器更新玻尔兹曼机参数。

        Args:
            max_steps (int): 最大训练步数。
            save_path (str): 模型和日志的保存路径。
            num_processes (int): 并行采样的进程数。默认为 1。
        """
        optimizer = torch.optim.Adam(
            self.bm_net.parameters(),
            lr=self.learning_parameters["learning_rate"],
            weight_decay=self.learning_parameters["weight_decay_rate"],
        )
        scheduler = CosineScheduleWithWarmup(
            optimizer,
            num_training_steps=max_steps,
            num_warmup_steps=int(max_steps / 20),
            num_cycles=0.5,
        )

        t_start = time.time()
        step = 0
        self.saver.save_info(self.bm_net, save_path, 0, 0.0)

        # 预先分配 Pool 以减少重复创建开销
        pool = mp.Pool(processes=num_processes)

        while step < max_steps:
            for batch_data in self.data:
                if step >= max_steps:
                    break

                optimizer.zero_grad()
                step += 1

                # 1. 负相位采样 (从当前模型分布采样)
                # 这个步骤通常对整个模型进行，直接由主进程执行
                state_all = self.bm_net.sample(self.worker).detach()

                # 2. 多进程并行正相位采样 (共享存储模式)
                # 将本批次数据设为共享内存
                self._sync_gpu_to_cpu()
                cpu_data = batch_data.cpu().share_memory_()

                # 按进程数切分数据
                chunks = torch.chunk(cpu_data, num_processes)

                # 准备子进程参数 (传递共享的模型对象)
                sd_args = [
                    (self.bm_net_cpu, self.worker, chunk, self.num_output)
                    for chunk in chunks
                    if chunk.size(0) > 0
                ]

                # 并行执行采样
                all_results = pool.map(process_solve_graph_worker, sd_args)

                # 3. 汇总结果并计算 Loss
                kl_divergence = torch.tensor(0.0, device=state_all.device)
                ncl = torch.tensor(0.0, device=state_all.device)

                # 收集所有进程的采样状态
                combined_v = torch.cat([res[0] for res in all_results], dim=0).to(
                    self.bm_net.device
                )
                combined_vi = torch.cat([res[1] for res in all_results], dim=0).to(
                    self.bm_net.device
                )

                # 计算 KL 散度项
                kl_divergence = self.bm_net.objective(combined_v, state_all)

                # 计算 NCL (Non-Contrastive Loss) 类似项
                # 保持原逻辑：将输出部分置零后计算 objective
                v_ncl = combined_v.clone()
                vi_ncl = combined_vi.clone()
                v_ncl[:, -self.num_output :] = 0.0
                vi_ncl[:, -self.num_output :] = 0.0
                ncl = self.bm_net.objective(v_ncl, vi_ncl)

                # 组合目标函数
                obj = (
                    self.cost_param["alpha"] * kl_divergence
                    + (1 - self.cost_param["alpha"]) * ncl
                )

                # 4. 反向传播与优化
                obj.backward()

                # 可视化 (保留原逻辑)
                if step % 10 == 0:
                    plt.figure(figsize=(8, 4))
                    plt.subplot(1, 2, 1)
                    plt.imshow(self.bm_net.quadratic_coef.detach().cpu().numpy())
                    plt.title("Weights")
                    plt.subplot(1, 2, 2)
                    plt.imshow(self.bm_net.quadratic_coef.grad.cpu().numpy())
                    plt.title("Gradients")
                    plt.show()

                optimizer.step()
                scheduler.step()

                # 输出与保存
                self.saver.output_loss(
                    step, kl_divergence.item(), ncl.item(), obj.item()
                )

                if step % 10 == 0:
                    t_now = time.time()
                    self.saver.save_info(self.bm_net, save_path, step, t_now - t_start)

        pool.close()
        pool.join()
