"""训练器"""

# Copyright (C) 2022-2025 Beijing QBoson Quantum Technology Co., Ltd.
#
# SPDX-License-Identifier: Apache-2.0

import time
import math
import multiprocessing as mp
import torch
from torch.optim.lr_scheduler import LambdaLR

import numpy as np
from kaiwu.torch_plugin import BoltzmannMachine

# from torch_plugin import BoltzmannMachine
import matplotlib.pyplot as plt

# from graph import Graph


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


def process_solve_graph(params):

    bm, sampler, s_visible, num_output = params
    # print("s_visible0.shape:::", s_visible.shape)
    pr_vi = []
    n_visible = s_visible.size(1)
    state_v = []  # visible states
    state_vi = []  # visible input states
    for i in range(s_visible.size(0)):
        # print("s_visible1.shape:", s_visible.shape)
        state_v_i = bm.condition_sample(sampler, s_visible).detach().numpy()
        state_vi_i = (
            bm.condition_sample(sampler, s_visible[:, :-num_output]).detach().numpy()
        )
        state_v.append(state_v_i)
        state_vi.append(state_vi_i)
        # print("state_vi_i.shape:", state_vi_i.shape)
        # probs_i = np.ones((len(state_vi),)) / len(state_vi)
        # probs_i = np.ones((len(state_vi) + 1,)) / (len(state_vi) + 1)
        # 这里有问题
        s_visible_np = (
            s_visible[i : i + 1, n_visible - num_output : n_visible]
            .cpu()
            .numpy()
            .reshape([1, num_output])
        )
        err = np.abs(state_vi_i[:, n_visible - num_output : n_visible] - s_visible_np)
        # print("err shape:", err.shape)
        # print("err sum axis 0 shape:", np.sum(err, axis=0).shape)
        # print("probs_i shape:", probs_i.shape)
        probs_i = (
            np.ones(
                (np.sum(err, axis=0) <= 1e-3).shape[0],
            )
            / (np.sum(err, axis=0) <= 1e-3).shape[0]
        )
        pr_vi_i = np.sum((np.sum(err, axis=0) <= 1e-3) * probs_i)
        pr_vi.append(pr_vi_i)

    state_v = np.concatenate(state_v, axis=0)
    state_vi = np.concatenate(state_vi, axis=0)
    return state_v, state_vi, pr_vi


# def process_solve_graph(params):

#     bm, sampler, s_visible, num_output = params
#     print("s_visible0.shape:::", s_visible.shape)
#     pr_vi = []
#     n_visible = s_visible.size(1)
#     state_v = []  # visible states
#     state_vi = []  # visible input states
#     state_v_i = bm.condition_sample(sampler, s_visible).detach().numpy()
#     state_vi_i = bm.condition_sample(sampler, s_visible[:, :-num_output]).detach().numpy()


#     s_visible_np = (
#         s_visible[:, n_visible - num_output : n_visible]
#         .cpu()
#         .numpy()
#     )
#     err = np.abs(state_vi_i[:, n_visible - num_output : n_visible] - s_visible_np)
#     print("err shape:", err.shape)
#     print("err sum axis 0 shape:", np.sum(err, axis=0).shape)
#     # print("probs_i shape:", probs_i.shape)
#     probs_i = (
#         np.ones(
#             (np.sum(err, axis=0) <= 1e-3).shape[0],
#         )
#         / (np.sum(err, axis=0) <= 1e-3).shape[0]
#     )
#     pr_vi_i = np.sum((np.sum(err, axis=0) <= 1e-3) * probs_i)
#     pr_vi.append(pr_vi_i)

#     # state_v = np.concatenate(state_v, axis=0)
#     # state_vi = np.concatenate(state_vi, axis=0)
#     return state_v_i, state_vi_i, pr_vi


class Trainer:
    """模型训练器

    Args:
        name (str): 名称，用于输出
        model (Model): 构造模型相关参数
        saver (Saver): 用于保存信息的对象
        worker (Optimizer): 求解器
        connection (np.ndarray): 连通矩阵
    """

    def __init__(
        self, data, saver, worker, num_visible=100, num_hidden=10, num_output=10
    ):
        self.data = data
        self.saver = saver
        self.worker = worker
        self.num_visible = num_visible
        self.num_hidden = num_hidden
        self.num_output = num_output

        # 权重的最大值
        self.learning_parameters = {"Momentum": {"Lambda": None}}
        self.bm_net = BoltzmannMachine(num_nodes=self.num_visible + self.num_hidden)
        self.cost_param = {"alpha": 0.5, "beta": 0.5}

    def set_cost_parameter(self, alpha, beta):
        """设置loss惩罚系数"""
        self.cost_param = {"alpha": alpha, "beta": beta}

    def set_learning_parameters(self, learning_rate, weight_decay_rate, momentum_rate):
        """设置训练参数"""
        self.learning_parameters = {
            "learning_rate": learning_rate,
            "weight_decay_rate": weight_decay_rate,
            "momentum_rate": momentum_rate,
        }

    def train(self, max_steps, save_path, num_processes=1):
        """训练模型

        Args:
            max_steps (int): 训练步数
            init_weight (np.ndarray): 初始权重
            save_path (str): 保存路径
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

        i = 0
        probs = torch.ones((self.num_visible,)) / self.num_visible
        t_end = time.time()
        self.saver.save_info(self.bm_net, save_path, 0, t_end - t_start)
        while i < max_steps:
            for data in self.data:
                # print("data.shape", data.shape)
                # print(data)
                # zero_gradients
                optimizer.zero_grad()
                i += 1
                if i >= max_steps:
                    break

                state_all = self.bm_net.sample(self.worker)
                pr_v = []
                for j in range(data.size(0)):
                    data = data.to(state_all.device)
                    err = torch.abs(state_all[:, : self.num_visible] - data[j : j + 1])
                    probs = probs.to(state_all.device)
                    pr_v_i = torch.sum((torch.sum(err, dim=0) <= 1e-3) * probs)
                    pr_v.append(pr_v_i.item())
                pr_v = np.array(pr_v)

                k_ranges = np.array_split(range(data.size(0)), num_processes)
                # print("data[k_range].shape", data[k_ranges[0]].shape)
                # print("0---------------------------")
                with mp.Pool(processes=num_processes) as pool:

                    sd_args = [
                        (self.bm_net, self.worker, data[k_range], self.num_output)
                        for k_range in k_ranges
                    ]
                    # async_results = pool.map_async(process_solve_graph, sd_args)
                    # all_results = async_results.get()

                    all_results = []
                    for args_i in sd_args:
                        ret = process_solve_graph(args_i)
                        all_results.append(ret)
                # print("01---------------------------")
                kl_divergence = torch.tensor(0.0)
                ncl = torch.tensor(0.0)
                pr_v = []
                pr_vi = []
                for sub_list in all_results:
                    # print("1---------------------------")
                    state_v, state_vi, _ = sub_list

                    state_v = torch.FloatTensor(state_v)
                    state_vi = torch.FloatTensor(state_vi)
                    kl_divergence = kl_divergence + self.bm_net.objective(
                        state_v, state_all
                    ) / len(all_results)
                    state_v = state_v.clone()
                    state_vi = state_vi.clone()
                    state_v[:, -self.num_output :] = 0.0
                    state_vi[:, -self.num_output :] = 0.0
                    ncl = ncl + self.bm_net.objective(state_v, state_vi) / len(
                        all_results
                    )
                # print("2---------------------------")
                # state_v = self.bm_net.condition_sample(self.worker, data)
                # state_vi = self.bm_net.condition_sample(self.worker, data[:, :-self.num_output])
                # kl_divergence = self.bm_net.objective(state_v, state)
                # state_v = state_v.clone()
                # state_vi = state_vi.clone()
                # state_v[:, -self.num_output:] = 0.0
                # state_vi[:, -self.num_output:] = 0.0
                # ncl = self.bm_net.objective(state_v, state_vi)

                obj = (
                    self.cost_param["alpha"] * kl_divergence
                    + (1 - self.cost_param["alpha"]) * ncl
                )
                obj.backward()
                if i % 10 == 0:
                    # print("draw")
                    # print("i:",i)
                    plt.figure(figsize=(8, 4))
                    plt.subplot(1, 2, 1)
                    plt.imshow(self.bm_net.quadratic_coef.detach().numpy())
                    plt.subplot(1, 2, 2)
                    plt.imshow(self.bm_net.quadratic_coef.grad.numpy())
                    plt.show()

                optimizer.step()
                scheduler.step()
                self.saver.output_loss(i, kl_divergence, ncl, obj.item())

                if i % 10 == 0:
                    t_end = time.time()
                    self.saver.save_info(self.bm_net, save_path, i, t_end - t_start)
            # t_end = time.time()
            # self.saver.save_info(self.bm_net, save_path, i, t_end - t_start)
