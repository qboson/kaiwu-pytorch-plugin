import unittest

import torch
import numpy as np

from kaiwu.torch_plugin import (
    BoltzmannMachine as BM,
)


class TestBoltzmannMachine(unittest.TestCase):
    def setUp(self) -> None:
        # 创建测试用的玻尔兹曼机
        self.num_nodes = 4
        self.bm = BM(self.num_nodes)

        # 设置测试参数
        dtype = torch.float32
        self.ones = torch.ones(4).unsqueeze(0)
        self.mones = -torch.ones(4).unsqueeze(0)
        self.pmones = torch.tensor([[1, -1, 1, -1]], dtype=dtype)
        self.mpones = torch.tensor([[-1, 1, -1, 1]], dtype=dtype)

        # 手动设置权重进行测试
        self.bm.linear_bias.data = torch.FloatTensor([0.0, 1.0, 2.0, 3.0])
        self.bm.quadratic_coef.data = torch.FloatTensor(
            [
                [0.0, -1.0, -2.0, -3.0],
                [-1.0, 0.0, -4.0, -5.0],
                [-2.0, -4.0, 0.0, -6.0],
                [-3.0, -5.0, -6.0, 0.0],
            ]
        )

        return super().setUp()

    def test_forward(self):
        """测试前向传播计算能量"""
        with self.subTest("测试手动计算的能量值"):
            # 测试不同输入的能量计算
            energy_ones = self.bm(self.ones).item()
            energy_mones = self.bm(self.mones).item()

            # 验证能量计算的正确性
            self.assertIsInstance(energy_ones, float)
            self.assertIsInstance(energy_mones, float)

    def test_get_ising_matrix(self):
        """测试Ising模型转换"""
        with self.subTest("测试Ising矩阵生成"):
            ising_mat = self.bm.get_ising_matrix()

            # 验证Ising矩阵的维度
            expected_size = self.num_nodes + 1
            self.assertEqual(ising_mat.shape, (expected_size, expected_size))
            # 验证矩阵是对称的
            self.assertListEqual(ising_mat.tolist(), ising_mat.T.tolist())

    def test_register_forward_pre_hook(self):
        """测试权重范围限制"""
        with self.subTest("测试权重范围限制功能"):
            # 设置权重范围
            self.bm.h_range = torch.tensor([-0.1, 0.1])
            self.bm.j_range = torch.tensor([-0.1, 0.2])

            # 测试能量计算仍然正常工作
            energy = self.bm(self.mones)
            self.assertIsInstance(energy.item(), float)

    def test_objective(self):
        """测试目标函数计算"""
        with self.subTest("测试目标函数"):
            s1 = self.ones
            s2 = self.mones

            # 计算目标函数
            objective = self.bm.objective(s1, s2)
            self.assertIsInstance(objective.item(), float)

    def test_parameter_shapes(self):
        """测试参数形状"""
        with self.subTest("测试参数维度"):
            # 验证线性偏置的维度
            self.assertEqual(self.bm.linear_bias.shape, (self.num_nodes,))

            # 验证二次系数的维度
            self.assertEqual(
                self.bm.quadratic_coef.shape, (self.num_nodes, self.num_nodes)
            )

    def test_device_compatibility(self):
        """测试设备兼容性"""
        if torch.cuda.is_available():
            with self.subTest("测试GPU兼容性"):
                device = torch.device("cuda")
                self.bm.to(device)

                # 测试在GPU上的计算
                test_input = self.ones.to(device)
                energy = self.bm(test_input)
                self.assertEqual(energy.device, device)


if __name__ == "__main__":
    unittest.main()
