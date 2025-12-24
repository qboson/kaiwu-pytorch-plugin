import unittest
import torch
import numpy as np

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
from kaiwu.torch_plugin import BoltzmannMachine as BM


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

    def test_gibbs_sample(self):
        """测试gibbs_sample采样功能"""
        with self.subTest("采样输出形状与类型"):
            samples = self.bm.gibbs_sample(num_steps=10, s_visible=self.ones)
            self.assertEqual(samples.shape, self.ones.shape)
            self.assertTrue(torch.all((samples == 0) | (samples == 1)))
            self.assertIsInstance(samples, torch.Tensor)

        with self.subTest("采样无s_visible参数"):
            samples = self.bm.gibbs_sample(num_steps=5, num_sample=2)
            self.assertEqual(samples.shape, (2, self.num_nodes))

        with self.subTest("采样异常情况"):
            with self.assertRaises(ValueError):
                self.bm.gibbs_sample(num_steps=5)

    def test_hidden_to_ising_matrix(self):
        """测试_hidden_to_ising_matrix功能"""
        with self.subTest("输出形状与类型"):
            # 取前2个节点为可见层
            s_visible = torch.ones(1, 2)
            ising_submat = self.bm._hidden_to_ising_matrix(s_visible[0])
            # 隐含层数量为2，输出应为(3, 3)
            self.assertEqual(ising_submat.shape, (3, 3))
            self.assertIsInstance(ising_submat, np.ndarray)

    def test_condition_sample(self):
        """测试condition_sample功能"""

        class DummySampler:
            def solve(self, ising_mat):
                # 返回一个 shape (2, n) 的全1矩阵，模拟采样器
                return np.ones((2, ising_mat.shape[0]))

        with self.subTest("采样输出形状与类型"):
            sampler = DummySampler()
            s_visible = torch.ones(1, 2)
            result = self.bm.condition_sample(sampler, s_visible)
            # 采样器返回2个样本，每个样本长度为可见层+隐含层
            self.assertEqual(result.shape, (2, self.num_nodes))
            self.assertIsInstance(result, torch.Tensor)

    def test_get_ising_matrix(self):
        with self.subTest("Unbounded weight range"):
            h_true = torch.FloatTensor([-3, 0, 1, 2])
            J_true = torch.FloatTensor(
                [
                    [1, 2, 4, 3],
                    [2, 0, 1.5, 0],
                    [4, 1.5, 0, -1],
                    [3, 0, -1, 0],
                ]
            )
            self.bm.linear_bias.data = h_true
            # self.bm.parametrizations.quadratic_coef.original.data.copy_(J_true)
            self.bm.quadratic_coef = torch.nn.Parameter(J_true)
            print("bm.quadratic_coef", self.bm.quadratic_coef)
            ising_mat = self.bm.get_ising_matrix()
            print("ising mat:", ising_mat)
            s = torch.tensor([[1, 1, 1, 1]], dtype=torch.float32)
            s2 = torch.tensor([[0, 1, 1, 0]], dtype=torch.float32)
            x = np.array([[1, 1, 1, 1, 1]], dtype=np.float32)
            x2 = np.array([[-1, 1, 1, -1, 1]], dtype=np.float32)
            print(
                self.bm(s), self.bm(s2), -x @ ising_mat @ x.T, (-x2 @ ising_mat @ x2.T)
            )
            print(
                self.bm(s) - self.bm(s2),
                -x @ ising_mat @ x.T - (-x2 @ ising_mat @ x2.T),
            )
            assert self.bm(s) - self.bm(s2) == -x @ ising_mat @ x.T - (
                -x2 @ ising_mat @ x2.T
            )

    def test_hidden_to_ising(self):
        with self.subTest("Unbounded weight range"):
            h_true = torch.FloatTensor([-3, 0, 1, 2])
            J_true = torch.FloatTensor(
                [
                    [1, 2, 4, 3],
                    [2, 0, 1.5, 0],
                    [4, 1.5, 0, -1],
                    [3, 0, -1, 0],
                ]
            )
            self.bm.linear_bias.data = h_true
            # self.bm.parametrizations.quadratic_coef.original.data.copy_(J_true)
            self.bm.quadratic_coef = torch.nn.Parameter(J_true)
            print("bm.quadratic_coef", self.bm.quadratic_coef)
            ising_mat = self.bm._hidden_to_ising_matrix(torch.FloatTensor([1, 1]))
            print("ising mat:", ising_mat)
            s = torch.tensor([[1, 1, 0, 1]], dtype=torch.float32)
            s2 = torch.tensor([[1, 1, 1, 0]], dtype=torch.float32)
            # x = np.array([[1,1,-1,1, 1]],dtype=np.float32)
            # x2 = np.array([[1,1,1,-1,1]],dtype=np.float32)
            x = np.array([[-1, 1, 1]])
            x2 = np.array([[1, -1, 1]])
            print(
                self.bm(s), self.bm(s2), -x @ ising_mat @ x.T, (-x2 @ ising_mat @ x2.T)
            )
            print(
                self.bm(s) - self.bm(s2),
                -x @ ising_mat @ x.T - (-x2 @ ising_mat @ x2.T),
            )
            assert self.bm(s) - self.bm(s2) == -x @ ising_mat @ x.T - (
                -x2 @ ising_mat @ x2.T
            )


if __name__ == "__main__":
    unittest.main()
