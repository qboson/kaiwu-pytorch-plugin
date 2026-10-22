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
            # E(s) = -s·b - 0.5·sᵀ·sym(Q)·s,其中 sym(Q) 取严格上三角及其转置。
            # 按 setUp 权重手工计算:ones=15, mones=27, pmones=-5, mpones=-9
            self.assertEqual(15.0, self.bm(self.ones).item())
            self.assertEqual(27.0, self.bm(self.mones).item())
            self.assertEqual(-5.0, self.bm(self.pmones).item())
            self.assertEqual(-9.0, self.bm(self.mpones).item())

        with self.subTest("测试批量输入能量逐样本对应"):
            batch = torch.vstack([self.ones, self.mones, self.pmones, self.mpones])
            self.assertListEqual(self.bm(batch).tolist(), [15.0, 27.0, -5.0, -9.0])

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
            # objective = E[s_positive].mean() - E[s_negative].mean()
            # ones/mones 能量为 15/27,差为 -12
            objective = self.bm.objective(self.ones, self.mones)
            self.assertEqual(-12.0, objective.item())

        with self.subTest("测试批量目标函数"):
            s1 = torch.vstack([self.ones, self.pmones])  # 均值 (15 + (-5)) / 2 = 5
            s2 = torch.vstack([self.mones, self.mpones])  # 均值 (27 + (-9)) / 2 = 9
            self.assertEqual(-4.0, self.bm.objective(s1, s2).item())

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
        with self.subTest("条件采样保持可见层并重采样隐层"):
            # 旧用例把全部 4 个节点都作为可见层传入,导致 Gibbs 循环内
            # 所有单元都被跳过、从未真正采样。这里可见层只覆盖前 2 个节点,
            # 并用极端偏置把隐层单元的条件概率固定为 1 和 0,使结果确定:
            # 可见层 [1, 0] 保持不变,隐层被重采样为 [1, 0]。
            bm = BM(self.num_nodes)
            bm.linear_bias.data = torch.FloatTensor([0.0, 0.0, 50.0, -50.0])
            bm.quadratic_coef.data = torch.zeros(self.num_nodes, self.num_nodes)
            s_visible = torch.tensor([[1.0, 0.0]])
            samples = bm.gibbs_sample(num_steps=5, s_visible=s_visible)
            self.assertEqual(samples.shape, (1, self.num_nodes))
            torch.testing.assert_close(
                samples, torch.tensor([[1.0, 0.0, 1.0, 0.0]])
            )
            # 输入的可见层不应被采样过程覆写
            torch.testing.assert_close(s_visible, torch.tensor([[1.0, 0.0]]))
            self.assertTrue(torch.all((samples == 0) | (samples == 1)))

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
