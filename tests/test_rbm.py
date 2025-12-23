import unittest

import torch
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
from kaiwu.torch_plugin import RestrictedBoltzmannMachine as RBM


class TestRestrictedBoltzmannMachine(unittest.TestCase):
    def setUp(self) -> None:
        # Create a triangle graph with an additional dangling vertex
        self.num_visible = 2
        self.num_hidden = 2

        # Manually set the parameter weights for testing
        dtype = torch.float32
        self.ones = torch.ones(4).unsqueeze(0)
        self.mones = -torch.ones(4).unsqueeze(0)
        self.pmones = torch.tensor([[1, -1, 1, -1]], dtype=dtype)
        self.mpones = torch.tensor([[-1, 1, -1, 1]], dtype=dtype)
        self.sample_1 = torch.vstack([self.ones, self.ones, self.ones, self.pmones])
        self.sample_2 = torch.vstack([self.ones, self.ones, self.ones, self.mpones])

        bm = RBM(self.num_visible, self.num_hidden)
        bm.linear_bias.data = torch.FloatTensor([0.0, 1, 2, 3])
        bm.quadratic_coef.data = torch.FloatTensor(
            [
                [0.0, -1.0],
                [-1.0, 0.0],
            ]
        )
        self.bm = bm
        return super().setUp()

    def test_forward(self):
        with self.subTest("Manually-computed energies"):
            self.assertEqual(-4, self.bm(self.ones).item())
            self.assertEqual(8, self.bm(self.mones).item())
            self.assertEqual(0, self.bm(self.pmones).item())
            self.assertEqual(-4, self.bm(self.mpones).item())
            self.assertListEqual([-4, -4, -4, 0], self.bm(self.sample_1).tolist())

    def test_get_ising_matrix(self):
        with self.subTest("Unbounded weight range"):
            h_true = torch.FloatTensor([-3, 0, 1, 2])
            J_true = torch.FloatTensor(
                [
                    [0.0, -1.0],
                    [-1.0, 0.0],
                ]
            )
            self.bm.linear_bias.data = h_true
            self.bm.quadratic_coef.data = J_true
            ising_mat = self.bm.get_ising_matrix()
            s = torch.tensor([[1, 1, 1, 0]], dtype=torch.float32)
            s2 = torch.tensor([[0, 1, 1, 1]], dtype=torch.float32)
            x = np.array([[1, 1, 1, -1, 1]], dtype=np.float32)
            x2 = np.array([[-1, 1, 1, 1, 1]], dtype=np.float32)
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

    def test_register_forward_pre_hook(self):
        with self.subTest("forward"):
            h_true = torch.FloatTensor([-3, 0, 1, 2])
            J_true = torch.FloatTensor(
                [
                    [0.0, -1.0],
                    [-1.0, 0.0],
                ]
            )
            self.bm.linear_bias.data = h_true
            self.bm.quadratic_coef.data = J_true
            self.assertEqual(
                -(-1 * (-1 * -1) + -1 * (-1 * -1)) - (-3 + 0 + 1 + 2) * -1,
                self.bm(self.mones).item(),
            )

    def test_objective(self):
        s1 = self.sample_1
        s2 = self.sample_2
        s3 = torch.vstack([self.sample_2, self.sample_2])
        self.assertEqual(1, self.bm.objective(s1, s2).item())
        self.assertEqual(1, self.bm.objective(s1, s3))

    def test_get_hidden(self):
        # 输入可见层，输出应为(batch_size, num_visible + num_hidden)
        s_visible = torch.rand(5, self.num_visible)
        s_all = self.bm.get_hidden(s_visible)
        self.assertEqual(s_all.shape, (5, self.num_visible + self.num_hidden))
        # 检查隐藏层概率范围
        hidden_probs = s_all[:, self.num_visible :]
        self.assertTrue(torch.all((hidden_probs >= 0) & (hidden_probs <= 1)))

        # 测试requires_grad=True
        s_visible_grad = torch.rand(2, self.num_visible, requires_grad=True)
        s_all_grad = self.bm.get_hidden(s_visible_grad, requires_grad=True)
        self.assertTrue(s_all_grad.requires_grad)

    def test_get_visible(self):
        # 输入隐藏层，输出应为(batch_size, num_visible + num_hidden)
        s_hidden = torch.rand(6, self.num_hidden)
        s_all = self.bm.get_visible(s_hidden)
        self.assertEqual(s_all.shape, (6, self.num_visible + self.num_hidden))
        # 检查可见层概率范围
        visible_probs = s_all[:, : self.num_visible]
        self.assertTrue(torch.all((visible_probs >= 0) & (visible_probs <= 1)))
        # 输出不应允许梯度
        s_hidden_grad = torch.rand(3, self.num_hidden, requires_grad=True)
        s_all_nograd = self.bm.get_visible(s_hidden_grad)
        self.assertFalse(s_all_nograd.requires_grad)


if __name__ == "__main__":
    unittest.main()
