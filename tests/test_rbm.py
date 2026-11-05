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




class TestRBMConstructorContract(unittest.TestCase):
    """构造期形状校验：文档中的 linear_bias 布局与代码约定必须一致。"""

    def test_rejects_linear_bias_with_documented_but_wrong_shape(self):
        # 回归：旧文档声称 linear_bias 形状为 [num_hidden]，按文档传参会
        # 静默构造出首次 get_hidden/forward/_to_ising_matrix 即崩溃的模型。
        with self.assertRaises(ValueError):
            RBM(num_visible=6, num_hidden=4, linear_bias=torch.zeros(4))

    def test_rejects_linear_bias_with_wrong_total_length(self):
        with self.assertRaises(ValueError):
            RBM(num_visible=6, num_hidden=4, linear_bias=torch.zeros(11))

    def test_rejects_quadratic_coef_with_wrong_shape(self):
        with self.assertRaises(ValueError):
            RBM(num_visible=6, num_hidden=4, quadratic_coef=torch.zeros(6, 5))

    def test_accepts_concatenated_linear_bias_layout(self):
        visible_bias = torch.full((6,), -0.25)
        hidden_bias = torch.full((4,), 0.5)
        bm = RBM(num_visible=6, num_hidden=4,
                 linear_bias=torch.cat([visible_bias, hidden_bias]))
        s_visible = torch.rand(3, 6)
        hidden_probs = bm.get_hidden(s_visible)[:, 6:]
        expected = torch.sigmoid(s_visible @ bm.quadratic_coef + hidden_bias)
        torch.testing.assert_close(hidden_probs, expected)

    def test_ising_objective_optimum_matches_brute_force_energy(self):
        import itertools
        torch.manual_seed(0)
        bm = RBM(num_visible=3, num_hidden=4)
        with torch.no_grad():
            bm.quadratic_coef.copy_(torch.randn(3, 4))
            bm.linear_bias.copy_(torch.randn(7) * 0.5)
        states = torch.tensor(
            list(itertools.product([0.0, 1.0], repeat=7)), dtype=torch.float32)
        with torch.no_grad():
            energies = bm(states)
        ising = torch.tensor(bm.get_ising_matrix(), dtype=torch.float64)
        bits = torch.tensor(
            list(itertools.product([0.0, 1.0], repeat=7)), dtype=torch.float64)
        # 翻转全部自旋（含辅助自旋）不改变二次目标，固定 aux=+1 枚举即可，
        # 解码必须遵循 AbstractBoltzmannMachine.sample 的 aux-spin 约定。
        spins = torch.cat(
            [2 * bits - 1, torch.ones(bits.shape[0], 1, dtype=torch.float64)], dim=1)
        objective = torch.einsum("bi,ij,bj->b", spins, ising, spins)
        solved_bits = bits[objective.argmax()]
        solved_energy = bm(solved_bits.float().unsqueeze(0)).item()
        self.assertAlmostEqual(
            solved_energy, energies.min().item(), places=4,
            msg="Ising 最优解能量应等于暴力枚举的模型能量最小值")


if __name__ == "__main__":
    unittest.main()
