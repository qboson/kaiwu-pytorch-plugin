import types
import unittest
import torch
import numpy as np

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
from kaiwu.torch_plugin.qvae import QVAE
from kaiwu.torch_plugin import RestrictedBoltzmannMachine


class DummyEncoder(torch.nn.Module):
    """模拟编码器：将输入 (B, D) 映射到潜在空间 (B, L)"""
    def __init__(self, latent_dim: int):
        super().__init__()
        self.latent_dim = latent_dim

    def forward(self, x):
        # x: (B, D) -> output: (B, L)
        return torch.randn(x.size(0), self.latent_dim, device=x.device, dtype=x.dtype)


class DummyDecoder(torch.nn.Module):
    """模拟解码器：从潜在空间 (B, L) 映射回 (B, D)"""
    def __init__(self, input_dim: int):
        super().__init__()
        self.input_dim = input_dim

    def forward(self, z):
        # z: (B, L) -> output: (B, D)
        return torch.randn(z.size(0), self.input_dim, device=z.device, dtype=z.dtype)

class DummySampler:
    def solve(self, ising_matrix):
        num_solution = 10
        return np.random.randint(0, 2, size=(num_solution, ising_matrix.shape[0]))


class DummyQVAE(QVAE):
    """测试用 QVAE 子类，直接返回预先传入的 encoder/decoder/bm/sampler"""
    def _create_encoder(self):
        return self.encoder

    def _create_decoder(self):
        return self.decoder

    def _create_bm(self, bm_type):
        return self.bm

    def _create_sampler(self, sampler_type):
        return self.sampler

class TestQVAE(unittest.TestCase):
    def setUp(self):
        # 输入数据维度 D
        self.input_dim = 8
        
        # 潜在空间维度 L（BM 的总单元数）
        self.latent_dim = 4

        # BM 维度分解：可见单元数 V，隐藏单元数 H，必须满足 V + H == L
        self.num_visible = self.latent_dim // 2   # V = 2
        self.num_hidden  = self.latent_dim - self.num_visible  # H = 2

        # 配置对象
        self.config = types.SimpleNamespace(
            num_latent_units=self.latent_dim,   # L
            loss_type='bernoulli',
            dist_beta=1.0,
            kl_beta=1e-6,
            weight_decay=0.01,
        )

        # 预创建模块
        self.encoder = DummyEncoder(self.latent_dim)  # 编码器: 输入 D，输出 L
        self.decoder = DummyDecoder(self.input_dim)  # 解码器: 输入 L，输出 D
        self.rbm = RestrictedBoltzmannMachine(self.num_visible, self.num_hidden)  # BM: 可见 V，隐藏 H，总单元数 L
        self.sampler = DummySampler()

        # 构造 QVAE
        self.qvae = DummyQVAE(
            input_dimension=self.input_dim,
            config=self.config,
            encoder=self.encoder,
            decoder=self.decoder,
            bm=self.rbm,
            sampler=self.sampler,
        )

        # 设置数据集均值（如需）
        self.mean_x = np.ones(self.input_dim) * 0.5
        self.qvae.set_dataset_mean(torch.tensor(self.mean_x, dtype=torch.float32))

    def test_forward(self):
        """测试 QVAE.forward 的维度传递"""
        batch_size = 2
        x = torch.randn(batch_size, self.input_dim)   # (B, D)
        recon_x, posterior, q, zeta = self.qvae.forward(x)

        self.assertEqual(recon_x.shape, x.shape)
        self.assertEqual(q.shape, (x.size(0), self.latent_dim))
        self.assertEqual(zeta.shape, (x.size(0), self.latent_dim))

    def test_loss(self):
        """测试 QVAE.loss 的输出类型和正值性"""
        batch_size = 2
        x = torch.randn(batch_size, self.input_dim)   # (B, D)
        recon_x, posterior, q, zeta = self.qvae.forward(x)

        # loss 内部会调用 _kl_dist_from -> _cross_entropy，
        # 其中 self.bm(q_prob) 要求 q_prob 最后一维为 L，且 L = V+H
        loss = self.qvae.loss(x, recon_x, posterior)
        self.assertIsInstance(loss.item(), float)
        self.assertGreater(loss.item(), 0.0)  # 损失应为正值


if __name__ == "__main__":
    unittest.main()
