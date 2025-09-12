import unittest
import torch
import numpy as np

from kaiwu.classical import SimulatedAnnealingOptimizer
from kaiwu.torch_plugin.qvae import QVAE
from kaiwu.torch_plugin import RestrictedBoltzmannMachine


class DummyEncoder(torch.nn.Module):
    def forward(self, x):
        return torch.randn_like(x)


class DummyDecoder(torch.nn.Module):
    def forward(self, z):
        return torch.randn_like(z)


class TestQVAE(unittest.TestCase):
    def setUp(self):
        self.num_var1 = 4
        self.num_var2 = 4
        self.encoder = DummyEncoder()
        self.decoder = DummyDecoder()
        self.rbm = RestrictedBoltzmannMachine(self.num_var1, self.num_var2)
        self.sampler = SimulatedAnnealingOptimizer()
        self.dist_beta = 1.0
        self.mean_x = np.ones(self.num_var1 + self.num_var2) * 0.5

        self.qvae = QVAE(
            encoder=self.encoder,
            decoder=self.decoder,
            rbm=self.rbm,
            sampler=self.sampler,
            dist_beta=self.dist_beta,
            mean_x=self.mean_x,
        )
        # 设置num_var1属性，供_cross_entropy使用
        self.qvae.num_var1 = self.num_var1

    def test_forward(self):
        x = torch.randn(2, self.num_var1 + self.num_var2)
        recon_x, posterior, q, zeta = self.qvae.forward(x)
        self.assertEqual(recon_x.shape, x.shape)
        self.assertEqual(q.shape, x.shape)
        self.assertEqual(zeta.shape, x.shape)

    def test_neg_elbo(self):
        x = torch.randn(2, self.num_var1 + self.num_var2)
        kl_beta = 1.0
        output, recon_x, neg_elbo, wd_loss, total_kl, cost, q, zeta = (
            self.qvae.neg_elbo(x, kl_beta)
        )
        self.assertEqual(output.shape, x.shape)
        self.assertEqual(recon_x.shape, x.shape)
        self.assertIsInstance(neg_elbo.item(), float)
        self.assertIsInstance(wd_loss.item(), float)
        self.assertIsInstance(total_kl.item(), float)
        self.assertIsInstance(cost.item(), float)
        self.assertEqual(q.shape, x.shape)
        self.assertEqual(zeta.shape, x.shape)


if __name__ == "__main__":
    unittest.main()
