import torch
import torch.nn as nn
import torch.nn.functional as F
from kaiwu.torch_plugin import QVAE


class QVAEEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, normalization_method="layer"):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        if normalization_method == "layer":
            self.norm = nn.LayerNorm(hidden_dim)
        elif normalization_method == "batch":
            self.norm = nn.BatchNorm1d(hidden_dim)
        else:
            raise ValueError("normalization_method must be 'layer' or 'batch'")
        self.dropout = nn.Dropout(0.1)
        self.fc2 = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        h = self.fc1(x)
        h = self.norm(h)
        h = F.relu(h)
        h = self.dropout(h)
        return self.fc2(h)


class QVAEDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim, normalization_method="layer"):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        if normalization_method == "layer":
            self.norm = nn.LayerNorm(hidden_dim)
        elif normalization_method == "batch":
            self.norm = nn.BatchNorm1d(hidden_dim)
        else:
            raise ValueError("normalization_method must be 'layer' or 'batch'")
        self.dropout = nn.Dropout(0.1)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, zeta):
        h = self.fc1(zeta)
        h = self.norm(h)
        h = F.relu(h)
        h = self.dropout(h)
        return self.fc2(h)


class CellQVAE(QVAE):
    def __init__(self, *args, n_batches=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_batches = n_batches

    def _decoder_input(self, zeta, batch_idx=None):
        if self.n_batches <= 0:
            return zeta
        if batch_idx is None:
            raise ValueError("batch_idx is required when n_batches > 0")
        batch_one_hot = F.one_hot(batch_idx, num_classes=self.n_batches).float().to(zeta.device)
        return torch.cat([zeta, batch_one_hot], dim=-1)

    def encode_for_loss(self, x, loss_type):
        encoder_x = x if loss_type == "mse" else x - self.train_bias
        q = self.encoder(encoder_x)
        posterior, zeta = self.posterior(q, self.dist_beta)
        return q, posterior, zeta

    def mse_elbo(self, x, batch_idx, kl_beta):
        q, posterior, zeta = self.encode_for_loss(x, "mse")
        recon_x = self.decoder(self._decoder_input(zeta, batch_idx))
        recon_loss = F.mse_loss(recon_x, x, reduction="sum") / x.size(0)
        kl = self._kl_dist_from(posterior).mean()
        return recon_loss + kl_beta * kl, kl, recon_loss, q

    def bernoulli_elbo(self, x, batch_idx, kl_beta):
        # encoder_x = x - self.train_bias
        q, posterior, zeta = self.encode_for_loss(x, "bernoulli")
        recon_x = self.decoder(self._decoder_input(zeta, batch_idx)) + self.train_bias
        output_dist = self._output_dist(recon_x)
        kl = self._kl_dist_from(posterior).mean()
        recon_loss = -output_dist.log_prob_per_var(x).sum(dim=1).mean()
        return recon_loss + kl_beta * kl, kl, recon_loss, q

    def _output_dist(self, recon_x):
        from kaiwu.torch_plugin.qvae_dist_util import FactorialBernoulliUtil

        return FactorialBernoulliUtil(recon_x)

    def cell_loss(self, x, batch_idx, loss_type, kl_beta):
        if loss_type == "mse":
            return self.mse_elbo(x, batch_idx, kl_beta)
        return self.bernoulli_elbo(x, batch_idx, kl_beta)

    def bm_phase_loss(self, q, bm_weight_decay=0.0):
        positive_state = (q.detach() > 0).float()
        loss = self.bm.objective(positive_state, self.bm.sample(self.sampler))
        loss = self.bm.objective(torch.sigmoid(q.detach()), self.bm.sample(self.sampler))
        if bm_weight_decay > 0:
            loss = loss + bm_weight_decay * (
                torch.sum(self.bm.quadratic_coef**2)
                + torch.sum(self.bm.linear_bias**2)
            )
        return loss

    def energy(self, x, loss_type):
        q, _, _ = self.encode_for_loss(x, loss_type)
        return self.bm((q > 0).float())
