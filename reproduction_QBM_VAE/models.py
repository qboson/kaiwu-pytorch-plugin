import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# SDK Compatibility Layer
try:
    from kaiwu.torch_plugin import RestrictedBoltzmannMachine
except ImportError:
    print("[WARN] Kaiwu SDK not detected. Falling back to Dummy RBM implementation.")


    class RestrictedBoltzmannMachine(nn.Module):
        """Mock RBM for non-quantum environments."""

        def __init__(self, num_visible, num_hidden, **kwargs):
            super().__init__()
            self.v_bias = nn.Parameter(torch.zeros(num_visible))

        def energy(self, z):
            return -(z * self.v_bias).sum(dim=1)


class PeptideEncoder(nn.Module):
    """Maps high-dimensional mass spectra to latent logits."""

    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 2048)
        self.fc2 = nn.Linear(2048, hidden_dim)
        self.fc_logits = nn.Linear(hidden_dim, latent_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.dropout(x, 0.3)
        x = F.relu(self.norm1(self.fc2(x)))
        return self.fc_logits(x)


class PeptideDecoder(nn.Module):
    """Reconstructs peptide sequences from latent states using GRU."""

    def __init__(self, latent_dim, hidden_dim, vocab_size):
        super().__init__()
        self.latent_to_hidden = nn.Linear(latent_dim, hidden_dim)
        self.embedding = nn.Embedding(vocab_size, 128)
        self.gru = nn.GRU(128, hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, z, target_seq):
        hidden = self.latent_to_hidden(z).unsqueeze(0)
        # Teacher forcing: Use ground truth previous token as input
        dec_input = target_seq[:, :-1]
        embedded = self.embedding(dec_input)
        output, _ = self.gru(embedded, hidden)
        prediction = self.fc_out(output)
        return prediction


class PeptideQVAE(nn.Module):
    """
    Quantum-Bounded Boltzmann Machine Variational Autoencoder (QBM-VAE).
    Integrates a quantum-inspired energy-based prior into the VAE latent space.
    """

    def __init__(self, input_dim=24501, hidden_dim=512, latent_dim=64, vocab_size=30, kl_beta=0.001):
        super().__init__()
        self.latent_dim = latent_dim
        self.kl_beta = kl_beta

        self.encoder = PeptideEncoder(input_dim, hidden_dim, latent_dim)
        self.decoder = PeptideDecoder(latent_dim, hidden_dim, vocab_size)

        # Quantum Prior Initialization
        self.rbm = RestrictedBoltzmannMachine(
            num_visible=latent_dim,
            num_hidden=latent_dim,
            h_range=[-1, 1],
            j_range=[-1, 1]
        )
        self._debug_flag = False

    def reparameterize(self, logits):
        """Bernoulli sampling relaxation (Gumbel-Softmax equivalent strategy)."""
        if self.training:
            return F.gumbel_softmax(logits, tau=1.0, hard=False)
        else:
            return (torch.sigmoid(logits) > 0.5).float()

    def compute_energy_safe(self, z):
        """Wrapper to handle SDK parameter naming variations dynamically."""
        if hasattr(self.rbm, 'energy'):
            return self.rbm.energy(z)

        # Reflection-based parameter discovery for backward compatibility
        bias_param = None
        possible_names = ['v_bias', 'visible_bias', 'bias_v', 'bv', 'b_v']
        for name in possible_names:
            if hasattr(self.rbm, name):
                bias_param = getattr(self.rbm, name)
                break

        if bias_param is not None:
            return -(z * bias_param).sum(dim=1)

        if not self._debug_flag:
            print("[WARN] RBM parameter binding failed. Energy term set to zero (Dry-run mode).")
            self._debug_flag = True

        return torch.zeros(z.size(0), device=z.device)

    def forward(self, x, target_seq):
        logits = self.encoder(x)
        z = self.reparameterize(logits)
        seq_logits = self.decoder(z, target_seq)
        energy = self.compute_energy_safe(z)
        return seq_logits, z, logits, energy

    def compute_loss(self, seq_logits, target_seq, logits_z, rbm_energy):
        """
        Calculates the variational objective:
        Loss = Reconstruction_Error + beta * (Energy - Entropy)
        """
        target = target_seq[:, 1:]
        # Alignment check
        min_len = min(seq_logits.size(1), target.size(1))
        seq_logits = seq_logits[:, :min_len, :]
        target = target[:, :min_len]

        # 1. Reconstruction Loss (Cross Entropy)
        ce_loss = F.cross_entropy(
            seq_logits.reshape(-1, seq_logits.size(-1)),
            target.reshape(-1),
            ignore_index=0
        )

        # 2. Regularization (Variational Free Energy approximation)
        avg_energy = torch.mean(rbm_energy)
        probs = torch.sigmoid(logits_z)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8) +
                             (1 - probs) * torch.log(1 - probs + 1e-8), dim=1).mean()

        prior_loss = avg_energy - entropy
        total_loss = ce_loss + self.kl_beta * prior_loss

        return total_loss, ce_loss, prior_loss