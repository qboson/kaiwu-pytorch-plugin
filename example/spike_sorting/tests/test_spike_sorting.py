"""Offline checks for the spike-sorting example.

These tests never download the Kaggle datasets and never initialise a Kaiwu
license: the synthetic signals and the tiny ``.ssd`` container are generated on
the fly, and the QBM-VAE check replaces the sampler with a local stand-in.
"""

import os
import sys

import numpy as np
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from metrics import evaluate_clustering  # noqa: E402
from models import (  # noqa: E402
    Autoencoder,
    GaussianVAE,
    SpikeQBMVAE,
    extract_features,
    train_qbm_vae,
)
from run_experiment import qbm_config  # noqa: E402
from spike_data import (  # noqa: E402
    PEAK_INDEX,
    extract_spikes,
    load_realdata,
    parse_ssd_descriptor,
)


class DummySampler:
    """Deterministic stand-in for ``SimulatedAnnealingOptimizer``."""

    def solve(self, ising_matrix):
        """Return random spins shaped ``(samples, nodes)`` like the real solver."""
        nodes = int(ising_matrix.shape[0])
        return np.random.default_rng(0).choice([-1, 1], size=(4, nodes)).astype(np.float32)


class DummyQBMVAE(SpikeQBMVAE):
    """QBM-VAE variant that skips the licensed sampler."""

    def _create_sampler(self, sampler_type=None):
        """Return the offline sampler."""
        return DummySampler()


def test_extract_spikes_aligns_peak():
    """The extracted waveforms must have their maximum on ``PEAK_INDEX``."""
    rng = np.random.default_rng(0)
    signal = rng.normal(0, 0.01, size=500)
    starts = np.array([50, 200, 350])
    for offset, start in enumerate(starts):
        signal[start + 5 + offset: start + 15 + offset] = np.array(
            [-1.0, -2.0, 3.0, 8.0, 4.0, 1.0, 0.0, -1.0, -0.5, 0.0]
        )

    spikes = extract_spikes(signal, starts, spike_length=79, align_to_peak=True)
    assert spikes.shape == (3, 79)
    assert np.all(np.argmax(spikes, axis=1) == PEAK_INDEX)


def test_extract_spikes_zscore():
    """Z-scored waveforms have zero mean and unit standard deviation."""
    signal = np.sin(np.linspace(0, 20 * np.pi, 400))
    spikes = extract_spikes(signal, np.array([10, 100, 200]), zscore=True)
    assert np.allclose(spikes.mean(axis=1), 0.0, atol=1e-5)
    assert np.allclose(spikes.std(axis=1), 1.0, atol=1e-5)


def test_parse_ssd_descriptor(tmp_path):
    """The SSD text descriptor exposes units, electrodes and spike counts."""
    descriptor = tmp_path / "recording.ssd"
    descriptor.write_text(
        "Spike Sorted Dataset (SSD)\n"
        "Format version:\n1.3\n\n"
        "Number of units:\n2\n\n"
        "List with the names of units:\nU01\nU02\n\n"
        "List with the recording elements (channels, tetrodes, etc) "
        "where each unit originates:\nEl_01\nEl_02\n\n"
        "Number of spikes in each unit:\n3\n2\n",
        encoding="utf-8",
    )
    meta = parse_ssd_descriptor(str(descriptor))
    assert meta["unit_names"] == ["U01", "U02"]
    assert meta["electrodes"] == ["El_01", "El_02"]
    assert meta["spike_counts"].tolist() == [3, 2]


def test_load_realdata_builds_labels(tmp_path):
    """Waveforms are grouped by unit and labelled from the descriptor counts."""
    length = 58
    counts = [3, 2]
    waveforms = np.arange(sum(counts) * length, dtype=np.float32).reshape(-1, length)
    waveforms.tofile(tmp_path / "recording.ssduw")
    (tmp_path / "recording.ssd").write_text(
        "Number of units:\n2\n\n"
        "List with the names of units:\nU01\nU02\n\n"
        "List with the recording elements (channels, tetrodes, etc) "
        "where each unit originates:\nEl_01\nEl_02\n\n"
        "Number of spikes in each unit:\n3\n2\n",
        encoding="utf-8",
    )
    data = load_realdata(str(tmp_path), align_to_peak=False, zscore=False)
    assert data.spikes.shape == (5, length)
    assert data.labels.tolist() == [0, 0, 0, 1, 1]


def test_metrics_on_separated_blobs():
    """Well separated blobs should give a near-perfect ARI."""
    rng = np.random.default_rng(0)
    features = np.vstack(
        [rng.normal(center, 0.1, size=(40, 2)) for center in [(0, 0), (10, 0), (0, 10)]]
    )
    labels = np.repeat([0, 1, 2], 40)
    scores = evaluate_clustering(features, labels)
    assert scores["n_clusters"] == 3
    assert scores["ari"] > 0.9
    assert scores["v_measure"] > 0.9


def test_baseline_forward_shapes():
    """AE and Gaussian VAE keep the waveform dimensionality."""
    batch = torch.randn(16, 79)
    recon, code = Autoencoder(79, latent_dim=2)(batch)
    assert recon.shape == (16, 79)
    assert code.shape == (16, 2)

    recon, mu, logvar = GaussianVAE(79, latent_dim=10)(batch)
    assert recon.shape == (16, 79)
    assert mu.shape == (16, 10)
    assert logvar.shape == (16, 10)


def test_qbm_vae_trains_with_offline_sampler():
    """The QBM-VAE training loop and feature extraction run end to end."""
    torch.manual_seed(0)
    np.random.seed(0)
    spikes = torch.randn(64, 79)
    model = DummyQBMVAE(79, torch.tanh, qbm_config(10))
    model = train_qbm_vae(model, spikes, epochs=1, batch_size=32)
    features = extract_features(model, spikes, "qbmvae")
    assert features.shape == (64, 10)
    assert torch.isfinite(features).all()


def test_qbm_vae_feature_representations():
    """``sigmoid`` features are the probabilities of the raw logits."""
    torch.manual_seed(0)
    model = DummyQBMVAE(79, torch.tanh, qbm_config(10))
    spikes = torch.randn(32, 79)
    logits = extract_features(model, spikes, "qbmvae", representation="q")
    probabilities = extract_features(model, spikes, "qbmvae", representation="sigmoid")
    assert probabilities.shape == logits.shape
    assert torch.allclose(probabilities, torch.sigmoid(logits), atol=1e-6)
    assert not torch.allclose(logits, probabilities)

    with pytest.raises(ValueError):
        extract_features(model, spikes, "qbmvae", representation="unknown")
