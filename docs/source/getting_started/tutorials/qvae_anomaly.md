# Q-VAE for Energy-Supervised Anomaly Detection

> **Status**: Coming soon — placeholder page; will be expanded into a full tutorial.

Tabular anomaly detection with a CleanEnergyQVAE: the Q-VAE latent model is
extended with a free-energy scoring head and an NCE-style contrastive
objective, so normal samples get lower free energy and anomalies get higher
free energy. Covers threshold search (best-F1 over a candidate grid) and
PR/ROC / score-distribution / t-SNE evaluation.

**Example**: `example/qvae_anomaly/train_public_ad.py` ·
`example/qvae_anomaly/evaluate_public_ad.py` ·
`example/qvae_anomaly/qvae_anomaly_demo.ipynb`

**Data**: ADBench `38_thyroid.npz` · Kaggle `creditcard.csv`
