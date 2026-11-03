# Q-VAE for Single-Cell Transcriptomics

> **Status**: Coming soon — placeholder page; will be expanded into a full tutorial.

Single-cell representation learning with a quantum variational autoencoder (QVAE): read a single-cell expression matrix, train a QVAE, extract low-dimensional representations, compute UMAP, analyze the energy distribution, and evaluate clustering quality against cell-type labels.

**Example**: `example/qvae_cell/train_qvae_cell.ipynb` · `example/qvae_cell/evaluate_benchmark.py`

**Data**: [Kaggle immune-data](https://www.kaggle.com/datasets/redwoodz/immune-data)


## Clustering score coverage

Clustering agreement metrics use only cells with both a reference annotation and a cluster assignment. Missing values are excluded before converting labels to strings, while actual text such as `"nan"` remains a valid label. Results include `n_evaluated_cells`; `n_clusters` counts clusters within that evaluated subset. An empty labeled subset raises `ValueError`.
