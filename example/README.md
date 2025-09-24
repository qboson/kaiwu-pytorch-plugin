**Language Versions**: [中文](example/README_ZH.md) | [English](example/README.md)

### Examples of BoltzmannMachine and RestrictedBoltzmannMachine

* `run_bm.py`: Demonstrates how to use the `BoltzmannMachine` class for model instantiation, sampling, objective function calculation, and parameter optimization. This is suitable for understanding the basic training workflow of a Boltzmann Machine, including sampling and gradient backpropagation.
* `run_rbm.py`: Demonstrates how to use the `RestrictedBoltzmannMachine` class for training, including hidden feature extraction, sampling, objective function calculation, and parameter optimization. This is suitable for understanding the typical application workflow of Restricted Boltzmann Machines.

Both scripts showcase the complete steps of model initialization, sampling, objective function calculation, backpropagation, and parameter updating. They can serve as quick-start references for working with Boltzmann Machine-related models.

---

### Classification Task: Handwritten Digit Recognition

This example demonstrates how to use a Restricted Boltzmann Machine (RBM) for feature learning and classification on the handwritten digits dataset (Digits). It is intended for beginners to understand the application workflow of RBMs in image feature extraction and classification, and can serve as a foundation for more advanced experiments and extensions. The main contents include:

* **Data augmentation and preprocessing**: Expanding the dataset of original 8x8 handwritten digit images by shifting them up, down, left, and right, followed by feature normalization using MinMaxScaler;
* **RBM model training**: Implementing the `RBMRunner` class to encapsulate the RBM training process, with support for visualizing generated samples and weight matrices during training;
* **Feature extraction and classification**: After training, using the hidden-layer representations from the RBM as features for classification with logistic regression;
* **Visualization and analysis**: Supporting sample generation and weight visualization during training to help observe and evaluate the learning effects of the model.

**Dependencies**

```
scikit-learn
matplotlib
scipy
```

Run the example via `example/rbm_digits/rbm_digits.ipynb`.

---

### Generation Task: Q-VAE for MNIST Image Generation

This example demonstrates how to train and evaluate a Quantum Variational Autoencoder (Q-VAE) model on the MNIST handwritten digit dataset. It is intended for those who wish to understand the training, generation, and evaluation workflow of Q-VAE models, and can serve as a foundation for further research on generative models. The main contents include:

* **Data loading and preprocessing**: Implementing a custom dataset class that supports batch indexing, combined with `ToTensor` transformation and flattening operations;
* **Model construction**: Building the Q-VAE architecture, including encoder and decoder modules, as well as RBM-based latent variable modeling;
* **Training process**: Designing and implementing a full training loop with tracking of loss, Evidence Lower Bound (ELBO), KL divergence, and other metrics, along with checkpoint saving;
* **Visualization and generation**: Providing side-by-side visualization of original, reconstructed, and generated images for intuitive model evaluation.

Run the example via `example/qvae_mnist/train_qvae.ipynb`.

**Dependencies**

```
torchvision==0.22.0
torchmetrics[image]
```

---

要不要我帮你整理成一个 **README.md** 模板（中英文对照），这样直接放到项目里就可以用了？
