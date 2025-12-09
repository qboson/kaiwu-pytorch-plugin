**Language Versions**: [中文](README_ZH.md) | [English](README.md)

### Examples of BoltzmannMachine and RestrictedBoltzmannMachine

* `run_bm.py`: Demonstrates how to use the `BoltzmannMachine` class for model instantiation, sampling, objective function calculation, and parameter optimization. This is suitable for understanding the basic training workflow of a Boltzmann Machine, including sampling and gradient backpropagation.
* `run_rbm.py`: Demonstrates how to use the `RestrictedBoltzmannMachine` class for training, including hidden feature extraction, sampling, objective function calculation, and parameter optimization. This is suitable for understanding the typical application workflow of Restricted Boltzmann Machines.

Both scripts showcase the complete steps of model initialization, sampling, objective function calculation, gradient descent, and parameter updating. They can serve as quick-start references for working with Boltzmann Machine-related models.

---

### Classification Task: Handwritten Digit Recognition

This section provides two complementary approaches for handwritten digit recognition using different neural network architectures, both demonstrating the power of unsupervised feature learning for classification tasks.

#### RBM-Based Classification

This example demonstrates how to use a Restricted Boltzmann Machine (RBM) for feature learning and classification on the handwritten digits dataset (Digits). It is intended for beginners to understand the application workflow of RBMs in image feature extraction and classification, and can serve as a foundation for more advanced experiments and extensions. The main contents include:

* **Data augmentation and preprocessing**: Expanding the dataset of original 8x8 handwritten digit images by shifting them up, down, left, and right, followed by feature normalization using MinMaxScaler;
* **RBM model training**: Implementing the `RBMRunner` class to encapsulate the RBM training process, with support for visualizing generated samples and weight matrices during training;
* **Feature extraction and classification**: After training, using the hidden-layer representations from the RBM as features for classification with logistic regression;
* **Visualization and analysis**: Supporting sample generation and weight visualization during training to help observe and evaluate the learning effects of the model.

Run the example via `example/rbm_digits/rbm_digits.ipynb`.

#### DBN-Based Classification

Building upon the RBM approach, this example demonstrates a complete Deep Belief Network (DBN) implementation with multiple RBM layers, offering more sophisticated feature learning and flexible training strategies. This implementation can be seen as a direct evolution of the RBM approach, demonstrating how stacking multiple RBMs enables learning increasingly abstract representations of the input data. The main contents include:

* **Hierarchical Feature Abstraction**: Each RBM layer learns to represent patterns in the activities of the layer below, creating features of features that capture increasingly abstract regularities in the data;
* **Unsupervised Pre-training**: Implementing the `DBNPretrainer` class for greedy layer-wise RBM training using contrastive divergence;
* **Dual Training Strategies**: 
  - *Fine-tuning Mode*: End-to-end backpropagation through the entire network after pre-training using `SupervisedDBNClassification`;
  - *Classifier Mode*: Traditional ML classifiers on DBN-extracted features;
* **Advanced Architecture**: PyTorch-based implementation with scikit-learn compatibility through `AbstractSupervisedDBN` base classes.

Run the example via `example/dbn_digits/supervised_dbn_digits.ipynb`.

**Dependencies**

```
scikit-learn
matplotlib
scipy
```

### Generation Task: Distribution Sampling Based on Boltzmann Machines (BM)

Boltzmann Machines (BMs) consist of fully connected units and can be trained in an unsupervised manner using methods such as Contrastive Divergence (CD). They are capable of generating new samples that closely match the statistical characteristics of the training data.

**Model Construction**:  
Train the BM jointly using Kullback–Leibler (KL) divergence and Noise-Contrastive Likelihood (NCL).

**Training Pipeline**:  
- Implement a `Trainer` class.  
- Integrate a learning rate scheduler.  
- Support distinct sampling strategies tailored to different phases of the sampling process (e.g., initial burn-in vs. final generation).

**Data Visualization**:  
Visualize the distribution of the generated samples to assess fidelity to the original data distribution.

Run the training script via `example/bm_generation/train_bm.ipynb`, and execute the sampling/testing script via `example/bm_generation/sample_bm.ipynb`.

**Dependencies**:
```text
kaiwu==1.3.0
pandas
matplotlib
```

---

### Generation Task: Q-VAE for MNIST Image Generation

This example demonstrates how to train and evaluate a Quantum Variational Autoencoder (Q-VAE) model on the MNIST handwritten digit dataset. It is intended for those who wish to understand the training, generation, and evaluation workflow of Q-VAE models, and can serve as a foundation for further research on generative models. The main contents include:

* **Data loading and preprocessing**: Standardized MNIST loading combined with flatten transform and GPU acceleration support;
* **Model construction**: Building the Q-VAE architecture, including encoder and decoder modules, as well as RBM-based latent variable modeling;
* **Training process**: Designing and implementing a full training loop with tracking of loss, Evidence Lower Bound (ELBO), KL divergence, and other metrics, along with checkpoint saving;
* **Generation capabilities**: Providing side-by-side visualization of original, reconstructed, and generated images for intuitive model evaluation.

Run the example via `example/qvae_mnist/train_qvae.ipynb`.

**Dependencies**

```
torchvision==0.22.0
torchmetrics[image]
```

---

### Representation Learning: Latent Feature Extraction and Classification

This extended example demonstrates how pre-trained Q-VAE representations can be leveraged for downstream classification tasks, embodying the core principle that "learning representations of the data that make it easier to extract useful information when building classifiers or other predictors" (Bengio, 2013). The main contents include:

* **Unsupervised Feature Learning**: Q-VAE encoder learns meaningful features without label supervision;
* **Transfer Learning**: Pre-trained representations enable efficient downstream task adaptation;
* **Multi-task Capability**: Same representations support both generation and classification;
* **Model Interpretability**: t-SNE visualization renables qualitative assessment of latent space structure, provides insights into class separation and cluster formation during training.

Run the example via `example/qvae_mnist/train_qvae_classifier.ipynb`.

**Dependencies**
```
torchvision==0.22.0
```
