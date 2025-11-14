# Copyright (C) 2022-2025 Beijing QBoson Quantum Technology Co., Ltd.
#
# SPDX-License-Identifier: Apache-2.0
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.optim import SGD
from torch.utils.data import DataLoader, TensorDataset

from kaiwu.classical import SimulatedAnnealingOptimizer
from kaiwu.torch_plugin.dbn import UnsupervisedDBN

# =================== Unsupervised DBN Trainer =====================
class DBNTrainer:
    """A trainer for the DBN model.

    This class handles the layer-wise pre-training of the DBN.
    """

    def __init__(
        self,
        learning_rate_rbm=0.1,
        n_epochs_rbm=10,
        batch_size=100,
        verbose=True,
        shuffle=True,
        drop_last=False,
        plot_img=False,
        random_state=None,
        dbn_ref=None,
    ):
        """Initializes the DBNTrainer.

        Args:
            learning_rate_rbm (float, optional): Learning rate for RBM training.
                Defaults to 0.1.
            n_epochs_rbm (int, optional): Number of training epochs for each RBM.
                Defaults to 10.
            batch_size (int, optional): The batch size for training.
                Defaults to 100.
            verbose (bool, optional): If True, prints training progress.
                Defaults to True.
            shuffle (bool, optional): If True, shuffles the training data.
                Defaults to True.
            drop_last (bool, optional): If True, drops the last incomplete batch.
                Defaults to False.
            plot_img (bool, optional): If True, plots training progress images.
                Defaults to False.
            random_state (int, optional): Seed for random number generators.
                Defaults to None.
            dbn_ref (UnsupervisedDBN, optional): A reference to the DBN model.
                Defaults to None.
        """
        self.learning_rate_rbm = learning_rate_rbm
        self.n_epochs_rbm = n_epochs_rbm
        self.batch_size = batch_size
        self.verbose = verbose
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.plot_img = plot_img
        self.random_state = random_state

        self.sampler = SimulatedAnnealingOptimizer(alpha=0.999, size_limit=100)
        self.dbn_ref = dbn_ref

    def train(self, dbn, data_in):
        """Pre-trains the DBN model layer by layer.

        Args:
            dbn (UnsupervisedDBN): An instance of the UnsupervisedDBN model.
            data_in (numpy.ndarray): The training data, shape (n_samples, n_features).

        Returns:
            UnsupervisedDBN: The trained DBN model.

        Raises:
            ValueError: If `dbn` is not an instance of UnsupervisedDBN.
        """
        if not isinstance(dbn, UnsupervisedDBN):
            raise ValueError("dbn must be an instance of UnsupervisedDBN")

        # Save DBN reference
        self.dbn_ref = dbn

        # Set random seed
        if self.random_state is not None:
            self._set_random_seed()

        input_data = data_in.astype(np.float32)

        # Create RBM layers if they don't exist
        if dbn.num_layers == 0:
            dbn.create_rbm_layer(data_in.shape[1])

        for idx in range(dbn.num_layers):  # Use num_layers and get_rbm_layer
            rbm = dbn.get_rbm_layer(idx)
            if self.verbose:
                n_visible = rbm.num_visible
                n_hidden = rbm.num_hidden
                print(
                    f"\n[DBN] Pre-training RBM layer {idx+1}/{dbn.num_layers}: "
                    f"{n_visible} -> {n_hidden}"
                )

            # Train the current RBM layer
            input_data = self._train_rbm_layer(rbm, input_data, idx)

        # Mark the model as trained
        dbn.mark_as_trained()
        return dbn

    def get_training_config(self):
        """Gets the training configuration.

        Returns:
            dict: A dictionary containing the training configuration.
        """
        return {
            "learning_rate_rbm": self.learning_rate_rbm,
            "n_epochs_rbm": self.n_epochs_rbm,
            "batch_size": self.batch_size,
            "verbose": self.verbose,
            "shuffle": self.shuffle,
            "drop_last": self.drop_last,
            "plot_img": self.plot_img,
            "random_state": self.random_state,
            "device": str(getattr(self.dbn_ref, "device", "unknown")),
        }

    def _train_rbm_layer(self, rbm, data_in, layer_idx):
        """Trains a single RBM layer.

        Args:
            rbm (RestrictedBoltzmannMachine): The RBM layer to train.
            data_in (numpy.ndarray): The input data for this layer.
            layer_idx (int): The index of the current layer.

        Returns:
            numpy.ndarray: The output of the trained RBM layer (hidden activations).
        """
        optimizer = SGD(rbm.parameters(), lr=self.learning_rate_rbm)

        # Use the input data for the current layer, not the original data
        data_in = torch.FloatTensor(data_in).to(rbm.device)

        dataset = TensorDataset(data_in)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            drop_last=self.drop_last,
        )

        if self.verbose:
            print("[DBN] Pre-training start:")

        # Record samples during training
        training_samples = []

        # Training loop
        for epoch in range(self.n_epochs_rbm):
            total_loss = 0.0  # Total objective value for the current epoch
            for i, (batch_x,) in enumerate(
                loader
            ):  # Get batch data, batch_x: size=[batch, n_visible]
                loss = self._train_batch(rbm, optimizer, batch_x)

                # Accumulate objective value
                total_loss += loss.item()

                # Print weight and bias statistics every 20 batches
                if self.verbose and i % 20 == 0:
                    self._print_layer_stats(rbm)

                    # Sample and weight visualization
                    if self.plot_img:
                        self._visualize_training_progress(rbm, i, epoch, batch_x)

                    # Record samples
                    if i % 50 == 0:  # Record every 50 batches
                        training_samples.append(
                            {
                                "batch": i,
                                "epoch": epoch,
                                "original": batch_x[:5]
                                .cpu()
                                .numpy(),  # Save a few original samples
                                "weights": rbm.quadratic_coef.detach()
                                .cpu()
                                .numpy()
                                .copy(),
                            }
                        )

                # Calculate average objective value for the current epoch
                avg_loss = total_loss / len(loader)

                # Print epoch average loss every 5 batches
                if self.verbose and i % 5 == 0:
                    print(f"Iteration {i+1}, Average Loss: {avg_loss:.6f}")

            # Print average loss and data shape for each RBM layer
            if self.verbose:
                print(f"Layer {layer_idx+1}, Epoch {epoch+1}: Loss {avg_loss:.6f}")
                print(f"Output shape after layer {layer_idx+1}: {data_in.shape}")

            # Print average loss for each epoch
            if self.verbose:
                print(
                    f"[RBM] Epoch {epoch+1}/{self.n_epochs_rbm} \tAverage Loss: {avg_loss:.6f}"
                )

            # Reconstruction evaluation at the end of each epoch
            if self.verbose and epoch % 1 == 0:  # Evaluate every epoch
                self._evaluate_reconstruction_quality(rbm, data_in, epoch, layer_idx)

        if self.verbose:
            print("[DBN] Pre-training finished")

        # Extract features as input for the next layer
        with torch.no_grad():
            hidden_output = rbm.get_hidden(data_in)
            return (
                hidden_output[:, rbm.num_visible :].cpu().numpy()
            )  # Extract only the hidden part

    def _train_batch(self, rbm, optimizer, batch_x):
        """Trains a single batch.

        Args:
            rbm (RestrictedBoltzmannMachine): The RBM layer.
            optimizer (torch.optim.Optimizer): The optimizer.
            batch_x (torch.Tensor): The input batch.

        Returns:
            torch.Tensor: The loss value for the batch.
        """
        h_prob = rbm.get_hidden(
            batch_x
        )  # Positive phase (compute hidden activations), size=[batch, n_hidden]
        s = rbm.sample(
            self.sampler
        )  # Negative phase (sample to reconstruct data), size=[[batch, n_visible + n_hidden]
        optimizer.zero_grad()  # Clear gradients

        # Calculate loss function (negative log-likelihood) + regularization terms
        w_decay = 0.02 * torch.sum(rbm.quadratic_coef**2)  # Weight decay
        b_decay = 0.05 * torch.sum(rbm.linear_bias**2)  # Bias decay
        loss = rbm.objective(h_prob, s) + w_decay + b_decay

        # Backpropagation and parameter update
        loss.backward()
        optimizer.step()
        return loss

    def _print_layer_stats(self, rbm):
        """Prints statistical information about the RBM layer.

        Args:
            rbm (RestrictedBoltzmannMachine): The RBM layer.
        """
        print(
            f"jmean {torch.abs(rbm.quadratic_coef).mean().item():.6f}"
            f"jmax {torch.abs(rbm.quadratic_coef).max().item():.6f}"
        )
        print(
            f"hmean {torch.abs(rbm.linear_bias).mean().item():.6f}"
            f"hmax {torch.abs(rbm.linear_bias).max().item():.6f}"
        )

    def _visualize_training_progress(self, rbm, batch_idx, epoch, current_batch):
        """Comprehensive visualization during training.

        Args:
            rbm (RestrictedBoltzmannMachine): The RBM layer.
            batch_idx (int): The current batch index.
            epoch (int): The current epoch.
            current_batch (torch.Tensor): The current batch data.
        """
        # Generate new samples (what the model has learned)
        self._visualize_generated_samples(rbm, batch_idx, epoch)

        # Weights and gradients visualization (how the model learns)
        self._visualize_weights_gradients(rbm, batch_idx, epoch)

        # Reconstruction of the current batch (real-time reconstruction ability)
        self._visualize_current_reconstruction(rbm, current_batch, batch_idx, epoch)

    def _visualize_generated_samples(self, rbm, batch_idx, epoch):
        """Visualizes samples generated from the model.

        Args:
            rbm (RestrictedBoltzmannMachine): The RBM layer.
            batch_idx (int): The current batch index.
            epoch (int): The current epoch.
        """
        with torch.no_grad():
            # Generate new samples from the model's distribution
            display_samples = (
                rbm.sample(self.sampler).cpu().numpy()[:20, : rbm.num_visible]
            )

        plt.figure(figsize=(16, 2))
        plt.imshow(self._gen_digits_image(display_samples, 8))
        plt.title(f"Generated Samples - Epoch {epoch+1}, Batch {batch_idx+1}")
        plt.axis("off")

        plt.show()

    def _visualize_weights_gradients(self, rbm, batch_idx, epoch):
        """Visualizes the weight matrix and its gradients.

        Args:
            rbm (RestrictedBoltzmannMachine): The RBM layer.
            batch_idx (int): The current batch index.
            epoch (int): The current epoch.
        """
        _, axes = plt.subplots(1, 3, figsize=(15, 4))

        # Weight matrix
        weights = rbm.quadratic_coef.detach().cpu().numpy()
        im0 = axes[0].imshow(weights, cmap="RdBu_r", aspect="auto")
        axes[0].set_title("Weight Matrix")
        plt.colorbar(im0, ax=axes[0])

        # Weight gradients
        grad = rbm.quadratic_coef.grad.detach().cpu().numpy()
        im1 = axes[1].imshow(grad, cmap="RdBu_r", aspect="auto")
        axes[1].set_title("Weight Gradients")
        plt.colorbar(im1, ax=axes[1])

        # Hidden unit biases
        h_bias = rbm.linear_bias[rbm.num_visible :].detach().cpu().numpy()
        axes[2].bar(range(len(h_bias)), h_bias)
        axes[2].set_title("Hidden Unit Biases")
        axes[2].set_xlabel("Hidden Unit Index")
        axes[2].set_ylabel("Bias Value")

        plt.suptitle(f"Model Parameters - Epoch {epoch+1}, Batch {batch_idx+1}")
        plt.tight_layout()

        plt.show()

    def _visualize_current_reconstruction(self, rbm, batch_data, batch_idx, epoch):
        """Visualizes the reconstruction quality of the current batch.

        Args:
            rbm (RestrictedBoltzmannMachine): The RBM layer.
            batch_data (torch.Tensor): The current batch data.
            batch_idx (int): The current batch index.
            epoch (int): The current epoch.
        """
        batch_numpy = batch_data.cpu().numpy()

        # Use static reconstruction method
        recon_imgs, _ = UnsupervisedDBN.reconstruct_with_rbm(rbm, batch_numpy)

        # Select a few samples to display
        n_show = min(8, batch_data.shape[0])
        original_imgs = batch_data[:n_show].cpu().numpy()

        _, axes = plt.subplots(2, n_show, figsize=(3 * n_show, 6))
        if n_show == 1:
            axes = axes.reshape(2, 1)

        for i in range(n_show):
            # Original image
            axes[0, i].imshow(original_imgs[i].reshape(8, 8), cmap="gray")
            axes[0, i].set_title(f"Original {i+1}")
            axes[0, i].axis("off")

            # Reconstructed image
            axes[1, i].imshow(recon_imgs[i].reshape(8, 8), cmap="gray")
            axes[1, i].set_title(f"Reconstructed {i+1}")
            axes[1, i].axis("off")

        plt.suptitle(f"Real-time Reconstruction - Epoch {epoch+1}, Batch {batch_idx+1}")
        plt.tight_layout()

        plt.show()

    def _evaluate_reconstruction_quality(self, rbm, input_data, epoch, layer_idx):
        """Periodically evaluates reconstruction quality using a static method.

        Args:
            rbm (RestrictedBoltzmannMachine): The RBM layer.
            input_data (torch.Tensor): The data to evaluate on.
            epoch (int): The current epoch.
            layer_idx (int): The index of the current layer.
        """
        n_eval = min(100, input_data.shape[0])
        eval_data = input_data[:n_eval]

        # Use static reconstruction method
        _, recon_errors = UnsupervisedDBN.reconstruct_with_rbm(rbm, eval_data)

        avg_recon_error = np.mean(recon_errors)
        print(
            f"[RBM] Layer {layer_idx+1}, Epoch {epoch+1}: "
            f"Reconstruction Error = {avg_recon_error:.6f}\n"
        )

    def _gen_digits_image(self, data_in, size=8):
        """Generates an image from digit data.

        Args:
            data_in (numpy.ndarray): The digit data.
            size (int, optional): The size of each digit image (size x size).
                Defaults to 8.

        Returns:
            numpy.ndarray: A horizontally stacked image of digits.
        """
        digits = data_in.reshape(20, size, size)
        image = np.hstack(digits)
        return image

    def _set_random_seed(self):
        """Sets the random seed for reproducibility."""
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.random_state)


class DBNPretrainer(BaseEstimator, TransformerMixin):
    """
    Scikit-learn兼容的DBN预训练
    """
    def __init__(
        self,
        hidden_layers_structure=[100, 100],
        learning_rate_rbm=0.1,
        n_epochs_rbm=10,
        batch_size=100,
        verbose=True,
        shuffle=True,
        drop_last=False,
        plot_img=False,
        random_state=None
    ):
        self.hidden_layers_structure = hidden_layers_structure
        self.learning_rate_rbm = learning_rate_rbm
        self.n_epochs_rbm = n_epochs_rbm
        self.batch_size = batch_size
        self.verbose = verbose
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.plot_img = plot_img
        self.random_state = random_state
        
        # 创建模型和训练器
        self._dbn = UnsupervisedDBN(hidden_layers_structure)
        self._trainer = DBNTrainer(
            learning_rate_rbm=learning_rate_rbm,
            n_epochs_rbm=n_epochs_rbm,
            batch_size=batch_size,
            verbose=verbose,
            shuffle=shuffle,
            drop_last=drop_last,
            plot_img=plot_img,
            random_state=random_state
        )

    def fit(self, X, y=None):
        """训练模型"""
        self._dbn.create_rbm_layer(X.shape[1])
        self._trainer.train(self._dbn, X)
        return self

    def transform(self, X):
        """特征变换"""
        return self._dbn.transform(X)

    # 提供访问RBM层的方法
    def get_rbm_layer(self, index):
        """获取指定RBM层"""
        return self._dbn.get_rbm_layer(index)

    @property
    def device(self):
        """返回device - 使用底层模型的属性"""
        return self._dbn.device
    
    @property
    def _n_layers(self):
        """返回层数 - 使用底层模型的属性"""
        return self._dbn.num_layers

    @property
    def _output_dim(self):
        """返回输出维度 - 使用底层模型的属性"""
        return self._dbn.output_dim