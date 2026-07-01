# -*- coding: utf-8 -*-
"""
Trainer module for QVAE training.

This module provides the Trainer class that handles data loading, model creation,
training loop, and result saving for QVAE models.
"""


import os
from datetime import datetime
import torch
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from torch.utils.data import DataLoader, Subset, TensorDataset
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json

from model import Config, QVAE, FeatureExtractor
from .model_tuner import ModelTuner

from utils.loadMNIST import loadMNIST
from utils.helpers import plot_MNIST_output, t_SNE, create_tsne_animation#, save_list_to_txt
from utils.logging import get_logger
from utils.exception import ValueError, BuildError

logger = get_logger(__name__)
# logger = logging.getLogger(__name__)


class Trainer:
    """统一的训练器类，封装数据加载、模型创建、训练循环、结果保存"""

    def __init__(self, config, custom_train_data=None, custom_test_data=None):
        """
        Initialize Trainer with configuration.

        Args:
            config (Config): Configuration object containing all parameters.
            custom_train_data (tuple, optional): (X, y) for custom training data.
            custom_test_data (tuple, optional): (X, y) for custom test data.
        """
        self.config = config
        self.custom_train_data = custom_train_data
        self.custom_test_data = custom_test_data

        # 输出目录
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = config.output_dir or f"./output/{self.config.type}_{self.timestamp}"
        os.makedirs(self.output_dir, exist_ok=True)

        # 内部组件
        self.model = None
        self.tuner = None
        self.train_loader = None
        self.test_loader = None
        self.dataset_mean = None

        # 训练历史
        self.train_losses = []
        self.test_losses = []

    def _setup_data(self):
        """加载数据并计算均值"""
        if self.custom_train_data is not None:
            X_train, y_train = self.custom_train_data
            X_test, y_test = self.custom_test_data
            train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
            test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))
            self.train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)
            self.test_loader = DataLoader(test_dataset, batch_size=self.config.batch_size, shuffle=False)
        else:
            self.train_loader, self.test_loader = loadMNIST(
                name=self.config.name,
                data_path=self.config.data_path,
                batch_size=self.config.batch_size,
                num_evts_train=self.config.num_train_samples,
                num_evts_test=self.config.num_test_samples,
                use_cuda=self.config.use_cuda
            )
        # 计算数据集均值
        total_sum = 0
        total_count = 0
        for data, _ in self.train_loader:
            total_sum += data.sum()
            total_count += data.numel()
        self.dataset_mean = total_sum / total_count
        logger.info(f"Dataset mean: {self.dataset_mean:.4f}")
        return self.train_loader, self.test_loader

    def _create_model(self):
        """创建模型和配置"""
        input_dim = self.train_loader.dataset[0][0].numel()
        logger.info(f"Input dimension: {input_dim}")

        if self.config.type == 'QVAE':
            model = QVAE(
                input_dimension=input_dim,
                activation_fct=self.config.activation_fct,
                bm_type=self.config.bm_type,
                sampler_type=self.config.sampler_type,
                config=self.config
            )
        elif self.config.type == 'CellQVAE':
            model = CellQVAE(
                input_dimension=input_dim,
                activation_fct=self.config.activation_fct,
                bm_type=self.config.bm_type,
                sampler_type=self.config.sampler_type,
                config=self.config,
                n_batches=self.n_batches,
            )
        else:
            raise ValueError(f"Unsupported model type: {self.config.type}")

        model.create_networks()
        self.model = model
        logger.info(f"Sampler created: {self.config.sampler_type}")
        logger.info(f"Model created: {self.config.type}")
        if self.config.type in ['QVAE', 'CellQVAE']:
            logger.info(f"BM created: {self.config.bm_type}")
        return model

    def _setup_tuner(self):
        """初始化 ModelTuner 并设置优化器"""
        self.tuner = ModelTuner(config=self.config)

        # 注册模型和数据加载器
        self.tuner.register_model(self.model)
        self.tuner.register_dataLoaders(self.train_loader, self.test_loader)
        self.tuner.outpath = self.output_dir

        # 创建优化器
        if self.config.type in ['QVAE', 'CellQVAE']:
            vae_params = list(self.model.encoder.parameters()) + list(self.model.decoder.parameters())
            vae_optimizer = torch.optim.Adam(vae_params, lr=self.config.lr)
            bm_optimizer = torch.optim.Adam(self.model.bm.parameters(), lr=self.config.bm_lr)
            self.tuner.register_two_optimisers(vae_optimizer, bm_optimizer)
        else:
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.lr)
            self.tuner.register_optimiser(optimizer)

        # 设置数据集均值
        if hasattr(self.model, 'set_dataset_mean'):
            self.model.set_dataset_mean(self.dataset_mean)
            logger.info("Set dataset mean")
    
        # 设置训练偏置
        if self.config.type in ["RBM_VAE", "QVAE", "CellQVAE"] and hasattr(self.model, 'set_train_bias'):
            self.model.set_train_bias(self.dataset_mean)
            logger.info("Set train bias")

        return self.tuner

    def train(self, run_tsne=False, compute_energy=False, tsne_interval=10, generate_animation=False):
        """执行完整训练流程"""
        logger.info(f"Start training {self.config.type}")
        self._setup_data()
        self._create_model()
        self._setup_tuner()

        # 提取测试集标签（用于能量可视化）
        y_test = []
        for _, labels in self.test_loader:
            y_test.extend(labels.numpy().tolist())
        y_test = np.array(y_test)

    
        # 调试信息
        logger.info(f"Test labels - unique: {np.unique(y_test)}, count: {len(y_test)}")
        logger.info(f"Test labels distribution: {np.bincount(y_test)}")

        tsne_frames = []

        epoch_pbar = tqdm(range(1, self.config.num_epochs + 1), desc="Training Progress")
        for epoch in epoch_pbar:
            train_loss = self.tuner.train(epoch)
            if hasattr(train_loss, 'item'):
                train_loss = train_loss.item()
            self.train_losses.append(train_loss)

            test_loss, input_data, output_data, label_list = self.tuner.test()
            if hasattr(test_loss, 'item'):
                test_loss = test_loss.item()
            self.test_losses.append(test_loss)

            epoch_pbar.set_description(f"Epoch {epoch}/{self.config.num_epochs} - Train Loss: {train_loss:.2f}, Test Loss: {test_loss:.2f}")

            if epoch % (self.config.num_epochs // 10) == 0 or epoch == self.config.num_epochs:
                self._save_reconstruction(epoch, input_data, output_data)
                logger.info(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Test Loss={test_loss:.4f}")

            if generate_animation and (epoch % tsne_interval == 0 or epoch == self.config.num_epochs):
                frame_path = self._save_tsne_frame(epoch)
                tsne_frames.append(frame_path)

        self.tuner.save_model(config_string=f"final_{self.config.type}")
        self._plot_training_curve()

        if generate_animation and tsne_frames:
            create_tsne_animation(tsne_frames, output_path=self.output_dir)
            import shutil
            frame_dir = os.path.join(self.output_dir, "temp_tsne_frames")
            if os.path.exists(frame_dir):
                shutil.rmtree(frame_dir)

        if run_tsne:
            logger.info("Generating t-SNE visualization...")
            self._visualize_tsne()

        if compute_energy:
            logger.info("Compute BM energy for each sample...")
            energies = self.compute_energy(self.model, self.test_loader)
            self._save_energy_plot(energies, y_test, output_path=self.output_dir)

        logger.info(f"{self.config.type} training completed")
        return self.model, self.train_losses, self.test_losses

    def _save_reconstruction(self, epoch, input_data, output_data):
        """保存重建图像"""
        if isinstance(input_data, list):
            x_true = input_data[0][:10].detach().numpy()
        else:
            x_true = input_data[:10].detach().numpy()

        if isinstance(output_data, list):
            x_recon = output_data[0][:10].detach().numpy()
        else:
            x_recon = output_data[:10].detach().numpy()

        plot_MNIST_output(
            x_true, x_recon,
            output=os.path.join(self.output_dir, f"reconstruction_epoch_{epoch}.png")
        )

    def _plot_training_curve(self):
        """绘制训练曲线并保存"""
        plt.figure(figsize=(6, 5))
        plt.plot(self.train_losses, label='Training Loss', color="blue", alpha=0.7)
        plt.plot(self.test_losses, label='Test Loss', color="red", alpha=0.7)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'{self.config.type} Training Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.output_dir, 'qvae_training_curve.png'))
        plt.show()

    def _save_energy_plot(self, energies, labels, output_path):
        """绘制能量分布小提琴图并保存"""
        # 将标签转换为字符串，确保 seaborn 视其为分类变量
        labels_str = [str(label) for label in labels]
        # df = pd.DataFrame({'energy': energies, 'label': labels_str})
        df = pd.DataFrame({'energy': energies, 'label': [str(l) for l in labels]})
        median_order = df.groupby('label')['energy'].median().sort_values().index.tolist()

        # # 获取唯一标签并排序（确保 0-9 顺序）
        # unique_labels = sorted(df['label'].unique())

        plt.figure(figsize=(10, 6))
        sns.violinplot(
            x='label', 
            y='energy', 
            data=df, 
            order=median_order,  # 按中位数排序
            inner='quartile', 
            palette='Spectral'
        )
        plt.title('QVAE Energy Distribution (sorted by median)', fontsize=14, pad=15)
        plt.xlabel('Class')
        plt.ylabel('Energy')
        save_path = os.path.join(output_path, 'qvae_energy_by_class.png')
        plt.savefig(save_path, dpi=300)
        plt.close()
        logger.info(f"Energy distribution plot saved to {save_path}")

    def _visualize_tsne(self, point_size=20, alpha=0.6, save_path=None, show=True):
        """
        执行 t-SNE 可视化并保存图像

        Args:
            point_size: 散点大小
            alpha: 透明度
            save_path: 保存路径，如果为 None 则自动生成
            show: 是否显示图像
        """
        # 检查模型是否已训练
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # 生成默认保存路径（如果未指定）
        if save_path is None:
            save_path = os.path.join(
                self.output_dir,
                f"{self.config.type}_t-SNE_epochs_{self.config.num_epochs}.png"
            )

        # 调用 t_SNE 函数
        df_tsne, final_save_path, training_status = t_SNE(
            test_loader=self.test_loader,
            qvae_model=self.model,
            point_size=point_size,
            alpha=alpha,
            epochs=self.config.num_epochs,
            save_path=save_path,
            show=show
        )

        return df_tsne, final_save_path, training_status

    def _save_tsne_frame(self, epoch):
        """
        保存单个epoch的t-SNE帧到临时文件夹
        """
        frame_dir = os.path.join(self.output_dir, "temp_tsne_frames")
        os.makedirs(frame_dir, exist_ok=True)
        frame_path = os.path.join(frame_dir, f"tsne_epoch_{epoch:03d}.png")

        # 调用 t_SNE 生成帧（不显示）
        _, _, _ = t_SNE(
            test_loader=self.test_loader,
            qvae_model=self.model,
            epochs=epoch,
            save_path=frame_path,
            show=False
        )
        return frame_path

    def save_results(self):
        """保存训练损失到文件（可选）"""
        config_dict = {}
        for k, v in self.config.__dict__.items():
            if k.startswith('_'):
                continue
            if k == 'activation_fct':
                # Convert torch.nn module to string name
                config_dict[k] = v.__class__.__name__ if hasattr(v, '__class__') else str(v)
            else:
                config_dict[k] = v

        results = {
            'train_losses': self.train_losses,
            'test_losses': self.test_losses,
            'config': config_dict
        }
        with open(os.path.join(self.output_dir, 'results.json'), 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {self.output_dir}")

    def extract_features(self, dataloader, feature_type='q'):
        extractor = FeatureExtractor(self.model, feature_type)
        return extractor.extract(dataloader)

    def compute_energy(self, model, dataloader):
        """Compute BM energy for each sample in dataloader."""
        model.eval()
        energies = []
        with torch.no_grad():
            for x, _ in dataloader:
                x = x.to(next(model.parameters()).device)
                energy = model.energy(x, self.config.loss_type)
                energies.extend(energy.cpu().numpy().tolist())
        return energies

    # def get_classifier_pipeline(self, feature_type='q', **classifier_kwargs):
    #     extractor = FeatureExtractor(self.model, feature_type)
    #     clf = MLPClassifier(**classifier_kwargs)
    #     return Pipeline([('extractor', extractor), ('classifier', clf)])