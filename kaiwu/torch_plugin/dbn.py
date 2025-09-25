# -*- coding: utf-8 -*-
"""deep belief network DBN模型
包含DBN的类  以及训练DBN+model/仅训练model的函数 训练DBN+model会保存训练过程中的似然值和预测准确率
"""
import os
import numpy as np
from scipy.ndimage import shift
import matplotlib.pyplot as plt
import seaborn as sns

from abc import ABC, abstractmethod
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
# from sklearn.utils.multiclass import unique_labels
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import confusion_matrix

import torch
import torch.nn as nn
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader, TensorDataset
from kaiwu.torch_plugin import RestrictedBoltzmannMachine
from kaiwu.classical import SimulatedAnnealingOptimizer

# =================== 无监督DBN =====================
class UnsupervisedDBN(BaseEstimator, TransformerMixin):
    """
    基于RestrictedBoltzmannMachine用于无监督预训练的DBN（堆叠RBM）
    参数:
        hidden_layers_structure (list): 每层隐藏单元个数，如[256, 256]表示两层。
        learning_rate_rbm (float): 每层RBM的学习率。
        n_epochs_rbm (int): 每层RBM的训练轮数。
        batch_size (int): 批大小。
        verbose (bool): 打印训练信息。
        shuffle (bool): 数据加载时是否打乱。
        drop_last (bool): DataLoader是否丢弃最后不足batch的样本。
        random_state (int): 随机种子。
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
        
        self.sampler = SimulatedAnnealingOptimizer(alpha=0.999, size_limit=100)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.rbm_layers = None
        self.input_dim = None

    def fit(self, X, y=None):  # 修改接口以符合scikit-learn约定
        """
        预训练阶段
        Args:
            X: 训练数据，形状为 (n_samples, n_features)
            y: 忽略，为兼容scikit-learn接口
        """
        # 设置随机种子
        if self.random_state is not None:
            self._set_random_seed()

        input_data = X.astype(np.float32)

        # 清空之前的RBM
        self.rbm_layers = []

        # 创建RBM层
        self._create_rbm_layer(X.shape[1])

        for idx, rbm in enumerate(self.rbm_layers):
            if self.verbose:
                n_visible = rbm.num_visible
                n_hidden = rbm.num_hidden
                print(f"\n[DBN] Pre-training RBM layer {idx+1}/{len(self.rbm_layers)}: {n_visible} -> {n_hidden}")

            # 训练当前RBM层
            input_data = self._train_rbm_layer(rbm, input_data, idx)
            
        return self

    def transform(self, X):
        """
        特征变换
        Args:
            X: 输入数据，形状为 (n_samples, n_features)
        Returns:
            隐藏层特征，形状为 (n_samples, n_hidden)
        """
        if self.rbm_layers is None:
            raise ValueError("DBN model not trained yet. Call fit first.")
        X_data = X.astype(np.float32)
        for rbm in self.rbm_layers:
            with torch.no_grad():
                hidden_output = rbm.get_hidden(torch.FloatTensor(X_data).to(self.device))
                X_data = hidden_output[:, rbm.num_visible:].cpu().numpy()  # 只取隐藏层部分
        return X_data

    def _create_rbm_layer(self, input_dim):
        """创建RBM层"""
        self.input_dim = input_dim
        self.rbm_layers = nn.ModuleList()
        
        current_dim = input_dim
        for n_hidden in self.hidden_layers_structure:
            rbm = RestrictedBoltzmannMachine(
                num_visible=current_dim,    # 可见层单元数（特征维度）
                num_hidden=n_hidden,        # 隐层单元数
                h_range=[-1, 1],            # 隐层偏置范围
                j_range=[-1, 1],            # 权重范围
            ).to(self.device)               # 将模型移动到指定设备（CPU/GPU）
            self.rbm_layers.append(rbm)
            current_dim = n_hidden

    def _train_rbm_layer(self, rbm, input_data, layer_idx):
        """训练单个RBM层"""
        optimizer = SGD(rbm.parameters(), lr=self.learning_rate_rbm)

        # 使用当前层的输入数据，而不是原始X
        X_torch = torch.FloatTensor(input_data).to(self.device)
        
        dataset = TensorDataset(X_torch)
        loader = DataLoader(
            dataset, 
            batch_size=self.batch_size, 
            shuffle=self.shuffle,
            drop_last=self.drop_last
        )

        if self.verbose:
            print("[DBN] Pre-training start:")

        # 记录训练过程中的样本
        training_samples = []

        # 训练循环
        for epoch in range(self.n_epochs_rbm):
            total_loss = 0.0                                      # 当前epoch的总目标值
            for i, (batch_x,) in enumerate(loader):               # 获取batch数据, batch_x: size=[batch, n_visible]
                loss = self._train_batch(rbm, optimizer, batch_x)

                # 累加目标值
                total_loss += loss.item()
                
                # 每隔20个batch打印一次权重和偏置的统计信息
                if self.verbose and i % 20 == 0:
                    self._print_layer_stats(rbm, i, epoch)

                    # 样本和权重可视化
                    if self.plot_img:
                        self._visualize_training_progress(rbm, i, epoch, batch_x)
                        # self._print_display_samples(rbm, i, epoch)

                    # 记录样本
                    if i % 50 == 0:  # 每50个batch记录一次
                        training_samples.append({
                            'batch': i,
                            'epoch': epoch,
                            'original': batch_x[:5].cpu().numpy(),  # 保存几个原始样本
                            'weights': rbm.quadratic_coef.detach().cpu().numpy().copy()
                        })
            
                # 计算当前epoch的平均目标值
                avg_loss = total_loss / len(loader)

                # 每隔5个batch打印一次epoch的平均损失
                if self.verbose and i % 5 == 0:
                    print(f"Iteration {i+1}, Average Loss: {avg_loss:.6f}")

            # 打印每层RBM的平均损失以及数据形状
            if self.verbose:
                print(f"Layer {layer_idx+1}, Epoch {epoch+1}: Loss {avg_loss:.6f}")
                print(f"Output shape after layer {layer_idx+1}: {input_data.shape}")
        
            # 打印每个epoch的平均损失
            if self.verbose:
                print(f"[RBM] Epoch {epoch+1}/{self.n_epochs_rbm} \tAverage Loss: {avg_loss:.6f}")

             # 每个epoch结束后的重建评估
            if self.verbose and epoch % 1 == 0:  # 每个epoch评估一次
                self._evaluate_reconstruction_quality(rbm, input_data, epoch, layer_idx)
        
        if self.verbose:
            print("[DBN] Pre-training finished")     
        
        # 提取特征作为下一层输入
        with torch.no_grad():
            hidden_output = rbm.get_hidden(X_torch)
            return hidden_output[:, rbm.num_visible:].cpu().numpy()  # 只取隐藏层部分

    def _train_batch(self, rbm, optimizer, batch_x):
        """训练单个batch"""
        h_prob = rbm.get_hidden(batch_x)      # 正相（计算隐层激活）, size=[batch, n_hidden]
        s = rbm.sample(self.sampler)          # 负相（采样重构数据）, size=[[batch, n_visible + n_hidden]
        optimizer.zero_grad()                 # 梯度清零

        # 计算损失函数（负对数似然）+正则项
        w_decay = 0.02 * torch.sum(rbm.quadratic_coef**2)    # 权重衰减
        b_decay = 0.05 * torch.sum(rbm.linear_bias**2)       # 偏置衰减
        loss = rbm.objective(h_prob, s) + w_decay + b_decay
        
        # 反向传播并更新参数
        loss.backward()
        optimizer.step()
        return loss

    def _print_layer_stats(self, rbm, batch_idx, epoch):
        """打印统计信息"""
        # print(f"Batch {batch_idx+1}: \n"
        print(f"jmean {torch.abs(rbm.quadratic_coef).mean().item():.6f}"
              f"jmax {torch.abs(rbm.quadratic_coef).max().item():.6f}")
        print(f"hmean {torch.abs(rbm.linear_bias).mean().item():.6f}"
              f"hmax {torch.abs(rbm.linear_bias).max().item():.6f}")

    def _visualize_training_progress(self, rbm, batch_idx, epoch, current_batch):
        """训练过程中的综合可视化"""
        # 生成新样本（模型学到了什么）
        self._visualize_generated_samples(rbm, batch_idx, epoch)
    
        # 权重和梯度可视化（模型如何学习）
        self._visualize_weights_gradients(rbm, batch_idx, epoch)
    
        # 当前batch的重建效果（实时重建能力）
        self._visualize_current_reconstruction(rbm, current_batch, batch_idx, epoch)

    def _visualize_generated_samples(self, rbm, batch_idx, epoch):
        """可视化生成的样本"""
        with torch.no_grad():
            # 从模型分布中生成新样本
            display_samples = (
                rbm.sample(self.sampler)
                .cpu()
                .numpy()[:20, : rbm.num_visible]
            )
    
        plt.figure(figsize=(16, 2))
        plt.imshow(self._gen_digits_image(display_samples, 8))
        plt.title(f"Generated Samples - Epoch {epoch+1}, Batch {batch_idx+1}")
        plt.axis('off')
    
        # # 保存生成样本图像
        # if self.save_training_plots:
        #     plt.savefig(f'results/generated_epoch{epoch+1}_batch{batch_idx}.png', 
        #                dpi=150, bbox_inches='tight')
        plt.show()

    def _visualize_weights_gradients(self, rbm, batch_idx, epoch):
        """可视化权重和梯度"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
        # 权重矩阵
        weights = rbm.quadratic_coef.detach().cpu().numpy()
        im0 = axes[0].imshow(weights, cmap='RdBu_r', aspect='auto')
        axes[0].set_title('Weight Matrix')
        plt.colorbar(im0, ax=axes[0])
    
        # 权重梯度
        grad = rbm.quadratic_coef.grad.detach().cpu().numpy()
        im1 = axes[1].imshow(grad, cmap='RdBu_r', aspect='auto')
        axes[1].set_title('Weight Gradients')
        plt.colorbar(im1, ax=axes[1])
    
        # 隐藏单元偏置
        h_bias = rbm.linear_bias[rbm.num_visible:].detach().cpu().numpy()
        axes[2].bar(range(len(h_bias)), h_bias)
        axes[2].set_title('Hidden Unit Biases')
        axes[2].set_xlabel('Hidden Unit Index')
        axes[2].set_ylabel('Bias Value')
    
        plt.suptitle(f'Model Parameters - Epoch {epoch+1}, Batch {batch_idx+1}')
        plt.tight_layout()
    
        # if self.save_training_plots:
        #     plt.savefig(f'results/weights_epoch{epoch+1}_batch{batch_idx}.png', 
        #                dpi=150, bbox_inches='tight')
        plt.show()

    def _visualize_current_reconstruction(self, rbm, batch_data, batch_idx, epoch, layer_idx=0):
        """可视化当前batch的重建效果"""

        batch_numpy = batch_data.cpu().numpy()
        recon_imgs, recon_errors = self.reconstruct_get_hidden(batch_numpy, layer_idx)
    
        # 选择几个样本显示
        n_show = min(4, batch_data.shape[0])
        original_imgs = batch_data[:n_show].cpu().numpy()
        # recon_imgs = visible_recon[:n_show].cpu().numpy()
    
        fig, axes = plt.subplots(2, n_show, figsize=(3*n_show, 6))
        if n_show == 1:
            axes = axes.reshape(2, 1)
    
        for i in range(n_show):
            # 原始图像
            axes[0, i].imshow(original_imgs[i].reshape(8, 8), cmap='gray')
            axes[0, i].set_title(f'Original {i+1}')
            axes[0, i].axis('off')
        
            # 重建图像
            axes[1, i].imshow(recon_imgs[i].reshape(8, 8), cmap='gray')
            axes[1, i].set_title(f'Reconstructed {i+1}')
            axes[1, i].axis('off')
    
        plt.suptitle(f'Real-time Reconstruction - Epoch {epoch+1}, Batch {batch_idx+1}')
        plt.tight_layout()
    
        # if self.save_training_plots:
        #     plt.savefig(f'results/recon_epoch{epoch+1}_batch{batch_idx}.png', 
        #                dpi=150, bbox_inches='tight')
        plt.show()

    def _evaluate_reconstruction_quality(self, rbm, input_data, epoch, layer_idx):
        """定期评估重建质量"""
        _, recon_errors = self.reconstruct_get_hidden(input_data, layer_idx)
    
        # 计算所有样本的平均重建误差
        mean_recon_error = np.mean(recon_errors)
        print(f"[RBM] Layer {layer_idx+1}, Epoch {epoch+1}: Reconstruction Error = {mean_recon_error:.6f}\n")
        # print(f"[RBM] Layer {layer_idx+1}, Epoch {epoch+1}: Reconstruction Error = {recon_errors.item():.6f}")
        
    def _gen_digits_image(self, X, size=8):
        """
        生成图片
        Args:
            X: 形状为 (20, size * size) 的数组
        Returns：
            拼接后的大图像，形状为 (8, 20 * size)
        """

        plt.rcParams["image.cmap"] = "gray"
        # 先将每个数字的特征向量还原为8x8图像
        digits = X.reshape(20, size, size)  # 形状：(20, 8, 8)
        # 将20个8x8的图片横向拼接
        image = np.hstack(digits)  # 形状：(8, 160)
        return image

    def reconstruct_get_hidden(self, X, layer_idx=0):
        """
        统一使用RBM的get_hidden方法进行重建
        """
        if self.rbm_layers is None or len(self.rbm_layers) == 0:
            raise ValueError("No RBM layers found. Please fit the model first.")
        
        if layer_idx >= len(self.rbm_layers):
            raise ValueError(f"Layer index {layer_idx} out of range.")
        
        rbm = self.rbm_layers[layer_idx]
        rbm.eval()
        
        # 转换为PyTorch张量
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        with torch.no_grad():
            # 使用RBM的get_hidden方法获取隐藏层激活
            hidden_act = rbm.get_hidden(X_tensor)
            hidden_part = hidden_act[:, rbm.num_visible:]   # 只取隐藏层部分
            
            # 使用权重转置进行重建
            visible_recon = torch.sigmoid(
                torch.matmul(hidden_part, rbm.quadratic_coef.t()) + 
                rbm.linear_bias[:rbm.num_visible]
            )
            
            # 计算重建误差
            recon_errors = torch.mean((X_tensor - visible_recon) ** 2, dim=1).cpu().numpy()
            
        return visible_recon.cpu().numpy(), recon_errors

    def _set_random_seed(self):
        """设置随机种子"""
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.random_state)

# =================== 抽象监督DBN =====================
class AbstractSupervisedDBN(BaseEstimator, ABC):
    """
    抽象监督DBN类，传递练无监督预训以及定义接口用于下游任务(微调网络和分类器)
    """
    def __init__(
        self, 
        hidden_layers_structure=[100, 100],
        learning_rate_rbm=0.1,
        n_epochs_rbm=10,
        batch_size=32,
        verbose=True,
        plot_img=False,
        random_state=None,
        # 新增微调参数
        fine_tuning=False,             # 是否进行微调
        learning_rate=0.1,             # 微调学习率
        n_iter_backprop=100,           # 反向传播迭代次数
        l2_regularization=1e-4,        # L2正则化
        activation_function='sigmoid', # 激活函数
        dropout_p=0.0                  # Dropout概率
        ):
        # 无监督网络配置
        self.hidden_layers_structure = hidden_layers_structure
        self.learning_rate_rbm = learning_rate_rbm
        self.n_epochs_rbm = n_epochs_rbm
        self.batch_size = batch_size
        self.verbose = verbose
        self.plot_img = plot_img
        self.random_state = random_state
        
        # 监督微调配置
        self.fine_tuning = fine_tuning
        self.learning_rate = learning_rate
        self.n_iter_backprop = n_iter_backprop
        self.l2_regularization = l2_regularization
        self.activation_function = activation_function
        self.dropout_p = dropout_p
        
        # 网络组件
        self.fine_tune_network = None
        self.classifier = None
        self.label_encoder = LabelEncoder()
        self.unsupervised_dbn = UnsupervisedDBN(
                hidden_layers_structure=self.hidden_layers_structure,
                learning_rate_rbm=self.learning_rate_rbm,
                n_epochs_rbm=self.n_epochs_rbm,
                batch_size=self.batch_size,
                verbose=self.verbose,
                plot_img=self.plot_img,
                random_state=self.random_state
            )

    def pre_train(self, X):
        """预训练无监督网络"""
        self.unsupervised_dbn.fit(X)
        return self

    def fit(self, X, y, pre_train=True):
        """训练模型 - 支持两种模式"""
        X, y = check_X_y(X, y)
        y_encoded = self.label_encoder.fit_transform(y)
        self.classes_ = self.label_encoder.classes_
        
        # 预训练阶段
        if pre_train:
            self.pre_train(X)
        
        # 根据模式选择训练方式
        if self.fine_tuning:
            # 微调阶段
            self._fine_tuning(X, y_encoded)
        else:
            # 分类器阶段
            self._train_classifier(X, y_encoded)
        return self

    def transform(self, X):
        """特征变换"""
        if self.unsupervised_dbn is None:
            raise ValueError("Model not fitted. Call fit first.")
        return self.unsupervised_dbn.transform(X)

    def predict(self, X):
        """预测 - 根据模式选择预测方法"""
        check_is_fitted(self)
        X = check_array(X)
        
        if self.fine_tuning:
            predictions = self._predict_with_fine_tuning(X)
        else:
            predictions = self._predict_with_classifier(X)
            
        return self.label_encoder.inverse_transform(predictions)

    def predict_proba(self, X):
        """预测概率"""
        check_is_fitted(self)
        X = check_array(X)
        
        if self.fine_tuning:
            return self._predict_proba_fine_tuning(X)
        else:
            return self._predict_proba_classifier(X)

    def score(self, X, y):
        """评分"""
        y_pred = self.predict(X)
        return np.mean(y_pred == y)

    # 抽象方法 - 需要在子类中实现
    @abstractmethod
    def _fine_tuning(self, X, y):
        """微调网络"""
        pass

    @abstractmethod
    def _predict_with_fine_tuning(self, X):
        """使用微调网络预测"""
        pass

    @abstractmethod
    def _predict_proba_fine_tuning(self, X):
        """使用微调网络预测概率"""
        pass

    @abstractmethod
    def _train_classifier(self, X, y):
        """训练分类器"""
        pass

    @abstractmethod
    def _predict_with_classifier(self, X):
        """使用分类器预测"""
        pass

    @abstractmethod
    def _predict_proba_classifier(self, X):
        """使用分类器预测概率"""
        pass

    def save_parameters(self, file_prefix="dbn_model"):
        """保存模型参数"""
        os.makedirs("data", exist_ok=True)
        
        # 保存预训练参数（两种模式都需要）
        if self.unsupervised_dbn and self.unsupervised_dbn.rbm_layers:
            for i, rbm in enumerate(self.unsupervised_dbn.rbm_layers):
                weights = rbm.quadratic_coef.detach().cpu().numpy()
                h_bias = rbm.linear_bias[rbm.num_visible:].detach().cpu().numpy()
                np.save(f"data/{file_prefix}_pretrain_layer{i}_weights.npy", weights)
                np.save(f"data/{file_prefix}_pretrain_layer{i}_bias.npy", h_bias)
            print(f"Pre-trained parameters saved for {len(self.unsupervised_dbn.rbm_layers)} layers")
        
        # 保存微调参数（仅微调模式）
        if self.fine_tuning and hasattr(self, 'fine_tune_network') and self.fine_tune_network is not None:
            for i, layer in enumerate(self.fine_tune_network):
                if isinstance(layer, nn.Linear):
                    weights = layer.weight.detach().cpu().numpy()
                    bias = layer.bias.detach().cpu().numpy()
                    np.save(f"data/{file_prefix}_finetune_layer{i}_weights.npy", weights)
                    np.save(f"data/{file_prefix}_finetune_layer{i}_bias.npy", bias)
            print(f"Fine-tuned parameters saved for {len([l for l in self.fine_tune_network if isinstance(l, nn.Linear)])} layers")
        
        # 保存分类器参数（仅分类器模式）
        if not self.fine_tuning and hasattr(self, 'classifier') and self.classifier is not None:
            import joblib
            classifier_path = f"data/{file_prefix}_classifier.pkl"
            joblib.dump(self.classifier, classifier_path)
            print(f"Classifier parameters saved to {classifier_path}")
        
        print("All parameters saved successfully!")

class PyTorchAbstractSupervisedDBN(AbstractSupervisedDBN):
    """
    PyTorch实现的抽象监督DBN，提供PyTorch相关的通用实现
    """
    def __init__(
        self, 
        classifier_type='logistic',
        clf_C=1.0, 
        clf_iter=100, 
        **kwargs
        ):
        # 确保fine_tuning参数有默认值
        if 'fine_tuning' not in kwargs:
            kwargs['fine_tuning'] = True
            
        super().__init__(**kwargs)
        self.classifier_type = classifier_type
        self.clf_C = clf_C
        self.clf_iter = clf_iter

    def _train_classifier(self, X, y):
        """训练下游分类器"""
        if self.verbose:
            print(f"Training pipline classifier: {self.classifier_type}")
        
        # 提取特征
        X_features = self.transform(X)
        
        # 初始化分类器
        if self.classifier_type == 'logistic':
            from sklearn.linear_model import LogisticRegression
            self.classifier = LogisticRegression(
                C=self.clf_C,
                max_iter=self.clf_iter,
                random_state=self.random_state
            )
        elif self.classifier_type == 'svm':
            from sklearn.svm import SVC
            self.classifier = SVC(
                C=self.clf_C,
                probability=True,
                random_state=self.random_state
            )
        elif self.classifier_type == 'random_forest':
            from sklearn.ensemble import RandomForestClassifier
            self.classifier = RandomForestClassifier(
                n_estimators=100,
                random_state=self.random_state
            )
        else:
            raise ValueError(f"Unsupported classifier type: {self.classifier_type}")

        # 训练分类器
        self.classifier.fit(X_features, y)
        
        if self.verbose:
            train_accuracy = self.classifier.score(X_features, y)
            print(f"Classifier training accuracy: {train_accuracy:.4f}")

    def _predict_with_classifier(self, X):
        """使用分类器预测"""
        X_features = self.transform(X)
        return self.classifier.predict(X_features)

    def _predict_proba_classifier(self, X):
        """使用分类器预测概率"""
        X_features = self.transform(X)
        return self.classifier.predict_proba(X_features)

    # PyTorch相关的通用方法
    def _initialize_layer_with_pretrained(self, layer, rbm):
        """使用预训练权重初始化层"""
        with torch.no_grad():
            weights = rbm.quadratic_coef.detach().cpu().numpy()
            h_bias = rbm.linear_bias[rbm.num_visible:].detach().cpu().numpy()
            layer.weight.data = torch.FloatTensor(weights.T)
            layer.bias.data = torch.FloatTensor(h_bias)

    def _get_activation_layer(self):
        """获取激活函数层"""
        activation_map = {
            'sigmoid': nn.Sigmoid(),
            'relu': nn.ReLU(),
            # 'tanh': nn.Tanh(),
            # 'leaky_relu': nn.LeakyReLU(),
        }
        
        if self.activation_function not in activation_map:
            raise ValueError(f"Unsupported activation: {self.activation_function}")
        
        return activation_map[self.activation_function]

    def _create_optimizer(self, parameters):
        """创建优化器"""
        return SGD(
            parameters,
            lr=self.learning_rate,
            weight_decay=self.l2_regularization
        )

    def _create_data_loader(self, X_tensor, y_tensor, shuffle=True):
        """创建数据加载器"""
        dataset = TensorDataset(X_tensor, y_tensor)
        return DataLoader(
            dataset, 
            batch_size=self.batch_size, 
            shuffle=shuffle
        )

    def get_feature_importance(self):
        """获取特征重要性（适用于分类器模式）"""
        if not self.fine_tuning and hasattr(self.classifier, 'coef_'):
            return np.abs(self.classifier.coef_[0])
        else:
            print("Feature importance is only available in classifier mode")
            return None

    def get_layer_activations(self, X, layer_idx=-1):
        """获取指定层的激活值"""
        if self.unsupervised_dbn is None or self.unsupervised_dbn.rbm_layers is None:
            raise ValueError("Model not fitted")
        
        X_data = X.astype(np.float32)
        
        # 逐层前向传播，直到指定层
        for i, rbm in enumerate(self.unsupervised_dbn.rbm_layers):
            if layer_idx == i:
                with torch.no_grad():
                    hidden_output = rbm.get_hidden(
                        torch.FloatTensor(X_data).to(self.unsupervised_dbn.device)
                    )
                    return hidden_output[:, rbm.num_visible:].cpu().numpy()
            
            with torch.no_grad():
                hidden_output = rbm.get_hidden(
                    torch.FloatTensor(X_data).to(self.unsupervised_dbn.device)
                )
                X_data = hidden_output[:, rbm.num_visible:].cpu().numpy()
        
        return X_data  # 如果layer_idx超出范围，返回最后一层的输出

# =================== 具体的分类DBN实现 =====================
class SupervisedDBNClassification(PyTorchAbstractSupervisedDBN, ClassifierMixin):
    """
    PyTorch实现的监督DBN分类器
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _fine_tuning(self, X, y):
        """微调实现"""
        if self.verbose:
            print("Starting fine-tuning...")
        
        self._build_fine_tune_network()
        
        # 转换为PyTorch张量
        X_tensor = torch.FloatTensor(X).to(self.unsupervised_dbn.device)
        y_tensor = torch.LongTensor(y).to(self.unsupervised_dbn.device)
        
        # 训练微调网络
        self._train_fine_tune_network(X_tensor, y_tensor)

    def _build_fine_tune_network(self):
        """构建微调网络"""
        layers = []
        input_size = self.unsupervised_dbn.rbm_layers[0].num_visible
        
        # 构建隐藏层（使用预训练权重初始化）
        for i, (rbm, hidden_size) in enumerate(zip(self.unsupervised_dbn.rbm_layers, 
                                                 self.hidden_layers_structure)):
            linear_layer = nn.Linear(input_size, hidden_size)
            self._initialize_layer_with_pretrained(linear_layer, rbm)
            layers.append(linear_layer)
            layers.append(self._get_activation_layer())
            
            if self.dropout_p > 0:
                layers.append(nn.Dropout(self.dropout_p))
                
            input_size = hidden_size
        
        # 输出层
        output_layer = nn.Linear(input_size, len(self.classes_))
        layers.append(output_layer)
        
        self.fine_tune_network = nn.Sequential(*layers)
        
        if self.verbose:
            print(f"Built fine-tuning network with {len(layers)} layers")
            print(f"Input size: {self.unsupervised_dbn.rbm_layers[0].num_visible}")
            print(f"Output size: {len(self.classes_)}")

    def _train_fine_tune_network(self, X_tensor, y_tensor):
        """训练微调网络"""
        criterion = nn.CrossEntropyLoss()
        optimizer = self._create_optimizer(self.fine_tune_network.parameters())
        loader = self._create_data_loader(X_tensor, y_tensor, shuffle=True)
        
        self.fine_tune_network.train()
        
        for epoch in range(self.n_iter_backprop):
            running_loss = 0.0
            correct = 0
            total = 0
            
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                outputs = self.fine_tune_network(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
            
            # 打印训练信息
            if self.verbose and (epoch + 1) % 10 == 0:
                accuracy = 100 * correct / total
                avg_loss = running_loss / len(loader)
                print(f"Fine-tuning Epoch {epoch+1}/{self.n_iter_backprop}, "
                      f"Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
        
        if self.verbose:
            final_accuracy = 100 * correct / total
            print("Fine-tuning completed. ")
            print(f"Fine-tuning network training accuracy: {final_accuracy:.2f}%")

    def _predict_with_fine_tuning(self, X):
        """使用微调网络预测 - 具体实现"""
        X_tensor = torch.FloatTensor(X).to(self.unsupervised_dbn.device)
        self.fine_tune_network.eval()
        
        with torch.no_grad():
            outputs = self.fine_tune_network(X_tensor)
            _, predicted = torch.max(outputs, 1)
            return predicted.cpu().numpy()

    def _predict_proba_fine_tuning(self, X):
        """使用微调网络预测概率 - 具体实现"""
        X_tensor = torch.FloatTensor(X).to(self.unsupervised_dbn.device)
        self.fine_tune_network.eval()
        
        with torch.no_grad():
            outputs = self.fine_tune_network(X_tensor)
            return torch.softmax(outputs, dim=1).cpu().numpy()

    def get_network_structure(self):
        """获取网络结构信息"""
        structure = {
            'pretrain_layers': len(self.unsupervised_dbn.rbm_layers),
            'fine_tune_layers': len(list(self.fine_tune_network)) if hasattr(self, 'fine_tune_network') and self.fine_tune_network else 0,
            'hidden_units': self.hidden_layers_structure,
            'num_classes': len(self.classes_),
            'mode': 'fine_tuning' if self.fine_tuning else 'classifier'
        }
        return structure