# Copyright (C) 2022-2025 Beijing QBoson Quantum Technology Co., Ltd.
#
# SPDX-License-Identifier: Apache-2.0
import os
import numpy as np
from scipy.ndimage import shift
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_digits
from abc import ABC, abstractmethod

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

import torch
import torch.nn as nn
from torch.optim import SGD
from torch.utils.data import DataLoader, TensorDataset
from dbn_trainer import DBNPretrainer

def translate_image(image, direction):
    "图片转换"
    if direction == "up":
        return shift(image, [-1, 0], mode="constant", cval=0)
    elif direction == "down":
        return shift(image, [1, 0], mode="constant", cval=0)
    elif direction == "left":
        return shift(image, [0, -1], mode="constant", cval=0)
    elif direction == "right":
        return shift(image, [0, 1], mode="constant", cval=0)
    else:
        raise ValueError("Invalid direction. Use 'up', 'down', 'left', or 'right'.")

def load_data(plot_img=False):
    "载入图片数据"
    digits = load_digits()
    images = digits.images  # 8x8 的图像矩阵
    labels = digits.target  # 对应的标签

    # 扩展数据集
    expanded_images = []
    expanded_labels = []
    for image, label in zip(images, labels):
        # 原始图像
        expanded_images.append(image)
        expanded_labels.append(label)
        # 向四个方向平移
        for direction in ["up", "down", "left", "right"]:
            translated_image = translate_image(image, direction)
            expanded_images.append(translated_image)
            expanded_labels.append(label)
            # 将列表转换为 NumPy 数组
    expanded_images = np.array(expanded_images)
    expanded_labels = np.array(expanded_labels)

    # 可视化图像数据和标签
    if plot_img:
        plt.figure(figsize=(16,9))
        for index in range(5):
            plt.subplot(1,5, index + 1)
            plt.imshow(expanded_images[index], origin="lower", cmap="gray")
            plt.title('Training: %i\n' % expanded_labels[index], fontsize = 18)

    # 将图像数据展平为二维数组 (n_samples, 64)
    n_samples = expanded_images.shape[0]
    data = expanded_images.reshape((n_samples, -1))
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        data, expanded_labels, test_size=0.2, random_state=42
    )

    # 使用sklearn的MinMaxScaler进行归一化
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test


        
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
        self.unsupervised_dbn = DBNPretrainer(
            hidden_layers_structure=self.hidden_layers_structure,
            learning_rate_rbm=self.learning_rate_rbm,
            n_epochs_rbm=self.n_epochs_rbm,
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
        if self.unsupervised_dbn and self.unsupervised_dbn._n_layers > 0:
            for i in range(self.unsupervised_dbn._n_layers):
                rbm = self.unsupervised_dbn.get_rbm_layer(i)  # 使用 get_rbm_layer
                weights = rbm.quadratic_coef.detach().cpu().numpy()
                h_bias = rbm.linear_bias[rbm.num_visible:].detach().cpu().numpy()
                np.save(f"data/{file_prefix}_pretrain_layer{i}_weights.npy", weights)
                np.save(f"data/{file_prefix}_pretrain_layer{i}_bias.npy", h_bias)
            print(f"Pre-trained parameters saved for {self.unsupervised_dbn._n_layers} layers")
        
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

class AbstractSupervisedDBNClassifier(AbstractSupervisedDBN):
    """
    抽象监督DBN，提供下游分类器训练和fine-tuning相关的通用工具
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
            train_accuracy = self.classifier.score(X_features, y)*100
            print(f"Classifier training accuracy: {train_accuracy:.2f}%")

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

    def get_layer_activations(self, X, layer_index=None, return_all_layers=False):
        """
        获取指定层或所有层的激活值
        """
        if self.unsupervised_dbn is None or self.unsupervised_dbn._n_layers == 0:
            raise ValueError("No RBM layers found. Please fit the model first.")
        
        X_data = X.astype(np.float32)
        all_activations = {}
        
        # 逐层前向传播计算激活值，直到指定层
        for i in range(self.unsupervised_dbn._n_layers):
            rbm = self.unsupervised_dbn.get_rbm_layer(i)
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X_data).to(self.unsupervised_dbn.device)
                hidden_output = rbm.get_hidden(X_tensor)
                
                # 提取隐藏层激活值（去掉可见层部分）
                layer_activation = hidden_output[:, rbm.num_visible:].cpu().numpy()
                all_activations[i] = layer_activation
                
                # 更新输入数据为当前层输出，用于下一层
                X_data = layer_activation
            
            # 如果只需要特定层，且已经到达该层，可以提前终止
            if layer_index is not None and i == layer_index and not return_all_layers:
                return layer_activation
        
        if return_all_layers:
            return all_activations
        else:
            # 返回指定层或最后一层
            if layer_index is not None:
                if layer_index in all_activations:
                    return all_activations[layer_index]
                else:
                    raise ValueError(f"Layer index {layer_index} out of range. Model has {self.unsupervised_dbn._n_layers} layers.")
            else:
                # 返回最后一层
                return all_activations[self.unsupervised_dbn._n_layers - 1]


# =================== 具体的分类DBN实现 =====================
class SupervisedDBNClassification(AbstractSupervisedDBNClassifier, ClassifierMixin):
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
        
        if self.unsupervised_dbn._n_layers > 0:
            first_rbm = self.unsupervised_dbn.get_rbm_layer(0)
            input_size = first_rbm.num_visible
        
        # 构建隐藏层（使用预训练权重初始化）
        for i in range(self.unsupervised_dbn._n_layers):  # 使用新的接口方法遍历层
            rbm = self.unsupervised_dbn.get_rbm_layer(i)
            hidden_size = self.hidden_layers_structure[i]
            
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
            print(f"Input size: {self.unsupervised_dbn.get_rbm_layer(0).num_visible}")
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
            'pretrain_layers': self.unsupervised_dbn._n_layers,
            'fine_tune_layers': len(list(self.fine_tune_network)) if hasattr(self, 'fine_tune_network') and self.fine_tune_network else 0,
            'hidden_units': self.hidden_layers_structure,
            'num_classes': len(self.classes_),
            'mode': 'fine_tuning' if self.fine_tuning else 'classifier'
        }
        return structure

class RBMVisualizer:
    def __init__(self, result_dir='results'):
        """可视化工具类
        
        Args:
            result_dir (str): 结果保存目录
        """
        self.result_dir = result_dir
        self._ensure_result_dir()
        
    def _ensure_result_dir(self):
        """确保结果目录存在"""
        os.makedirs(self.result_dir, exist_ok=True)

    # 绘制权重
    def plot_weights(self, rbm, n_visible=64, grid_shape=(8, 16), figsize=(16, 7), 
                     title_suffix="RBM Weights", save_as="qbm_weights", save_pdf=False):
        """绘制RBM权重
        
        Args:
            rbm: RBM模型
            n_visible (int): 可见单元数量
            grid_shape (tuple): 网格形状 (rows, cols)
            figsize (tuple): 图形大小
            title_suffix (str): 标题后缀
            save_as (str): 保存文件名
            save_pdf (bool): 是否保存为PDF
        """
        weights = rbm.quadratic_coef.detach().cpu().numpy()
    
        fig, axes = plt.subplots(grid_shape[0], grid_shape[1], 
                                 gridspec_kw={'wspace':0.1, 'hspace':0.1}, 
                                 figsize=figsize)
        fig.suptitle(f'{rbm.num_hidden} components extracted by RBM - {title_suffix}', fontsize=16)
        fig.subplots_adjust()
    
        for i, ax in enumerate(axes.flatten()):
            if i < weights.shape[1]:
                # 重塑权重为图像形状
                weight_img = weights[:, i].reshape(int(np.sqrt(n_visible)), int(np.sqrt(n_visible)))
                ax.imshow(weight_img, cmap=plt.cm.gray)
            ax.axis('off')
    
        # 保存结果
        if save_pdf:
            plt.savefig(f'{self.result_dir}/{save_as}.pdf', 
                        dpi=300, bbox_inches='tight', format='pdf')
        plt.show()
    
    def plot_confusion_matrix(self, y_true, y_pred, 
                              title_suffix="", save_as="confusion_matrix", save_pdf=False):
        """绘制混淆矩阵
        
        Args:
            y_true: 真实标签
            y_pred: 预测标签
            title_suffix (str): 标题后缀
            save_as (str): 保存文件名
            save_pdf (bool): 是否保存为PDF
        """
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix ({title_suffix})', fontsize=18)
        plt.xlabel('Predicted Label', fontsize=16)
        plt.ylabel('True Label', fontsize=16)
        
        if save_pdf:
            plt.savefig(f'{self.result_dir}/{save_as}_{title_suffix}.pdf', 
                        dpi=300, bbox_inches='tight', format='pdf')
        plt.tight_layout()
        plt.show()

    def plot_reconstructed_images(self, rbm, X, y, layer_index=0, n_images=10, 
                                  title_suffix="", save_pdf=False, img_shape=None):
        """
        绘制原始和重构图像的对比
        
        Args:
            X: 输入图像数据
            y: 图像标签
            layer_index: 使用的RBM层索引
            n_images: 要显示的图像数量
            title_suffix: 标题后缀
            save_pdf: 是否保存为PDF
            img_shape: 图像形状，如(8,8)。如果为None，则尝试自动推断
        """
        
        # 限制图像数量
        n_images = min(n_images, X.shape[0])
        X_sample = X[:n_images]
        y_sample = y[:n_images]
        
        # 重构图像
        X_recon, recon_errors = rbm.reconstruct(X_sample, layer_index)
        
        # 推断图像形状
        if img_shape is None:
            n_features = X_sample.shape[1]
            img_size = int(np.sqrt(n_features))
            if img_size * img_size == n_features:
                img_shape = (img_size, img_size)
        
        # 创建图形
        fig, axes = plt.subplots(2, n_images, figsize=(2*n_images, 4))
        if n_images == 1:
            axes = axes.reshape(2, 1)
        
        # 设置标题
        plt.suptitle(f'Original vs Reconstructed Images ({title_suffix})', fontsize=16)
        
        # 绘制图像
        for i in range(n_images):
            # 原始图像
            if img_shape[0] * img_shape[1] == X_sample.shape[1]:
                axes[0, i].imshow(X_sample[i].reshape(img_shape), cmap='gray')
            else:
                # 如果形状不匹配，显示前img_shape[0]*img_shape[1]个像素
                axes[0, i].imshow(X_sample[i][:img_shape[0]*img_shape[1]].reshape(img_shape), cmap='gray')
            axes[0, i].set_title(f'Label: {y_sample[i]}', fontsize=10)
            axes[0, i].axis('off')
            
            # 重构图像
            if img_shape[0] * img_shape[1] == X_recon.shape[1]:
                axes[1, i].imshow(X_recon[i].reshape(img_shape), cmap='gray')
            else:
                axes[1, i].imshow(X_recon[i][:img_shape[0]*img_shape[1]].reshape(img_shape), cmap='gray')
            axes[1, i].set_title(f'Recon (err: {recon_errors[i]:.4f})', fontsize=10)
            axes[1, i].axis('off')
        
        # 添加y轴标签
        axes[0, 0].set_ylabel('Original', rotation=90, size=12)
        axes[1, 0].set_ylabel('Reconstructed', rotation=90, size=12)
        
        plt.tight_layout()
        
        # 保存结果
        if save_pdf:
            plt.savefig(f'results/reconstructed_images_{title_suffix}.pdf', 
                        dpi=300, bbox_inches='tight', format='pdf')
        
        plt.show()
        
        # 打印平均重构误差
        avg_error = np.mean(recon_errors)
        print(f"Average reconstruction error: {avg_error:.4f}")
        
        return recon_errors
