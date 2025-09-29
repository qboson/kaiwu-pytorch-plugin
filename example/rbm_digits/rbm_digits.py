import os
import numpy as np
from scipy.ndimage import shift
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_digits
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix

from kaiwu.torch_plugin import UnsupervisedDBN, DBNTrainer

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

class RBMPretrainer(BaseEstimator, TransformerMixin):
    """
    Scikit-learn兼容的RBM预训练
    """
    def __init__(
        self,
        n_components=100,
        learning_rate_rbm=0.1,
        n_epochs_rbm=10,
        batch_size=100,
        verbose=True,
        shuffle=True,
        drop_last=False,
        plot_img=False,
        random_state=None
    ):
        self.n_components = n_components
        self.learning_rate_rbm = learning_rate_rbm
        self.n_epochs_rbm = n_epochs_rbm
        self.batch_size = batch_size
        self.verbose = verbose
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.plot_img = plot_img
        self.random_state = random_state
        
        # 创建模型和训练器
        self._rbm = UnsupervisedDBN([n_components])
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
        self._rbm.create_rbm_layer(X.shape[1])
        self._trainer.train(self._rbm, X)
        return self

    def transform(self, X):
        """特征变换"""
        return self._rbm.transform(X)

    # 提供访问RBM层的方法
    def rbm_layer(self, index):
        """获取指定RBM层"""
        return self._rbm.get_rbm_layer(index)

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
        # X_recon, recon_errors = rbm.reconstruct_get_hidden(X_sample, layer_index)
        
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