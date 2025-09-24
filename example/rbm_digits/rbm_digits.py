import os
import numpy as np
from scipy.ndimage import shift
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix

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
            _ensure_result_dir()
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