import os
import datetime
import numpy as np
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
import imageio
import pandas as pd
from plotnine import ggplot, aes, geom_point, ggtitle
# from ggplot import *
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix

def plot_training_curves(train_loss_history, val_loss_history, 
                        train_acc_history, val_acc_history, 
                        save_path=None, show=True):
    """
    绘制训练和验证的损失及准确率曲线
    
    Args:
        train_loss_history: 训练损失历史
        val_loss_history: 验证损失历史  
        train_acc_history: 训练准确率历史
        val_acc_history: 验证准确率历史
        save_path: 图像保存路径
    """
    plt.figure(figsize=(12, 5))
    
    # 损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(train_loss_history, label='Training Loss', color='blue', alpha=0.7)
    plt.plot(val_loss_history, label='Validation Loss', color='red', alpha=0.7)
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(train_acc_history, label='Training Accuracy', color='blue', alpha=0.7)
    plt.plot(val_acc_history, label='Validation Accuracy', color='red', alpha=0.7)
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    # plt.close()

    # 自动保存
    if save_path is None:
        # 生成默认保存路径
        # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f"results/mlp_training_curves_{timestamp}.png"
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Training curves saved to: {save_path}")
    plt.show()
    
    if show:
        plt.show()
    else:
        plt.close()  # 不显示时关闭图像，节省内存

def plot_flattened_images_grid(features: torch.Tensor, grid_size: int = 8, save_path: str = None):
    """
    显示并可选保存前 grid_size * grid_size 个 28x28 灰度图像。

    Args:
        features (torch.Tensor): 形状为 [N, 784] 的张量，每行为一个扁平的 28x28 图像。
        grid_size (int): 图像网格边长（默认 8，即显示前 64 张图像）。
        save_path (str): 如果提供，将保存图像到该路径。
    """
    assert features.dim() == 2 and features.size(1) == 784, "features 应为 [N, 784] 的张量"
    num_images = grid_size * grid_size
    assert features.size(0) >= num_images, f"features 中至少应包含 {num_images} 张图像"

    features_numpy = features[:num_images].detach().cpu().numpy()

    fig, axes = plt.subplots(grid_size, grid_size, figsize=(5, 5))
    for i in range(grid_size):
        for j in range(grid_size):
            idx = i * grid_size + j
            img = features_numpy[idx].reshape(28, 28)
            axes[i, j].imshow(img, cmap='gray')
            axes[i, j].axis('off')

    plt.tight_layout()
    
    if save_path:
        # 确保目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')  # 实际保存图像
        print(f"图像已保存到: {save_path}")
        plt.close()  # 保存后关闭图形
    else:
        plt.show()  # 没有保存路径时显示图像
        plt.close()

def t_SNE(test_loader, qvae_model, use_std=True, point_size=20, alpha=0.6, 
          epochs=None, save_path=None, show=True):
    """
    QVAE版本的t-SNE可视化
    
    Args:
        test_loader: 测试数据加载器
        qvae_model: QVAE模型
        use_std: 是否使用标准差
        point_size: 点大小
        alpha: 透明度
        epochs: 训练轮数
        save_path: 保存路径
        show: 是否显示图像
    """
    features = []
    labels = []
    
    qvae_model.eval()
    device = next(qvae_model.parameters()).device
    
    with torch.no_grad():
        for batch_idx, (example_data, example_targets) in enumerate(test_loader):
            example_data = example_data.to(device)
            
            # QVAE前向传播 - 获取潜变量zeta
            _, _, _, zeta = qvae_model(example_data)
            
            zeta_np = zeta.cpu().numpy()
            
            for idx in range(zeta_np.shape[0]):
                features.append(zeta_np[idx])
                labels.append(example_targets[idx].item())
    
    # 创建DataFrame
    feat_cols = [f'dim_{i}' for i in range(zeta_np.shape[1])]
    df = pd.DataFrame(features, columns=feat_cols)
    df['label'] = labels
    df['label'] = df['label'].apply(lambda i: str(i))
    
    print(f"Extracted {len(features)} samples with {zeta_np.shape[1]} dimensions")
    
    # 执行t-SNE
    print("Running t-SNE...")
    tsne = TSNE(n_components=2, verbose=1, perplexity=30, max_iter=300, random_state=42)
    tsne_results = tsne.fit_transform(df[feat_cols].values)
    
    df_tsne = df.copy()
    df_tsne['x-tsne'] = tsne_results[:, 0]
    df_tsne['y-tsne'] = tsne_results[:, 1]
    
    # 可视化
    # plt.figure(figsize=(10, 8))
    # scatter = plt.scatter(df_tsne['x-tsne'], df_tsne['y-tsne'], 
    #                      c=df_tsne['label'].astype(int), 
    #                      cmap='tab10', s=point_size, alpha=alpha)
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(df_tsne['x-tsne'], df_tsne['y-tsne'], 
                        c=df_tsne['label'].astype(int), 
                        cmap='tab10', s=point_size, alpha=alpha)
    
    # 添加颜色条
    cbar = plt.colorbar(scatter, ax=ax, label='Digit')
    
    # 动态标题和文件名
    training_status = "fully_trained" if epochs and epochs >= 20 else f"epochs_{epochs}"
    title = f't-SNE Visualization of QVAE Latent Space ({training_status})'
    plt.title(title)
    plt.xlabel('t-SNE dimension 1')
    plt.ylabel('t-SNE dimension 2')
    plt.tight_layout()
    
    # 自动保存
    if save_path is None:
        # 生成默认保存路径
        # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f"results/t-SNE_QVAE_{training_status}_{timestamp}.png"
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"t-SNE plot saved to: {save_path}")
    plt.show()
    
    if show:
        plt.show()
    else:
        plt.close()  # 不显示时关闭图像，节省内存
    
    return df_tsne, save_path

def create_tsne_animation(frame_paths, output_path, duration=500):
    """
    从t-SNE帧创建GIF动画（使用统一尺寸）
    """
    
    images = []
    base_size = None
    
    for path in frame_paths:
        if os.path.exists(path):
            img = Image.open(path)
            
            # 统一尺寸
            if base_size is None:
                base_size = img.size
            else:
                if img.size != base_size:
                    img = img.resize(base_size, Image.Resampling.LANCZOS)
            
            # 转换为numpy数组
            img_array = np.array(img)
            images.append(img_array)
    
    if images:
        # 确保所有图像形状一致
        target_shape = images[0].shape
        uniform_images = []
        
        for img in images:
            if img.shape != target_shape:
                # 调整到目标形状
                pil_img = Image.fromarray(img)
                pil_img = pil_img.resize((target_shape[1], target_shape[0]), Image.Resampling.LANCZOS)
                img = np.array(pil_img)
            uniform_images.append(img)
        
        imageio.mimsave(output_path, uniform_images, duration=duration)
        print(f"Training evolution animation saved to: {output_path}")
        print(f"Animation contains {len(uniform_images)} frames, all with shape {target_shape}")
    else:
        print("No frames found to create animation")

# ============ 统计信息分析 ============
def describe_statistic_per_label(test_loader, qvae_model):
    """
    QVAE版本的统计信息分析
    """
    features = []
    labels = []
    
    qvae_model.eval()
    device = next(qvae_model.parameters()).device
    
    with torch.no_grad():
        for batch_idx, (example_data, example_targets) in enumerate(test_loader):
            example_data = example_data.to(device)
            
            # QVAE前向传播
            _, _, _, zeta = qvae_model(example_data)
            
            zeta_np = zeta.cpu().numpy()
            
            for idx in range(zeta_np.shape[0]):
                features.append(zeta_np[idx])
                labels.append(example_targets[idx].item())
    
    feat_cols = [f'dim_{i}' for i in range(zeta_np.shape[1])]
    df = pd.DataFrame(features, columns=feat_cols)
    df['label'] = labels
    df['label'] = df['label'].apply(lambda i: str(i))
    
    print("Statistical description per label:")
    print(df.groupby("label").describe())
    
    return df

# ============ 混淆矩阵 ============
def show_confusion_matrix(test_loader, qvae_model, classifier, device, save_path=None, show=True):
    """
    显示混淆矩阵
    
    Args:
        test_loader: 测试数据加载器
        qvae_model: QVAE模型，用于提取潜变量
        classifier: 分类器模型
        device: 设备
        save_path: 保存路径
        show: 是否显示图像
    """
    y_test = []
    y_pred = []
    
    qvae_model.eval()
    classifier.eval()
    
    with torch.no_grad():
        for batch_idx, (example_data, example_targets) in enumerate(test_loader):
            example_data = example_data.to(device)
            example_targets = example_targets.to(device)
            
            # 关键：先通过QVAE提取潜变量
            _, _, _, zeta = qvae_model(example_data)
            
            # 然后使用潜变量输入分类器
            outputs = classifier(zeta)
            _, predicted = torch.max(outputs.data, 1)
            
            y_pred.extend(predicted.cpu().numpy())
            y_test.extend(example_targets.cpu().numpy())
    
    # 创建混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    labels = [str(i) for i in range(10)]
    
    # 归一化
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # 可视化
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=labels, yticklabels=labels,
           title="Normalized Confusion Matrix",
           ylabel='True label',
           xlabel='Predicted label')
    
    # plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # 添加文本标注
    fmt = '.2f'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    fig.tight_layout()
    # plt.show()
    # 自动保存

    if save_path is None:
        # 生成默认保存路径
        # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f"results/confusion_matrix_{training_status}_{timestamp}.png"
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"confusion matrix saved to: {save_path}")
    plt.show()
    
    if show:
        plt.show()
    else:
        plt.close()  # 不显示时关闭图像，节省内存
    
    return cm