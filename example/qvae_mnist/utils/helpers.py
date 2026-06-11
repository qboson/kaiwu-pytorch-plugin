# -*- coding: utf-8 -*-
"""
Unsorted helper functions
"""
import os
import datetime
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.colors import LogNorm
import gif
import imageio
from PIL import Image
import torch
from torch.distributions import Exponential
from sklearn.manifold import TSNE
from torchvision import transforms
from torchmetrics.image.fid import FrechetInceptionDistance
from typing import Union, Optional

import logging
logger = logging.getLogger(__name__)

def save_list_to_txt(filename, data):
    with open(filename, "w") as f:
        for value in data:
            f.write(f"{value:.6f}\n")

def plot_training_curves(
    train_loss_history,
    val_loss_history,
    train_acc_history,
    val_acc_history,
    save_path=None,
    show=True,
):
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
    plt.plot(train_loss_history, label="Training Loss", color="blue", alpha=0.7)
    plt.plot(val_loss_history, label="Validation Loss", color="red", alpha=0.7)
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(train_acc_history, label="Training Accuracy", color="blue", alpha=0.7)
    plt.plot(val_acc_history, label="Validation Accuracy", color="red", alpha=0.7)
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
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

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Training curves saved to: {save_path}")
    plt.show()

    if show:
        plt.show()
    else:
        plt.close()  # 不显示时关闭图像，节省内存

#@title t-SNE Visualization for QVAE
def t_SNE(
    test_loader,
    qvae_model,
    point_size=20,
    alpha=0.6,
    epochs=None,
    save_path=None,
    show=True,
):
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
    feat_cols = [f"dim_{i}" for i in range(zeta_np.shape[1])]
    df = pd.DataFrame(features, columns=feat_cols)
    df["label"] = labels
    df["label"] = df["label"].apply(lambda i: str(i))

    logger.info(f"Extracted {len(features)} samples with {zeta_np.shape[1]} dimensions")

    # 执行t-SNE
    logger.info("Running t-SNE...")
    tsne = TSNE(n_components=2, verbose=1, perplexity=30, max_iter=500, random_state=42)
    tsne_results = tsne.fit_transform(df[feat_cols].values)

    df_tsne = df.copy()
    df_tsne["x-tsne"] = tsne_results[:, 0]
    df_tsne["y-tsne"] = tsne_results[:, 1]

    # 可视化
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(
        df_tsne["x-tsne"],
        df_tsne["y-tsne"],
        c=df_tsne["label"].astype(int),
        cmap="tab10",
        s=point_size,
        alpha=alpha,
    )

    # 添加颜色条
    cbar = plt.colorbar(scatter, ax=ax, label="Digit")

    # 动态标题和文件名
    # training_status = "fully_trained" if epochs and epochs >= 20 else f"epochs_{epochs}"
    training_status = f"epochs_{epochs}"
    title = f"t-SNE Visualization of QVAE Latent Space ({training_status})"
    plt.title(title)
    plt.xlabel("t-SNE dimension 1")
    plt.ylabel("t-SNE dimension 2")
    plt.tight_layout()

    # 自动保存
    if save_path is None:
        # 生成默认保存路径
        # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f"results/t-SNE_QVAE_{training_status}_{timestamp}.png"

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    logger.info(f"t-SNE plot saved to: {save_path}")
    plt.show()

    if show:
        plt.show()
    else:
        plt.close()  # 不显示时关闭图像，节省内存

    return df_tsne, save_path, training_status

def create_tsne_animation(frame_paths, output_path, animation_filename="qvae_training_evolution.gif", duration=500):
    """
    从t-SNE帧创建GIF动画
    """
    if not frame_paths:
        logger.warning("No frames found to create animation")
        return
        
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
                pil_img = pil_img.resize(
                    (target_shape[1], target_shape[0]), Image.Resampling.LANCZOS
                )
                img = np.array(pil_img)
            uniform_images.append(img)
            
        animation_path = os.path.join(output_path, animation_filename)
        imageio.mimsave(animation_path, uniform_images, duration=duration)
        logger.info(f"Training evolution animation saved to: {animation_path}")
        logger.info(f"Animation contains {len(uniform_images)} frames, all with shape {target_shape}")
    else:
        logger.warning("No valid frames found to create animation")

# QVAE evaluation helper functions
def plot_flattened_images_grid(
    features: torch.Tensor, grid_size: int = 8, save_path: str = None
):
    assert features.dim() == 2 and features.size(1) == 784, "features 应为 [N, 784] 的张量"
    num_images = grid_size * grid_size
    assert features.size(0) >= num_images, f"features 中至少应包含 {num_images} 张图像"

    features_numpy = features[:num_images].detach().cpu().numpy()

    fig, axes = plt.subplots(grid_size, grid_size, figsize=(5, 5))
    for i in range(grid_size):
        for j in range(grid_size):
            idx = i * grid_size + j
            img = features_numpy[idx].reshape(28, 28)
            axes[i, j].imshow(img, cmap="gray")
            axes[i, j].axis("off")

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved flattened images grid to: {save_path}")
        plt.close()
    else:
        plt.show()
        plt.close()

def init_qvae_sampler(model):
    import kaiwu as kw
    from kaiwu.classical import SimulatedAnnealingOptimizer
    from kaiwu.cim import CIMOptimizer, PrecisionReducer

    if model.sampler_type == 'cim':
        kw.common.CheckpointManager.save_dir = './tmp'
        sampler = CIMOptimizer(task_name="qvae_sampling", wait=True)
        worker = PrecisionReducer(
            sampler,
            precision=8,
            truncated_precision=10,
            target_bits=550,
            only_feasible_solution=False,
        )
    elif model.sampler_type == 'sa':
        worker = SimulatedAnnealingOptimizer(size_limit=100, alpha=0.99)
    else:
        raise ValueError(f"Unsupported sampler type: {model.sampler_type}")
    return worker


def _apply_train_bias(model, generated_x: torch.Tensor) -> torch.Tensor:
    bias = None
    if hasattr(model, 'train_bias') and model.train_bias is not None:
        bias = model.train_bias
    elif hasattr(model, '_train_bias') and model._train_bias is not None:
        bias = model._train_bias
    if bias is not None:
        return generated_x + bias.to(generated_x.device)
    return generated_x


def generate_qvae_images(model, save_path, grid_size=8):
    model.eval()
    sampler = init_qvae_sampler(model)
    z = model.bm.sample(sampler)
    zeta = torch.distributions.Exponential(model.dist_beta).sample(z.shape).to(z.device)
    zeta = torch.where(z == 0.0, zeta, 1 - zeta)
    with torch.no_grad():
        generated_x = model.decoder(zeta)
        generated_x = _apply_train_bias(model, generated_x)
        generated_x = torch.sigmoid(generated_x)

    generated_save_path = os.path.join(save_path, "generated_x.png")
    print(f"Visualizing generated images, saving to: {generated_save_path}")
    plot_flattened_images_grid(
        generated_x.cpu(), grid_size=grid_size, save_path=generated_save_path
    )
    return generated_x, generated_save_path


def get_real_images(dataloader, n_images=10000):
    images = []
    for batch_imgs, _ in dataloader:
        images.append(batch_imgs)
        if sum(img.shape[0] for img in images) >= n_images:
            break
    return torch.cat(images, dim=0)[:n_images]


def generate_qvae_samples(model, dist_beta, n_images=10000, batch_size=64):
    model.eval()
    imgs = []
    sampler = init_qvae_sampler(model)
    with torch.no_grad():
        for _ in tqdm(range(n_images // batch_size)):
            z = model.bm.sample(sampler)
            shape = z.shape
            smoothing_dist = Exponential(dist_beta)
            # 从平滑分布采样
            zeta = smoothing_dist.sample(shape)
            zeta = zeta.to(z.device)
            zeta = torch.where(z == 0.0, zeta, 0)
            # zeta = torch.randn(256, 256).to(device)
            generated_x = model.decoder(zeta)

            generated_x = generated_x + model._train_bias

            generated_x = torch.sigmoid(generated_x)

            imgs.append(generated_x)
    return torch.cat(imgs, dim=0)[:n_images]

class FIDImagePreprocessor:
    """预处理图像以适配 InceptionV3（FID 模型）输入要求"""
    
    def __init__(self, target_size: tuple = (299, 299)):
        self.target_size = target_size
        self.resize = transforms.Resize(target_size, antialias=True)
    
    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        # 1. 转换为 (B, 1, 28, 28) —— 输入来自 MNIST 形状
        images = images.view(-1, 1, 28, 28)
        
        # 2. 扩展为三通道 (N, 3, 28, 28)
        images = images.repeat(1, 3, 1, 1)
        
        # 3. 缩放到 (299, 299)
        images = self.resize(images)
        return images

class FIDCalculator:
    """FID 计算器，封装内部状态和批次更新逻辑"""
    
    def __init__(self, device: torch.device, feature: int = 64):
        self.device = device
        self.fid = FrechetInceptionDistance(feature=feature).to(device)
        self.preprocessor = FIDImagePreprocessor()
    
    def update_batch(self, images: torch.Tensor, real: bool):
        """处理并更新单一批次"""
        # 输入假定为 [0,1] float，先缩放至 [0,255] uint8
        images_uint8 = (images * 255).clamp(0, 255).to(torch.uint8)
        # 预处理（形状、通道、尺寸）
        processed = self.preprocessor(images_uint8)
        # 移动到设备并更新
        self.fid.update(processed.to(self.device), real=real)
    
    def compute(self) -> float:
        """计算最终 FID 分数"""
        return self.fid.compute().item()
    
    def reset(self):
        self.fid.reset()

def compute_fid_in_batches(
    fake_imgs: Union[torch.Tensor, list, tuple],
    real_imgs: Union[torch.Tensor, list, tuple],
    device: torch.device,
    batch_size: int = 64
) -> float:
    """计算 FID"""
    # 确保输入为 Tensor，并转为 float 类型（便于缩放）
    if not isinstance(fake_imgs, torch.Tensor):
        fake_imgs = torch.tensor(fake_imgs, dtype=torch.float32)
    if not isinstance(real_imgs, torch.Tensor):
        real_imgs = torch.tensor(real_imgs, dtype=torch.float32)
    
    # 创建计算器
    calculator = FIDCalculator(device)
    
    # 更新真实图像批次
    for i in range(0, len(real_imgs), batch_size):
        batch = real_imgs[i:i+batch_size]
        calculator.update_batch(batch, real=True)
    
    # 更新生成图像批次
    for i in range(0, len(fake_imgs), batch_size):
        batch = fake_imgs[i:i+batch_size]
        calculator.update_batch(batch, real=False)
    
    return calculator.compute()

# def compute_fid_in_batches(fake_imgs, real_imgs, device, batch_size=64):
#     fid = FrechetInceptionDistance(feature=64).to(device)

#     def preprocess(images):
#         # 转换为图像格式 (B, 1, 28, 28)
#         images = images.view(-1, 1, 28, 28)
#         # 扩展为三通道
#         images = images.repeat(1, 3, 1, 1)
#         # 调整大小到 299x299
#         resize = transforms.Resize((299, 299), antialias=True)
#         return resize(images)

#     # 如果不是 tensor，先转成 tensor
#     if not isinstance(fake_imgs, torch.Tensor):
#         fake_imgs = torch.tensor(fake_imgs, dtype=torch.uint8)
#     if not isinstance(real_imgs, torch.Tensor):
#         real_imgs = torch.tensor(real_imgs, dtype=torch.uint8)

#     # 归一化到 [0, 255] 并转为 uint8（假定输入是 float 在 [0,1] 范围）
#     fake_imgs = (fake_imgs * 255).clamp(0, 255).to(torch.uint8)
#     real_imgs = (real_imgs * 255).clamp(0, 255).to(torch.uint8)

#     # 转换为图像并更新 FID
#     for i in range(0, len(real_imgs), batch_size):
#         batch = real_imgs[i : i + batch_size]
#         batch = preprocess(batch)
#         fid.update(batch.to(device), real=True)

#     for i in range(0, len(fake_imgs), batch_size):
#         batch = fake_imgs[i : i + batch_size]
#         batch = preprocess(batch)
#         fid.update(batch.to(device), real=False)

#     return fid.compute().item()

def evaluate_qvae_fid(trainer, fake_imgs, real_imgs, device, save_path=None, batch_size=64):
    """
    计算 FID 分数，适用于输入为 (N, 784) 的展平图像（如 MNIST）

    参数:
        fake_imgs: 生成图像，shape = (N, 784)
        real_imgs: 真实图像，shape = (M, 784)
        device: 使用 'cuda' 或 'cpu'
    返回:
        FID 分数
    """
    fid_score = compute_fid_in_batches(
        fake_imgs, real_imgs, 
        device=device, 
        batch_size=batch_size
    )
    if save_path is not None:
        fid_result_path = os.path.join(save_path, "fid_results.txt")
        with open(fid_result_path, "w") as f:
            f.write(f"FID分数: {fid_score:.4f}\n")
            f.write(f"训练轮次: {trainer.num_epochs}\n")
            f.write(f"潜在维度: {trainer.model._latent_dimensions}\n")
            f.write(f"评估时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        print(f"Saved FID result to {fid_result_path}")
    return fid_score


#@title Helper Functions
def plot_latent_space(zeta, label, output='', dimensions=2):
    logger.info("Plotting Latent Space")
    fig = plt.figure()
    if dimensions==0:
        i=0
        j=1
        # Create plot
        plt.title('Latent Space Representation for MNIST')
        lz0=zeta[:,i]
        lz1=zeta[:,j]
        ll=label
        df=pd.DataFrame(list(zip(lz0,lz1,ll)),columns=['z0','z1','label'])
        for l in range(10):
            maskeddf=df.loc[df.label==l]
            plt.scatter(maskeddf['z0'], maskeddf['z1'], alpha=0.5, s=10, label=l, cmap="inferno")
        plt.xlabel(r"$\zeta_{0}$ ".format(i))
        plt.ylabel(r"$\zeta_{0}$ ".format(j))
        plt.legend(loc='upper right', bbox_to_anchor=(1.135,1.))
    else:
        plt.title('{0}-dimensional Latent Space Representation for MNIST'.format(dimensions))
        plt.axis('off')
        idx=1
        for i in range(0,dimensions):
            for j in range(0,dimensions):
                # Create plot
                ax = fig.add_subplot(dimensions,dimensions, idx)
                lz0=zeta[:,i]
                lz1=zeta[:,j]
                ll=label
                df=pd.DataFrame(list(zip(lz0,lz1,ll)),columns=['z0','z1','label'])
                for l in range(10):
                    maskeddf=df.loc[df.label==l]
                    ax.scatter(maskeddf['z0'], maskeddf['z1'], alpha=0.5, s=10, label=l, cmap="inferno")
                ax.set_xlabel(r"$\zeta_{0}$".format(i))
                ax.set_ylabel(r"$\zeta_{0}$ ".format(j))
                if idx==dimensions:
                    ax.legend(loc='upper right', bbox_to_anchor=(1.3,1.))
                idx+=1
    fig = plt.gcf()
    fig.savefig(output+".pdf")

def plot_image(image, layer, vmin=None, vmax=None):
    fig = plt.figure(figsize=(20,20))

    cbar = plt.colorbar(fraction=0.0455)
    cbar.set_label(r'Energy (MeV)', y=0.83)
    cbar.ax.tick_params()
   
    xticks = range(sizes[layer*2 + 1])
    yticks = range(sizes[layer*2])
    if layer == 0:
        xticks = xticks[::10]
    plt.xticks(xticks)
    plt.yticks(yticks)
    plt.xlabel(r'$\eta$ Cell ID')
    plt.ylabel(r'$\phi$ Cell ID')

    plt.tight_layout()
    return im


    ax_idx=0
    print(x_true.shape)
    for i in range(n_samples):
    # for i in range(n_rows):
    #     for j in range(n_cols): 
        if ax_idx%n_cols==0:
            ax_idx+=1
        current_ax=plt.subplot(n_rows, n_cols , i+1)
        plt.imshow(x_true[i].reshape((28, 28)))
        plt.gray()
        current_ax.get_xaxis().set_visible(False)
        current_ax .get_yaxis().set_visible(False)
    fig = plt.gcf()
    # fig.tight_layout()
    fig.savefig(output, bbox_inches='tight')

# Make images respond to changes in the norm of other images (e.g. via the
# "edit axis, curves and images parameters" GUI on Qt), but be careful not to
# recurse infinitely!
def update(changed_image):
    for im in images:
        if (changed_image.get_cmap() != im.get_cmap()
                or changed_image.get_clim() != im.get_clim()):
            im.set_cmap(changed_image.get_cmap())
            im.set_clim(changed_image.get_clim())

def plot_calo_jet_generated(x_recon, n_samples=5, output="./output/testCalo.png", do_gif=False):
    for i in range(n_samples):

        plt.figure(figsize=(10, 3.5))

        images=[]
        for j in range(0,len(x_recon)):
            x_out=x_recon[j]
            if j==0:
                shape=(3,96)
            elif j==1:
                shape=(12,12)
            else:
                shape=(12,6)

            reco_image=x_out[i].reshape(shape)

            #TODO this is arbitrary...
            minVal=reco_image.min(1,keepdim=True)[0]*15
            minVal=reco_image.min(1,keepdim=True)[0]

            reco_image[reco_image<minVal]=0            
            ax1 = plt.subplot(1, len(x_recon), j + 1)
            ax1.set_box_aspect(1)
            if j==0:
                ax1.set_ylabel(r'$\phi$ Cell ID')
            ax1.set_xlabel(r'$\eta$ Cell ID')

            im=plt.imshow(reco_image,aspect="auto",cmap="cool",interpolation="none",norm=LogNorm(None,None))
            images.append(im)

        fig = plt.gcf()
        # fig.subplots_adjust(right=0.8)
        # Find the min and max of all colors for use in setting the color scale.
        vmin = min(image.get_array().min() for image in images)
        vmax = max(image.get_array().max() for image in images)
        norm = colors.LogNorm(vmin=1e-5, vmax=vmax)
        for im in images:
            im.set_norm(norm)

        cbar=fig.colorbar(images[0], ax=[fig.axes], orientation='vertical', fraction=.02)    
        cbar.ax.set_ylabel('Energy Fraction', rotation=270)

        for im in images:
            im.callbacksSM.connect('changed', update)
        fig.suptitle('Geant4 vs. sVAE Calorimeter shower')
        # plt.tight_layout()
        fig.savefig(output.replace(".png","_{0}.png".format(i)))

def plot_calo_image_sequence(x_true, x_recon, input_dimension, layer=0, n_samples=5, output="./output/testCalo.png", do_gif=False):
    for i in range(n_samples):

        plt.figure(figsize=(10, 7))
        # plt.subplots_adjust(right=0.8)

        images=[]
        for j in range(0,len(x_true)):
            x=x_true[j]
            x_out=x_recon[j]
            # plt.ylim([0,12])

            ax1 = plt.subplot(2, len(x_true), j + 1)
            ax1.set_box_aspect(1)
            if j==0:
                ax1.set_ylabel(r'$\phi$ Cell ID')
                # ax1.get_xaxis().set_visible(False)
            # else:
                # ax1.get_xaxis().set_visible(False)
                # ax1.get_yaxis().set_visible(False)

            im=plt.imshow(x[i],aspect="auto",cmap="cool",interpolation="none",norm=LogNorm(None,None))
            images.append(im)
            reco_image=x_out[i].reshape(x[i].shape)
            #TODO this is arbitrary...
            minVal=reco_image.min(1,keepdim=True)[0]*15
            minVal=reco_image.min(1,keepdim=True)[0]

            reco_image[reco_image<minVal]=0
        
            ax2 = plt.subplot(2, len(x_true), j + 1 + len(x_true))
            ax2.set_box_aspect(1)
            if j==0:
                ax2.set_ylabel(r'$\phi$ Cell ID')
            ax2.set_xlabel(r'$\eta$ Cell ID')
            # else:
            #     ax2.get_yaxis().set_visible(False)

            im2=plt.imshow(reco_image,aspect="auto",cmap="cool",interpolation="none",norm=LogNorm(None,None))
            # images.append(im2)

        fig = plt.gcf()
        # fig.subplots_adjust(right=0.8)
        # Find the min and max of all colors for use in setting the color scale.
        vmin = min(image.get_array().min() for image in images)
        vmax = max(image.get_array().max() for image in images)
        norm = colors.LogNorm(vmin=1e-5, vmax=vmax)
        for im in images:
            im.set_norm(norm)

        cbar=fig.colorbar(images[0], ax=[fig.axes], orientation='vertical', fraction=.02)    
        cbar.ax.set_ylabel('Energy Fraction', rotation=270)

        for im in images:
            im.callbacksSM.connect('changed', update)
        fig.suptitle('Geant4 vs. sVAE Calorimeter shower')
        # plt.tight_layout()
        fig.savefig(output.replace(".png","_{0}.png".format(i)))

@gif.frame
def plot_calo_images(x_true, x_recon, layer=0, n_samples=5, output="./output/testCalo.png", do_gif=False):
    plt.figure(figsize=(10, 4.5))
    axes_rec=[]
    axes_true=[]
    images=[]
    for i in range(n_samples):
        # plot original image
        ax1 = plt.subplot(2, n_samples, i + 1)
        im = plt.imshow(x_true[i],
        # aspect=float(96)/3,
        norm=LogNorm(None,None)
        )

        if i==0:    
            plt.ylabel(r'$\phi$ Cell ID')
            ax1.get_xaxis().set_visible(False)

        else:
            ax1.get_xaxis().set_visible(False)
            ax1.get_yaxis().set_visible(False)
        
        reco_image=x_recon[i].reshape(x_true[i].shape)
        #TODO this is arbitrary...
        minVal=reco_image.min(1,keepdim=True)[0]*5
        minVal=reco_image.min(1,keepdim=True)[0]

        reco_image[reco_image<minVal]=0
        
        ax2 = plt.subplot(2, n_samples, i + 1 + n_samples)
        plt.imshow(reco_image,
            norm=LogNorm(None,None),
        )
        if i==0:    
            plt.ylabel(r'$\phi$ Cell ID')
        else:
             ax2.get_yaxis().set_visible(False)
        plt.xlabel(r'$\eta$ Cell ID')
        axes_true.append(ax1)
        axes_rec.append(ax2)

    # cbar = plt.colorbar(fraction=0.0455)
    # cbar.set_label(r'Energy (MeV)', y=0.83)
    # cbar.ax.tick_params()
   
    # xticks = range(sizes[layer*2 + 1])
    # yticks = range(sizes[layer*2])
    # if layer == 0:
    #     xticks = xticks[::10]
    # plt.xticks(xticks)
    # plt.yticks(yticks)

    if not do_gif:
        plt.tight_layout()
        fig = plt.gcf()
    #  plt.show()
        fig.savefig(output)
        import sys
        sys.exit()
    #     im = plt.imshow(image,
    #            aspect=float(sizes[layer*2 + 1])/sizes[layer*2],
    #            interpolation='nearest',
    #            norm=LogNorm(vmin, vmax)
    # )

#         plt.imshow(x_true[i].reshape((28, 28)))
#         # plt.gray()
#         # ax.get_xaxis().set_visible(False)
#         # ax.get_yaxis().set_visible(False)

#         ax = plt.subplot(2, n_samples, i + 1 + n_samples)
#         decImg=x_recon[i].reshape((28, 28))
#         plt.imshow(decImg)
#         plt.gray()
#         ax.get_xaxis().set_visible(False)
#         ax.get_yaxis().set_visible(False)
#     fig = plt.gcf()
#   #  plt.show()
#     fig.savefig(output)


def plot_MNIST_output(x_true, x_recon, n_samples=5, output="./output/testVAE.png"):
    plt.figure(figsize=(10, 4.5))
    for i in range(n_samples):
        # plot original image
        ax = plt.subplot(2, n_samples, i + 1)
        plt.imshow(x_true[i].reshape((28, 28)))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(2, n_samples, i + 1 + n_samples)
        decImg=x_recon[i].reshape((28, 28))
        plt.imshow(decImg)
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    fig = plt.gcf()
  #  plt.show()
    fig.savefig(output)

#@title Helper Functions
def plot_autoencoder_outputs(model, n, dims):
    decoded_imgs = model.decode(x_test)

    # number of example digits to show
    n = 5
    plt.figure(figsize=(10, 4.5))
    for i in range(n):
        # plot original image
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(x_test[i].reshape(*dims))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        if i == n/2:
            ax.set_title('Original Images')

        # plot reconstruction 
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(decoded_imgs[i].reshape(*dims))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        if i == n/2:
            ax.set_title('Reconstructed Images')
    plt.show()

def plot_loss(history):
    historydf = pd.DataFrame(history.history, index=history.epoch)
    plt.figure(figsize=(8, 6))
    historydf.plot(ylim=(0, historydf.values.max()))
    plt.title('Loss: %.3f' % history.history['loss'][-1])
    
def plot_compare_histories(history_list, name_list, plot_accuracy=True):
    dflist = []
    min_epoch = len(history_list[0].epoch)
    losses = []
    for history in history_list:
        h = {key: val for key, val in history.history.items() if not key.startswith('val_')}
        dflist.append(pd.DataFrame(h, index=history.epoch))
        min_epoch = min(min_epoch, len(history.epoch))
        losses.append(h['loss'][-1])

    historydf = pd.concat(dflist, axis=1)

    metrics = dflist[0].columns
    idx = pd.MultiIndex.from_product([name_list, metrics], names=['model', 'metric'])
    historydf.columns = idx
    
    plt.figure(figsize=(6, 8))

    ax = plt.subplot(211)
    historydf.xs('loss', axis=1, level='metric').plot(ylim=(0,1), ax=ax)
    plt.title("Training Loss: " + ' vs '.join([str(round(x, 3)) for x in losses]))
    
    if plot_accuracy:
        ax = plt.subplot(212)
        historydf.xs('acc', axis=1, level='metric').plot(ylim=(0,1), ax=ax)
        plt.title("Accuracy")
        plt.xlabel("Epochs")
    
    plt.xlim(0, min_epoch-1)
    plt.tight_layout()


@gif.frame
def gif_output(x_true, x_recon, epoch=None, max_epochs=None, train_loss=-1,test_loss=-1, outpath="./output/testVAE.gif"):
    #trained with list-like code
    n_samples=5
    plt.figure(figsize=(10, 4.5))

    for i in range(n_samples):

        # plot original image
        ax = plt.subplot(2, 5, i + 1)
        plt.imshow(x_true[i])
        # plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(2, n_samples, i + 1 + n_samples)
        decImg=x_recon[i].reshape(x_true[i].shape)
        plt.imshow(decImg)
        # plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        if i==0:
            ax.text(x=0,y=35,s="Epoch {0}/{1}. Train Loss {2:.2f}. Test Loss {3:.2f}.".format(epoch,max_epochs,train_loss,test_loss))

def plot_generative_output(x_true, n_samples=100, output="./output/testVAE.png"):
    n_cols=5
    n_rows=int(n_samples/n_cols)
    fig,ax = plt.subplots(figsize=(n_cols,n_rows),nrows=n_rows, ncols=n_cols)
    ax_idx=0
    print(x_true.shape)
    for i in range(n_samples):
    # for i in range(n_rows):
    #     for j in range(n_cols): 
        if ax_idx%n_cols==0:
            ax_idx+=1
        current_ax=plt.subplot(n_rows, n_cols , i+1)
        plt.imshow(x_true[i].reshape((28, 28)))
        plt.gray()
        current_ax.get_xaxis().set_visible(False)
        current_ax .get_yaxis().set_visible(False)
    fig = plt.gcf()
    # fig.tight_layout()
    fig.savefig(output, bbox_inches='tight')
#         ax = plt.subplot(, n_samples, i + 1)
#         plt.imshow(x_true[i].reshape((28, 28)))
#         plt.gray()
#         ax.get_xaxis().set_visible(False)
#         ax.get_yaxis().set_visible(False)
#     fig = plt.gcf()
#   #  plt.show()
#     fig.savefig(output)
# plot_generative_output(1)