# Copyright (C) 2022-2025 Beijing QBoson Quantum Technology Co., Ltd.
#
# SPDX-License-Identifier: Apache-2.0
from tqdm import tqdm
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Exponential
from torchvision import transforms
from torchmetrics.image.fid import FrechetInceptionDistance
from kaiwu.classical import SimulatedAnnealingOptimizer


def save_list_to_txt(filename, data):
    with open(filename, "w") as f:
        for value in data:
            f.write(f"{value:.6f}\n")


def get_real_images(dataloader, n_images=10000):
    images = []
    for batch_imgs, _ in dataloader:
        images.append(batch_imgs)
        if sum(img.shape[0] for img in images) >= n_images:
            break
    return torch.cat(images, dim=0)[:n_images]


def generate_images_original_vae(model, latent_dim, n_images=10000, batch_size=64):
    model.eval()
    imgs = []
    with torch.no_grad():
        for _ in tqdm(range(n_images // batch_size)):
            z = torch.randn(batch_size, latent_dim).to(device)
            img = model.decoder(z).cpu()
            imgs.append(img)
    return torch.cat(imgs, dim=0)[:n_images]


def generate_images_qvae(model, latent_dim, dist_beta, n_images=10000, batch_size=64):
    model.eval()
    imgs = []
    sampler = SimulatedAnnealingOptimizer(alpha=0.95)
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

            generated_x = generated_x + model.train_bias

            generated_x = torch.sigmoid(generated_x)

            imgs.append(generated_x)
    return torch.cat(imgs, dim=0)[:n_images]


def compute_fid_in_batches(fake_imgs, real_imgs, device, batch_size=64):
    """
    计算 FID 分数，适用于输入为 (N, 784) 的展平图像（如 MNIST）

    参数:
        fake_imgs: 生成图像，shape = (N, 784)
        real_imgs: 真实图像，shape = (M, 784)
        batch_size: 每个批次处理多少图像
        device: 使用 'cuda' 或 'cpu'
    返回:
        FID 分数
    """
    fid = FrechetInceptionDistance(feature=64).to(device)

    def preprocess(images):
        # 转换为图像格式 (B, 1, 28, 28)
        images = images.view(-1, 1, 28, 28)
        # 扩展为三通道
        images = images.repeat(1, 3, 1, 1)
        # 调整大小到 299x299
        resize = transforms.Resize((299, 299), antialias=True)
        return resize(images)

    # 如果不是 tensor，先转成 tensor
    if not isinstance(fake_imgs, torch.Tensor):
        fake_imgs = torch.tensor(fake_imgs, dtype=torch.uint8)
    if not isinstance(real_imgs, torch.Tensor):
        real_imgs = torch.tensor(real_imgs, dtype=torch.uint8)

    # 归一化到 [0, 255] 并转为 uint8（假定输入是 float 在 [0,1] 范围）
    fake_imgs = (fake_imgs * 255).clamp(0, 255).to(torch.uint8)
    real_imgs = (real_imgs * 255).clamp(0, 255).to(torch.uint8)

    # 转换为图像并更新 FID
    for i in range(0, len(real_imgs), batch_size):
        batch = real_imgs[i : i + batch_size]
        batch = preprocess(batch)
        fid.update(batch.to(device), real=True)

    for i in range(0, len(fake_imgs), batch_size):
        batch = fake_imgs[i : i + batch_size]
        batch = preprocess(batch)
        fid.update(batch.to(device), real=False)

    return fid.compute().item()
