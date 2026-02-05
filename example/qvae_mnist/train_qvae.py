# Copyright (C) 2022-2026 Beijing QBoson Quantum Technology Co., Ltd.
#
# SPDX-License-Identifier: Apache-2.0
"""
QVAE训练模块
该模块实现了一个量子启发的变分自编码器，包含训练、可视化以及下游分类任务。
主要功能：
1. QVAE模型的训练和保存
2. 训练过程中的t-SNE可视化
3. 基于QVAE特征的MLP分类器训练
"""
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from kaiwu.classical import SimulatedAnnealingOptimizer
from kaiwu.torch_plugin import RestrictedBoltzmannMachine, BoltzmannMachine, QVAE
from utils import save_list_to_txt
from models import Encoder, Decoder, MLP
from visualizers import t_SNE, plot_training_curves, create_tsne_animation

def create_model(train_loader, input_dim, hidden_dim, latent_dim,
                 weight_decay, dist_beta, num_var1, num_var2,
                 lr, device):
    """创建QVAE模型
    
    Args:
        train_loader: 训练数据加载器
        input_dim: 输入维度
        hidden_dim: 隐藏层维度
        latent_dim: 潜在空间维度
        weight_decay: 权重衰减系数
        dist_beta: 分布beta参数
        num_var1: RBM可见层维度
        num_var2: RBM隐藏层维度
        lr: 学习率
        device: 计算设备
        
    Returns:
        tuple: (模型, 优化器)
    """
    # 计算数据均值
    mean_x = 0
    for x, _ in train_loader:
        mean_x += x.mean(dim=0)
    mean_x = mean_x / len(train_loader)
    mean_x = mean_x.cpu().numpy()

    # 重新创建编码器、解码器
    encoder = Encoder(input_dim, hidden_dim, latent_dim, weight_decay)
    decoder = Decoder(latent_dim, hidden_dim, input_dim, weight_decay)

    # 初始化bm和sampler
    bm = RestrictedBoltzmannMachine(
        num_visible=num_var1,
        num_hidden=num_var2,
    )
    # bm = BoltzmannMachine(num_nodes=num_var1 + num_var2)
    sampler = SimulatedAnnealingOptimizer(alpha=0.95)

    # 创建QVAE模型（参数与训练时完全一致）
    model = QVAE(
        encoder=encoder,
        decoder=decoder,
        bm=bm,
        sampler=sampler,
        dist_beta=dist_beta,
        mean_x=mean_x,
        num_vis=num_var1
    ).to(device)

    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    return model, optimizer

def _train_epoch(model, train_loader, optimizer, kl_beta, device):
    """训练单个epoch
    
    Args:
        model: QVAE模型
        train_loader: 训练数据加载器
        optimizer: 优化器
        kl_beta: KL散度权重
        device: 计算设备
        
    Returns:
        tuple: (平均损失, 平均负ELBO, 平均KL散度, 平均代价)
    """
    total_loss, total_elbo, total_kl, total_cost = 0.0, 0.0, 0.0, 0.0  # 初始化本轮累计损失等指标
    model.train()

    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)  # 将输入数据移动到指定设备（CPU/GPU）
        optimizer.zero_grad()  # 梯度清零

        # 前向传播，计算负ELBO、权重衰减、KL散度等
        # output, recon_x, neg_elbo, wd_loss, kl, cost, _, _ = model.neg_elbo(
        _, _, neg_elbo, wd_loss, kl, cost, _, _ = model.neg_elbo(
            data, kl_beta
        )
        loss = neg_elbo + wd_loss  # 总损失 = 负ELBO + 权重衰减
        loss.backward()  # 反向传播，计算梯度

        optimizer.step()  # 更新模型参数

        # 累加本批次的各项指标
        total_loss += loss.item()
        total_elbo += neg_elbo.item()
        total_kl += kl.item()
        total_cost += cost.item()

    # 计算平均指标
    n_batches = len(train_loader)
    return (
        total_loss / n_batches,
        total_elbo / n_batches,
        total_kl / n_batches,
        total_cost / n_batches
    )

def _train_mlp_epoch(model, data_loader, optimizer, criterion, device):
    """训练单个epoch
    
    Args:
        model: MLP模型
        data_loader: 训练数据加载器
        optimizer: 优化器
        criterion: 损失函数
        device: 计算设备
        
    Returns:
        tuple: (训练准确率, 平均训练损失)
    """
    train_loss, train_correct, train_total = 0, 0, 0
    model.train()

    for batch_x, batch_y in data_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        optimizer.zero_grad()

        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        train_total += batch_y.size(0)
        train_correct += (predicted == batch_y).sum().item()

    # 计算训练集准确率
    train_acc = 100 * train_correct / train_total
    avg_train_loss = train_loss / len(data_loader)

    return train_acc, avg_train_loss

def _eval_mlp_epoch(model, data_loader, criterion, device):
    """验证单个epoch
    
    Args:
        model: MLP模型
        data_loader: 验证数据加载器
        criterion: 损失函数
        device: 计算设备
    Returns:
        tuple: (验证准确率, 平均验证损失)
    """
    val_loss, val_correct, val_total = 0, 0, 0
    model.eval()

    with torch.no_grad():
        for batch_x, batch_y in data_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)  # 计算验证损失
            val_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            val_total += batch_y.size(0)
            val_correct += (predicted == batch_y).sum().item()

        # 计算测试集准确率
        val_acc = 100 * val_correct / val_total
        avg_val_loss = val_loss / len(data_loader)

    return val_acc, avg_val_loss

# ============ 训练Q-VAE模型 ============
def train_qvae_with_tsne(
    train_loader,  # 用于训练QVAE
    test_loader,  # 用于t-SNE可视化测试数据
    device,
    tsne_interval=1,  # 每隔多少个epoch做一次t-SNE
    animation_save_path="qva_training_evolution.gif",
    # VAE部分参数
    input_dim=784,  # 图片拉伸后的维度
    hidden_dim=512,  # fc1压缩后的维度
    latent_dim=256,  # 隐变量维度， num_visible + num_hidden
    num_var1=128,  # RBM可见层维度
    num_var2=128,  # RBM藏层维度
    dist_beta=10,  # 重叠分布的beta
    weight_decay=0.01,
    batch_size=256,
    epochs=20,
    lr=1e-3,
    kl_beta=0.000001,
    save_path="./models/",
):
    # 创建保存临时图像的文件夹
    os.makedirs("temp_tsne_frames", exist_ok=True)
    tsne_frames = []

    # 创建模型
    model, optimizer = create_model(
        train_loader,
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        weight_decay=weight_decay,
        dist_beta=dist_beta,
        num_var1=num_var1,
        num_var2=num_var2,
        lr=lr,
        device=device
    )

    # 训练循环
    loss_history = []
    elbo_history = []
    kl_history = []
    cost_history = []

    model.train()  # 设置模型为训练模式
    for epoch in tqdm(range(1, epochs + 1), desc="Training QVAE"):  # 遍历每个训练轮次
        # 训练一个epoch
        avg_loss, avg_elbo, avg_kl, avg_cost = _train_epoch(
            model, train_loader, optimizer, kl_beta, device
        )

        # 记录历史指标
        loss_history.append(avg_loss)
        elbo_history.append(avg_elbo)
        kl_history.append(avg_kl)
        cost_history.append(avg_cost)

        save_list_to_txt(os.path.join(save_path, "loss_history.txt"), loss_history)
        save_list_to_txt(os.path.join(save_path, "elbo_history.txt"), elbo_history)
        save_list_to_txt(os.path.join(save_path, "cost_history.txt"), cost_history)
        save_list_to_txt(os.path.join(save_path, "kl_history.txt"), kl_history)

        # # 保存当前轮次的模型参数
        # model_save_path = os.path.join(save_path, f'davepp_epoch{epoch}.pth')
        # torch.save(model.state_dict(), model_save_path)

        # 打印本轮训练结果
        print(f"Epoch {epoch}/{epochs}: "
              f"Loss: {avg_loss:.4f}, "
              f"elbo: {avg_elbo:.4f}, "
              f"KL: {avg_kl:.4f}, "
              f"Cost: {avg_cost:.4f}")

        # 每隔tsne_interval个epoch做一次t-SNE
        if epoch % tsne_interval == 0 or epoch == epochs:
            print(f"Generating t-SNE visualization for epoch {epoch}...")

            # 生成t-SNE图像
            frame_path = f"temp_tsne_frames/tsne_epoch_{epoch:03d}.png"
            # frame_path = os.path.join(save_path, f"temp_tsne_frames/tsne_epoch_{epoch:03d}.png")
            _ = t_SNE(
                test_loader,
                model,
                epochs=epoch,  # 传入当前epoch
                save_path=frame_path,
                show=False,  # 训练过程中不显示，只保存
            )
            tsne_frames.append(frame_path)

    # 生成动画
    print("Creating training evolution animation...")
    create_tsne_animation(tsne_frames, animation_save_path)

    # 清理临时文件
    for frame in tsne_frames:
        if os.path.exists(frame):
            os.remove(frame)
    os.rmdir("temp_tsne_frames")

    # 保存模型
    model_save_path = os.path.join(save_path, f"qvae_mnist.pth")
    torch.save(model.state_dict(), model_save_path)
    return model

def train_qvae(
    train_loader,  # 用于训练QVAE
    device,
    input_dim=784,  # 图片拉伸后的维度
    hidden_dim=512,  # fc1压缩后的维度
    latent_dim=256,  # 隐变量维度， num_visible + num_hidden
    num_var1=128,  # RBM可见层维度
    num_var2=128,  # RBM藏层维度
    dist_beta=10,  # 重叠分布的beta
    weight_decay=0.01,
    batch_size=256,
    epochs=20,
    lr=1e-3,
    kl_beta=0.000001,
    save_path="./models/",
):
    # 创建模型
    model, optimizer = create_model(
        train_loader,
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        weight_decay=weight_decay,
        dist_beta=dist_beta,
        num_var1=num_var1,
        num_var2=num_var2,
        lr=lr,
        device=device
    )

    # 训练循环
    loss_history = []
    elbo_history = []
    kl_history = []
    cost_history = []

    model.train()  # 设置模型为训练模式
    for epoch in tqdm(range(1, epochs + 1), desc="Training QVAE"):  # 遍历每个训练轮次
        # 训练一个epoch
        avg_loss, avg_elbo, avg_kl, avg_cost = _train_epoch(
            model, train_loader, optimizer, kl_beta, device
        )

        # 记录历史指标
        loss_history.append(avg_loss)
        elbo_history.append(avg_elbo)
        kl_history.append(avg_kl)
        cost_history.append(avg_cost)

        save_list_to_txt(os.path.join(save_path, "loss_history.txt"), loss_history)
        save_list_to_txt(os.path.join(save_path, "elbo_history.txt"), elbo_history)
        save_list_to_txt(os.path.join(save_path, "cost_history.txt"), cost_history)
        save_list_to_txt(os.path.join(save_path, "kl_history.txt"), kl_history)

        # # 保存当前轮次的模型参数
        # model_save_path = os.path.join(save_path, f'davepp_epoch{epoch}.pth')
        # torch.save(model.state_dict(), model_save_path)

        # 打印本轮训练结果
        print(f"Epoch {epoch}/{epochs}: "
              f"Loss: {avg_loss:.4f}, "
              f"elbo: {avg_elbo:.4f}, "
              f"KL: {avg_kl:.4f}, "
              f"Cost: {avg_cost:.4f}")

    # 保存模型
    model_save_path = os.path.join(save_path, f"qvae_mnist.pth")
    torch.save(model.state_dict(), model_save_path)
    return model


# ============ 从Q-VAE提取潜变量特征 ============
def extract_qvae_latent_features(qvae, dataloader):
    qvae.eval()
    all_features = []
    all_labels = []

    device = next(qvae.parameters()).device

    with torch.no_grad():
        for data, labels in dataloader:
            data = data.to(device)

            # Q-VAE前向传播，获取潜变量zeta
            # Q-VAE的forward返回: (recon_x, posterior, q, zeta)
            _, _, q, zeta = qvae(data)

            # all_features.append(zeta.cpu())
            all_features.append(zeta.cpu())
            all_labels.append(labels)

    features = torch.cat(all_features, dim=0)
    labels = torch.cat(all_labels, dim=0)
    return features, labels


# ============ 训练MLP分类器 ============
def train_mlp_classifier(
    features,
    labels,
    device,
    epochs=100,
    lr=1e-3,
    weight_decay=1e-4,
    batch_size=64,
    seed=42,
    smoke_test=False,
    show=True,
    save_path="./models/",
):
    # 数据分割
    X_train, X_val, y_train, y_val = train_test_split(
        features.numpy(), labels.numpy(), test_size=0.4, random_state=seed
    )

    X_train = torch.FloatTensor(X_train)
    y_train = torch.LongTensor(y_train)
    X_val = torch.FloatTensor(X_val)
    y_val = torch.LongTensor(y_val)

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # MLP模型
    mlp = MLP(input_dim=features.shape[1], output_dim=10).to(device)
    optimizer = torch.optim.Adam(mlp.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    # 记录训练历史
    train_loss_history = []
    val_loss_history = []
    train_acc_history = []
    val_acc_history = []

    # 训练循环
    best_acc = 0
    for epoch in tqdm(range(1, epochs + 1), desc="Training MLP"):
        # 训练阶段
        train_acc, avg_train_loss = _train_mlp_epoch(
            model=mlp,
            data_loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device
        )

        # 验证阶段
        val_acc, avg_val_loss = _eval_mlp_epoch(
            model=mlp,
            data_loader=val_loader,
            criterion=criterion,
            device=device
        )

        # 记录历史
        train_loss_history.append(avg_train_loss)
        val_loss_history.append(avg_val_loss)
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)

        # 打印训练和验证指标
        if epoch % 5 == 0:
            print(
                f"Epoch {epoch:3d}: "
                f"Train Loss: {avg_train_loss:.4f}, "
                f"Val Loss: {avg_val_loss:.4f}, "  # 新增验证损失
                f"Train Acc: {train_acc:.2f}%, "
                f"Val Acc: {val_acc:.2f}%"
            )

        if val_acc > best_acc:
            best_acc = val_acc
            model_save_path = os.path.join(save_path, "best_mlp_classifier.pth")
            torch.save(mlp.state_dict(), model_save_path)

    print(f"Best Validation Accuracy: {best_acc:.2f}%")

    # 绘制训练曲线
    curves_save_path = ""
    if not smoke_test:
        curves_save_path = os.path.join(
            save_path, f"mlp_training_curves_epochs_{epochs}.png"
        )
        plot_training_curves(
            train_loss_history=train_loss_history,
            val_loss_history=val_loss_history,
            train_acc_history=train_acc_history,
            val_acc_history=val_acc_history,
            save_path=curves_save_path,
            show=show,
        )
    return mlp, best_acc, curves_save_path
