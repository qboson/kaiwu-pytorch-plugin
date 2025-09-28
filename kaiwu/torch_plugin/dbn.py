# -*- coding: utf-8 -*-
"""deep belief network DBN模型
包含DBN的类  以及训练DBN+model/仅训练model的函数 训练DBN+model会保存训练过程中的似然值和预测准确率
"""
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.optim import SGD
from torch.utils.data import DataLoader, TensorDataset
from kaiwu.torch_plugin import RestrictedBoltzmannMachine
from kaiwu.classical import SimulatedAnnealingOptimizer

# =================== 无监督DBN通用模型 =====================
class UnsupervisedDBN(nn.Module):
    """
    无监督DBN（堆叠RBM）通用模型架构
    参数:
        hidden_layers_structure (list): 每层隐藏单元个数
        device (torch.device): 计算设备
    """
    def __init__(
        self, 
        hidden_layers_structure=[100, 100]
        ):
        super(UnsupervisedDBN, self).__init__()
        self.hidden_layers_structure = hidden_layers_structure

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.rbm_layers = None
        self.input_dim = None
        self._is_trained = False

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

        self._is_trained = False
        return self

    def forward(self, X):
        """前向传播 - 特征变换"""
        if self.rbm_layers is None:
            raise ValueError("Model not built yet. Call _create_rbm_layer first.")
        if not self._is_trained:
            raise ValueError("Model not trained yet. Call mark_as_trained() after training.")
        
        X_data = X.astype(np.float32)
        for rbm in self.rbm_layers:
            with torch.no_grad():
                hidden_output = rbm.get_hidden(torch.FloatTensor(X_data).to(self.device))
                X_data = hidden_output[:, rbm.num_visible:].cpu().numpy()  # 只取隐藏层部分
        return X_data

    def transform(self, X):
        """sklearn兼容的transform方法"""
        return self.forward(X)

    def reconstruct(self, X, layer_index=0):
        """从指定层重建输入"""
        if self.rbm_layers is None or len(self.rbm_layers) == 0:
            raise ValueError("No RBM layers found. Please fit the model first.")

        if layer_index >= len(self.rbm_layers):
            raise ValueError(f"Layer index {layer_index} out of range.")
        
        rbm = self.rbm_layers[layer_index]
        return self.reconstruct_with_rbm(rbm, X, self.device)

    def mark_as_trained(self):
        """标记模型为已训练状态"""
        self._is_trained = True
        return self

    def get_rbm_layer(self, index):
        """获取指定索引的RBM层"""
        if index < len(self.rbm_layers):
            return self.rbm_layers[index]
        return None

    @staticmethod
    def reconstruct_with_rbm(rbm, X, device=None):
        """
        使用单个RBM重建数据
        """
        if device is None:
            device = rbm.device
        
                # 转换为PyTorch张量
        X_tensor = torch.FloatTensor(X).to(device)
        
        with torch.no_grad():
            # 使用RBM的get_hidden获取隐藏层表示
            hidden_act = rbm.get_hidden(X_tensor)
            hidden_part = hidden_act[:, rbm.num_visible:]   # 只取隐藏层部分
            
            # 重建可见层（使用权重转置）
            visible_recon = torch.sigmoid(
                torch.matmul(hidden_part, rbm.quadratic_coef.t()) + 
                rbm.linear_bias[:rbm.num_visible]
            )
            
            # 计算重建误差
            recon_errors = torch.mean((X_tensor - visible_recon) ** 2, dim=1).cpu().numpy()
            
        return visible_recon.cpu().numpy(), recon_errors

    @property
    def num_layers(self):
        """返回RBM层数"""
        return len(self.rbm_layers)

    @property
    def output_dim(self):
        """返回输出维度"""
        if len(self.rbm_layers) > 0:
            return self.rbm_layers[-1].num_hidden
        return self.input_dim

# =================== 无监督DBN训练器 =====================
class DBNTrainer:
    """
    DBN训练器，包含训练模块
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
        dbn_ref=None
    ):
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

    def train(self, dbn, X):
        """
        预训练DBN模型
        Args:
            dbn: UnsupervisedDBN实例
            X: 训练数据，形状为 (n_samples, n_features)
        Returns:
            训练后的DBN模型
        """
        if not isinstance(dbn, UnsupervisedDBN):
            raise ValueError("dbn must be an instance of UnsupervisedDBN")

        # 保存DBN
        self.dbn_ref = dbn

        # 设置随机种子
        if self.random_state is not None:
            self._set_random_seed()

        input_data = X.astype(np.float32)
        
        # 创建RBM层
        if dbn.num_layers == 0:
            dbn._create_rbm_layer(X.shape[1])

        for idx in range(dbn.num_layers):  # 使用 num_layers 和 get_rbm_layer 属性
            rbm = dbn.get_rbm_layer(idx)
            if self.verbose:
                n_visible = rbm.num_visible
                n_hidden = rbm.num_hidden
                print(f"\n[DBN] Pre-training RBM layer {idx+1}/{dbn.num_layers}: "
                      f"{n_visible} -> {n_hidden}")

            # 训练当前RBM层
            input_data = self._train_rbm_layer(rbm, input_data, idx)
        
        # 标记模型为已训练
        dbn.mark_as_trained()
        return dbn

    def _train_rbm_layer(self, rbm, input_data, layer_idx):
        """训练单个RBM层"""
        optimizer = SGD(rbm.parameters(), lr=self.learning_rate_rbm)
        
        # 使用当前层的输入数据，而不是原始X
        X_torch = torch.FloatTensor(input_data).to(rbm.device)

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

    def _visualize_current_reconstruction(self, rbm, batch_data, batch_idx, epoch, layer_index=0):
        """可视化当前batch的重建效果"""

        batch_numpy = batch_data.cpu().numpy()

        # 使用静态重建方法
        recon_imgs, recon_errors = UnsupervisedDBN.reconstruct_with_rbm(rbm, batch_numpy)
    
        # 选择前几个样本显示
        n_show = min(8, batch_data.shape[0])
        original_imgs = batch_data[:n_show].cpu().numpy()
    
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
        """定期评估重建质量 - 使用静态方法"""
        n_eval = min(100, input_data.shape[0])
        eval_data = input_data[:n_eval]
        
        # 使用静态重建方法
        _, recon_errors = UnsupervisedDBN.reconstruct_with_rbm(
            rbm, eval_data
        )
        
        avg_recon_error = np.mean(recon_errors)
        print(f"[RBM] Layer {layer_idx+1}, Epoch {epoch+1}: "
              f"Reconstruction Error = {avg_recon_error:.6f}\n")


    def _gen_digits_image(self, X, size=8):
        """生成数字图像"""
        digits = X.reshape(20, size, size)
        image = np.hstack(digits)
        return image

    def _set_random_seed(self):
        """设置随机种子"""
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.random_state)