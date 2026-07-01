import os
import random
from types import SimpleNamespace

import kaiwu as kw
import numpy as np
import torch
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from kaiwu.classical import SimulatedAnnealingOptimizer
from kaiwu.cim import CIMOptimizer
from kaiwu.preprocess import PrecisionReducer
from kaiwu.torch_plugin import BoltzmannMachine

from models import CellQVAE, QVAEDecoder, QVAEEncoder


def set_seed(seed):
    """固定 Python、NumPy 和 PyTorch 随机种子，提升实验可复现性。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class Trainer:
    """封装数据准备、模型创建、训练循环和表示提取。"""

    metric_keys = ("loss", "neg_elbo", "kl", "recon_loss", "bm_loss")

    def __init__(self, args, device):
        self.args = args
        self.device = device
        self.n_batches = 0

    def adata_to_array(self, adata):
        """把 AnnData.X 转为 float32 矩阵，并按损失函数准备输入范围。"""
        x = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
        x = np.asarray(x, dtype=np.float32)
        if self.args.loss_type == "mse":
            return x

        # Bernoulli 重构要求输入接近概率值；若原始数据超出 [0, 1]，先做 min-max 归一化。
        x_min = float(np.nanmin(x))
        x_max = float(np.nanmax(x))
        if x_min < 0.0 or x_max > 1.0:
            x = (x - x_min) / (x_max - x_min + 1e-8)
        return np.clip(x, 0.0, 1.0).astype(np.float32, copy=False)

    def batch_indices(self, adata, batch_key):
        """把 obs 中的 batch 分类列编码为整数索引，供 decoder 拼接 one-hot 使用。"""
        batch_categories = adata.obs[batch_key].astype("category")
        self.n_batches = len(batch_categories.cat.categories)
        return batch_categories.cat.codes.to_numpy()

    def make_loaders(self, x, batch_indices):
        """构造训练、验证和全量评估 DataLoader。"""
        # 训练/验证只划分细胞索引，保证表达矩阵和 batch 索引保持同步。
        train_idx, val_idx = train_test_split(
            np.arange(x.shape[0]),
            test_size=self.args.val_percentage,
            random_state=self.args.seed,
        )
        x_tensor = torch.tensor(x, dtype=torch.float32)
        batch_tensor = torch.tensor(batch_indices, dtype=torch.long)
        return (
            DataLoader(
                TensorDataset(x_tensor[train_idx], batch_tensor[train_idx]),
                batch_size=self.args.batch_size,
                shuffle=True,
            ),
            DataLoader(
                TensorDataset(x_tensor[val_idx], batch_tensor[val_idx]),
                batch_size=self.args.batch_size,
                shuffle=False,
            ),
            DataLoader(
                TensorDataset(x_tensor, batch_tensor),
                batch_size=self.args.batch_size,
                shuffle=False,
            ),
        )

    def compute_mean_x(self, train_loader):
        """计算训练集每个基因的均值，作为 QVAE 的 train_bias。"""
        total = None
        count = 0
        for x, _ in train_loader:
            batch_sum = x.sum(dim=0)
            total = batch_sum if total is None else total + batch_sum
            count += x.shape[0]
        return (total / count).cpu().numpy()

    def create_sampler(self):
        """创建 BM 负相采样器：本地模拟退火或远端 CIM。"""
        if self.args.sampler_type == "sa":
            return SimulatedAnnealingOptimizer(
                initial_temperature=self.args.sa_initial_temperature,
                alpha=self.args.sa_alpha,
                cutoff_temperature=self.args.sa_cutoff_temperature,
                iterations_per_t=self.args.sa_iterations_per_t,
                size_limit=self.args.sa_size_limit,
                rand_seed=self.args.sa_rand_seed,
            )
        if self.args.sampler_type != "cim":
            raise ValueError(f"Unsupported sampler_type: {self.args.sampler_type}")

        # CIM 任务会产生中间文件，统一放到 tmp_dir，避免污染当前目录。
        kw.common.CheckpointManager.save_dir = self.args.tmp_dir
        sampler = CIMOptimizer(
            task_name=self.args.task_name,
            project_no=self.args.project_no,
            wait=True,
            task_mode=self.args.task_mode,
            sample_number=self.args.sample_number,
        )
        return PrecisionReducer(
            sampler,
            precision=self.args.precision,
            truncated_precision=self.args.truncated_precision,
            target_bits=self.args.target_bits,
            only_feasible_solution=False,
        )

    def create_model(self, input_dim, train_loader):
        """根据数据维度和命令行参数创建 CellQVAE 及两个优化器。"""
        if self.args.latent_dim != self.args.num_visible + self.args.num_hidden:
            raise ValueError("latent_dim must equal num_visible + num_hidden")

        # latent_dim 同时作为 QVAE 后验维度和 BM 变量数。
        encoder = QVAEEncoder(
            input_dim,
            self.args.hidden_dim,
            self.args.latent_dim,
            self.args.normalization_method,
        )
        # decoder 输入为潜变量 zeta 加 batch one-hot；没有 batch 时 n_batches 为 0。
        decoder = QVAEDecoder(
            self.args.latent_dim + self.n_batches,
            self.args.hidden_dim,
            input_dim,
            self.args.normalization_method,
        )
        config = SimpleNamespace(
            num_latent_units=self.args.latent_dim,
            loss_type=self.args.loss_type,
            dist_beta=self.args.dist_beta,
            kl_beta=self.args.kl_beta,
            weight_decay=0.0,
        )
        model = CellQVAE(
            input_dimension=input_dim,
            activation_fct=torch.nn.ReLU(),
            config=config,
            encoder=encoder,
            decoder=decoder,
            bm=BoltzmannMachine(self.args.latent_dim, device=self.device),
            sampler=self.create_sampler(),
            n_batches=self.n_batches,
        ).to(self.device)
        mean_x = self.compute_mean_x(train_loader)
        model.set_dataset_mean(mean_x)
        model.set_train_bias(mean_x)
        # VAE 部分和 BM 部分分开优化，便于使用不同学习率和更新目标。
        vae_optimizer = torch.optim.Adam(
            list(model.encoder.parameters()) + list(model.decoder.parameters()),
            lr=self.args.lr,
        )
        bm_optimizer = torch.optim.Adam(model.bm.parameters(), lr=self.args.rbm_lr)
        return model, vae_optimizer, bm_optimizer

    def run_epoch(self, model, loader, vae_optimizer, bm_optimizer, train=True):
        """运行一个训练或验证 epoch，并返回平均指标。"""
        model.train(train)
        totals = {key: 0.0 for key in self.metric_keys}

        for x, batch_idx in loader:
            x = x.to(self.device)
            batch_idx = batch_idx.to(self.device)
            if train:
                vae_optimizer.zero_grad()

            with torch.set_grad_enabled(train):
                # 第一阶段：更新 encoder/decoder，使重构项和 KL 项组成的 ELBO 更优。
                output, posterior, q, _ = model(x, batch_idx)
                loss = model.loss(x, output, posterior)
                if train:
                    loss.backward()
                    vae_optimizer.step()

            bm_loss_value = 0.0
            if train:
                # 第二阶段：固定 encoder 输出 q，单独用 BM 目标更新玻尔兹曼机参数。
                bm_optimizer.zero_grad()
                bm_loss = model.bm_loss(q, self.args.bm_weight_decay)
                bm_loss.backward()
                bm_optimizer.step()
                bm_loss_value = bm_loss.item()

            totals["loss"] += loss.item()
            totals["neg_elbo"] += loss.item()
            totals["kl"] += model.last_kl_loss.item()
            totals["recon_loss"] += model.last_recon_loss.item()
            totals["bm_loss"] += bm_loss_value

        return {key: value / len(loader) for key, value in totals.items()}

    def train(self, model, train_loader, val_loader, vae_optimizer, bm_optimizer, ckpt_path):
        """完整训练流程：记录曲线、保存最佳权重、按 patience 早停。"""
        history = {
            f"{split}_{key}": []
            for split in ("train", "val")
            for key in self.metric_keys
        }
        best_val = float("inf")
        best_state = None
        patience = 0

        progress = tqdm(
            range(1, self.args.epochs + 1),
            desc="Training KPP QVAE",
            disable=getattr(self.args, "disable_tqdm", False),
        )
        log_every = max(1, int(getattr(self.args, "train_log_every", 1)))

        for epoch in progress:
            train_metrics = self.run_epoch(model, train_loader, vae_optimizer, bm_optimizer, True)
            val_metrics = self.run_epoch(model, val_loader, vae_optimizer, bm_optimizer, False)

            for key in self.metric_keys:
                # 训练曲线按 split_metric 命名，便于直接转成 DataFrame 保存。
                history[f"train_{key}"].append(train_metrics[key])
                history[f"val_{key}"].append(val_metrics[key])

            if epoch % log_every == 0 or epoch == self.args.epochs:
                print(
                    f"[Epoch {epoch}] "
                    f"train_loss={train_metrics['loss']:.4f} "
                    f"val_loss={val_metrics['loss']:.4f} "
                    f"train_recon={train_metrics['recon_loss']:.4f} "
                    f"val_recon={val_metrics['recon_loss']:.4f} "
                    f"train_kl={train_metrics['kl']:.4f} "
                    f"val_kl={val_metrics['kl']:.4f} "
                    f"bm_loss={train_metrics['bm_loss']:.4f}"
                )

            if val_metrics["loss"] < best_val:
                # 内存中保存一份最佳状态，同时写 checkpoint 到磁盘。
                best_val = val_metrics["loss"]
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
                torch.save(model.state_dict(), ckpt_path)
                patience = 0
            else:
                patience += 1

            if self.args.checkpoint_every > 0 and epoch % self.args.checkpoint_every == 0:
                # 周期性 checkpoint 用于中途诊断或恢复，不影响 best checkpoint。
                torch.save(
                    model.state_dict(),
                    os.path.join(os.path.dirname(ckpt_path), f"model_epoch{epoch}.pth"),
                )

            if self.args.early_stopping_patience > 0 and patience >= self.args.early_stopping_patience:
                print(f"Early stopping triggered at epoch {epoch}")
                break

        if best_state is not None:
            # 返回前恢复验证集最优参数，保证后续表示提取使用最佳模型。
            model.load_state_dict(best_state)
        return history

    def extract_representation(self, model, eval_loader):
        """从全量数据中提取 q 或 zeta 表示。"""
        model.eval()
        reps = []
        with torch.no_grad():
            for x, batch_idx in eval_loader:
                _, _, q, zeta = model(
                    x.to(self.device), batch_idx.to(self.device)
                )
                # q 是 encoder logits，zeta 是后验采样/松弛后的潜变量表示。
                rep = q if self.args.representation == "q" else zeta
                reps.append(rep.cpu().numpy())
        return np.concatenate(reps, axis=0)

    def compute_energy(self, model, eval_loader):
        """计算每个细胞在 BM 潜空间中的能量值。"""
        model.eval()
        energies = []
        with torch.no_grad():
            for x, _ in eval_loader:
                energies.extend(model.energy(x.to(self.device), self.args.loss_type).cpu().numpy().tolist())
        return energies
