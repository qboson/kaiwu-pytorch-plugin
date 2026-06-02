# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#多进程相关
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import concurrent.futures 
import multiprocessing
from multiprocessing import shared_memory

import numpy as np
import itertools
import hydra.utils
import lightning as L
import torch
from transformers import BertForMaskedLM
from transformers.models.bert.configuration_bert import BertConfig
from bionemo.moco.interpolants import MDLM
from bionemo.moco.distributions.time import UniformTimeDistribution
from genmol.utils.utils_moco import AntitheticUniformTimeDistribution
from bionemo.moco.schedules.noise.continuous_noise_transforms import LogLinearExpNoiseTransform
from bionemo.moco.distributions.prior import DiscreteMaskedPrior

from genmol.utils.ema import ExponentialMovingAverage
from genmol.utils.utils_data import get_tokenizer
from genmol.utils.utils_save import clean_checkpoint, fast_forward_info

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

import kaiwu as kw


# =====================================================================
# 【极其重要】：为了让 ProcessPoolExecutor 正确序列化并传递任务，
# Worker 函数必须定义在全局作用域（类 EBM 的外部）。
# =====================================================================
def _sa_multiprocess_worker(shm_name, sub_quadratic_shape, sub_quadratic_dtype, np_ising_bias_chunk):
    """
    单个 worker 进程执行的模拟退火采样任务。

    参数:
        shm_name (str): 共享内存名称，存储 hidden-hidden 二次系数矩阵的副本。
        sub_quadratic_shape (tuple): 矩阵形状 (n_hid, n_hid)。
        sub_quadratic_dtype (np.dtype): 矩阵 dtype。
        np_ising_bias_chunk (np.ndarray): 当前 chunk 对应的 Ising 偏置，形状 (chunk_size, n_hid)。
    返回:
        chunk_sols (list of np.ndarray): 每个样本的低能态隐藏层二值解。
    """
    # 在子进程中重新导入所需模块
    import numpy as np
    from multiprocessing import shared_memory
    from kaiwu.classical import FastSimulatedAnnealingOptimizer

    # 通过共享内存名称打开只读矩阵，避免重复序列化
    existing_shm = shared_memory.SharedMemory(name=shm_name)
    np_sub_quadratic = np.ndarray(sub_quadratic_shape, dtype=sub_quadratic_dtype, buffer=existing_shm.buf)

    # 每个进程独立实例化求解器，避免共享导致死锁
    local_sampler = FastSimulatedAnnealingOptimizer(alpha=0.9, size_limit=10, process_num=1)

    n_hid = np_sub_quadratic.shape[0]
    chunk_sols = []

    # 遍历当前 chunk 的偏置，逐个求解
    for i in range(np_ising_bias_chunk.shape[0]):
        bias = np_ising_bias_chunk[i]
        # 构建 (n_hid+1) × (n_hid+1) 的 Ising 矩阵
        mat = np.zeros((n_hid + 1, n_hid + 1), dtype=np_sub_quadratic.dtype)
        mat[:-1, :-1] = np_sub_quadratic
        mat[:-1, -1] = bias
        mat[-1, :-1] = bias

        sol = local_sampler.solve(mat)
        if sol.shape[0] == 0:
            chunk_sols.append(np.empty((0, n_hid), dtype=np_sub_quadratic.dtype))
        else:
            # 将 Ising 解 { -1, 1 } 映射为 {0, 1}
            chunk_sols.append((sol[:, :-1] * sol[:, [-1]] + 1.0) / 2.0)

    # 释放本进程对共享内存的引用（共享内存段本身由主进程负责最终销毁）
    existing_shm.close()
    return chunk_sols


class GenMol(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        # set up tokenizer
        self.tokenizer = get_tokenizer()
        self.mask_index = self.tokenizer.mask_token_id
        self.bos_index = self.tokenizer.bos_token_id
        self.eos_index = self.tokenizer.eos_token_id
        # set up backbone   
        self.backbone = BertForMaskedLM(BertConfig.from_dict(dict(self.config.model)))
        # set up mdlm
        if self.config.training.antithetic_sampling:
            time_distribution = AntitheticUniformTimeDistribution(sampling_eps = self.config.training.sampling_eps)
        else:
            time_distribution = UniformTimeDistribution()
        prior = DiscreteMaskedPrior(num_classes = self.tokenizer.vocab_size, mask_dim = self.mask_index)
        noise_schedule = LogLinearExpNoiseTransform()
        self.mdlm = MDLM(time_distribution=time_distribution,
                          prior_distribution=prior,
                          noise_schedule = noise_schedule)
        # set up ema
        if self.config.training.ema > 0:
            self.ema = ExponentialMovingAverage(self.backbone.parameters(), decay=self.config.training.ema)
        else:
            self.ema = None
        

    def on_load_checkpoint(self, checkpoint):
        if self.ema:
            self.ema.load_state_dict(checkpoint['ema'])
        self.fast_forward_epochs, self.fast_forward_batches = fast_forward_info(checkpoint)
        
    def on_save_checkpoint(self, checkpoint):
        if self.ema:
            checkpoint['ema'] = self.ema.state_dict()
        clean_checkpoint(checkpoint, self.trainer.accumulate_grad_batches)
        if 'sampler' not in checkpoint.keys():
            checkpoint['sampler'] = {}
        if hasattr(self.trainer.train_dataloader.sampler, 'state_dict'):
            sampler_state_dict = self.trainer.train_dataloader.sampler.state_dict()
            checkpoint['sampler']['random_state'] = sampler_state_dict.get('random_state', None)
        else:
            checkpoint['sampler']['random_state'] = None

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.backbone.parameters(),
            lr=self.config.optim.lr,
            betas=(self.config.optim.beta1, self.config.optim.beta2),
            eps=self.config.optim.eps,
            weight_decay=self.config.optim.weight_decay)

        scheduler = hydra.utils.instantiate(
            {'_target_': 'transformers.get_constant_schedule_with_warmup',
             'num_warmup_steps': 2500},
             optimizer=optimizer)
        scheduler_dict = {
            'scheduler': scheduler,
            'interval': 'step',
            'name': 'lr'}
        return [optimizer], [scheduler_dict]

    def on_train_start(self):
        self.backbone.train()
        if self.ema:
            self.ema.move_shadow_params_to_device(self.device)
        
    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        if self.ema:
            self.ema.update(itertools.chain(self.backbone.parameters()))
        
    def forward(self, x, attention_mask=None):
        with torch.amp.autocast('cuda', dtype=torch.float32):
            return self.backbone(x, attention_mask)['logits']
    
    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        # sample time
        t = self.mdlm.sample_time(input_ids.shape[0])
        # forward process to add mask tokens
        xt = self.mdlm.forward_process(input_ids, t)
        # forward model pass
        with torch.amp.autocast('cuda', dtype=torch.float32):
            logits = self.backbone(xt, attention_mask)["logits"]
        # compute loss
        if self.config.training.global_mean_loss:
            loss = self.mdlm.loss(logits, input_ids, xt, t, mask=attention_mask, global_mean=True)
        else:
            loss = self.mdlm.loss(logits, input_ids, xt, t, mask=attention_mask).mean()
        self.log(name='train_loss',
                 value=loss.item(),
                 on_step=True,
                 on_epoch=False,
                 prog_bar=True,
                 sync_dist=True)
        return loss

class BertEBM(nn.Module):

    def __init__(
        self,
        bert_model,
        hidden_size,
        vocab_size
    ):
        super().__init__()


        # energy head
        self.energy_head_type = "bm"
        # self.energy_head_type = "mlp"
        if self.energy_head_type == "mlp":
            self.energy_head = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.SiLU(),
                nn.Linear(hidden_size, 1)
            )
        elif self.energy_head_type == "bm":
            from kaiwu.classical import FastSimulatedAnnealingOptimizer
            from kaiwu.torch_plugin import BoltzmannMachine
            kw.common.CheckpointManager.save_dir = "./tmp"
            self.energy_head = BoltzmannMachine(hidden_size + 128).cuda()
            self.samplerKW = FastSimulatedAnnealingOptimizer(
                initial_temperature=100,
                alpha=0.9,
                cutoff_temperature=1e-2,
                iterations_per_t=10,
                # size_limit=2,
                size_limit=10,
                # thread_num=10
            )

        self.process_pool = None
        self.num_workers = 50
            # 必须使用 'spawn' 上下文，避免 fork 时 PyTorch CUDA 状态导致死锁
        ctx = multiprocessing.get_context('spawn')
        self.process_pool = concurrent.futures.ProcessPoolExecutor(
                max_workers=self.num_workers,
                mp_context=ctx
            )


        # =========================================================
        # backbone
        # =========================================================
        self.encoder = bert_model.bert.encoder
        self.embeddings = bert_model.bert.embeddings

        self.token_embed = (
            self.embeddings.word_embeddings
        )

        # =========================================================
        # xt + x0 fusion
        # =========================================================
        self.vocab_proj = nn.Linear(
            2 * hidden_size,
            hidden_size
        )

        # =========================================================
        # sigma conditioning
        # =========================================================
        self.sigma_map = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )

        # =========================================================
        # attentive pooling
        # =========================================================
        self.pool_proj = nn.Linear(
            hidden_size,
            1
        )

        # =========================================================
        # energy head
        # =========================================================
        # self.energy_head = nn.Sequential(
        #     nn.Linear(hidden_size, hidden_size),
        #     nn.ReLU(),
        #     nn.Linear(hidden_size, 1)
        # )

        # =========================================================
        # projection head
        #
        # contrastive representation space
        # =========================================================
        self.projection_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 256)
        )

    # =============================================================
    # encoder
    # =============================================================
    def encode(
        self,
        xt,
        x0,
        attention_mask=None,
        sigma=None
    ):

        # ---------------------------------------------------------
        # token embedding
        # ---------------------------------------------------------
        xt_emb = self.token_embed(xt)
        x0_emb = self.token_embed(x0)

        # ---------------------------------------------------------
        # concatenate
        # ---------------------------------------------------------
        x = torch.cat(
            [xt_emb, x0_emb],
            dim=-1
        )

        x = self.vocab_proj(x)

        # ---------------------------------------------------------
        # position embedding
        # ---------------------------------------------------------
        position_ids = torch.arange(
            x.size(1),
            device=x.device
        ).unsqueeze(0)

        pos_emb = self.embeddings.position_embeddings(
            position_ids
        )

        x = x + pos_emb

        # ---------------------------------------------------------
        # sigma embedding
        # ---------------------------------------------------------
        sigma_emb = self.sigma_map(
            sigma.unsqueeze(-1)
        )

        x = x + sigma_emb[:, None, :]

        # ---------------------------------------------------------
        # attention mask
        # ---------------------------------------------------------
        if attention_mask is None:
            attention_mask = torch.ones_like(xt)

        extended_mask = attention_mask[:, None, None, :]
        extended_mask = extended_mask.float()
        extended_mask = (
            1.0 - extended_mask
        ) * -10000.0

        # ---------------------------------------------------------
        # transformer encoder
        # ---------------------------------------------------------
        encoder_outputs = self.encoder(
            x,
            attention_mask=extended_mask,
            return_dict=True
        )

        hidden = encoder_outputs.last_hidden_state

        # ---------------------------------------------------------
        # attentive pooling
        # ---------------------------------------------------------
        attn = torch.softmax(
            self.pool_proj(hidden),
            dim=1
        )

        pooled = (
            attn * hidden
        ).sum(dim=1)

        return pooled

    # =============================================================
    # forward
    # =============================================================
    def forward(
        self,
        xt,
        x0,
        attention_mask=None,
        sigma=None
    ):

        pooled = self.encode(
            xt,
            x0,
            attention_mask,
            sigma
        )


        if self.energy_head_type == "mlp":
            energy = self.energy_head(pooled)
        elif self.energy_head_type == "bm":
            # energy = self.energy_head(mean_pool)
            # hidden_out = self.energy_head.condition_sample(
            #     self.samplerKW, mean_pool
            # )
            # print('hidden_out.shape',hidden_out.shape)
            # B = mean_pool.size(0)
            # K = hidden_out.size(0) // B

            # energy = self.energy_head(hidden_out)  # [B*K]

            # energy = energy.view(B, K)   # [B, 50]
            # energy = energy.mean(dim=1)  # [B]
            energy = self._parallel_bm_energy(pooled)
            
        # ---------------------------------------------------------
        # projection
        # ---------------------------------------------------------
        z = self.projection_head(
            pooled
        )

        z = F.normalize(
            z,
            dim=-1
        )

        return {
            "energy": energy,
            "pooled": pooled,
            "z": z
        }
    
        ### BM相关
    def shutdown_pool(self):
        """显式关闭多进程池，释放资源。"""
        if self.process_pool is not None:
            self.process_pool.shutdown(wait=True)
            self.process_pool = None

    def __del__(self):
        """析构时尝试安全关闭进程池。"""
        try:
            self.shutdown_pool()
        except Exception:
            pass


    def _parallel_bm_energy(self, x):
        """
        使用多进程并行求解 BM 隐藏层低能态，并计算平均能量。
        所有数据组装与 GPU 计算仍为批量模式，CPU 部分使用共享内存避免重复序列化。
        """
        B = x.size(0)                         # batch size
        n_vis = x.size(1)                     # visible units
        n_hid = self.energy_head.num_nodes - n_vis   # hidden units
        device = x.device
        dtype = x.dtype

        # =====================================================================
        # 1. 纯 GPU 阶段：准备 Ising 模型参数
        # =====================================================================
        with torch.no_grad():
            q_coef = self.energy_head.quadratic_coef   # (N, N)
            l_bias = self.energy_head.linear_bias                    # (N,)

            # 提取 hidden-hidden 二次项、hidden-visible 交互项及偏置
            sub_quadratic = q_coef[n_vis:, n_vis:]                  # (n_hid, n_hid)
            sub_column_sums = torch.sum(sub_quadratic, dim=0)       # (n_hid,)
            sub_quadratic_vh = q_coef[n_vis:, :n_vis]              # (n_hid, n_vis)

            # 针对每个批次样本计算等效 Ising 偏置
            sub_linear = torch.matmul(x, sub_quadratic_vh.t()) + l_bias[n_vis:]   # (B, n_hid)
            ising_bias = sub_linear / 4.0 + sub_column_sums / 4.0

            # 转移到 CPU 并保存为 numpy 数组
            np_sub_quadratic = (sub_quadratic / 8.0).cpu().numpy().astype(np.float32)
            np_ising_bias = ising_bias.cpu().numpy().astype(np.float32)

        # =====================================================================
        # 2. 将 shared quadratic matrix 放入共享内存（消除重复序列化）
        # =====================================================================
        shm = shared_memory.SharedMemory(create=True, size=np_sub_quadratic.nbytes)
        # 在共享内存中创建 numpy 数组视图并复制数据
        shared_array = np.ndarray(np_sub_quadratic.shape, dtype=np_sub_quadratic.dtype, buffer=shm.buf)
        shared_array[:] = np_sub_quadratic[:]

        try:
            # 准备任务划分
            num_workers = min(B, self.num_workers)
            chunks = np.array_split(np.arange(B), num_workers)
            # 过滤掉空 chunk（不应该出现，但作为保护）
            chunks = [c for c in chunks if len(c) > 0]

            # 统一使用者以共享内存路径：提交每个 chunk 对应的 ising_bias 切片
            futures = [
                self.process_pool.submit(
                    _sa_multiprocess_worker,
                    shm.name,                        # 共享内存名称
                    np_sub_quadratic.shape,          # 矩阵形状
                    np_sub_quadratic.dtype,          # 数据类型
                    np_ising_bias[c]                 # 只传递该 chunk 的偏置（切片）
                )
                for c in chunks
            ]

            # 收集结果
            nested_sols = [f.result() for f in futures]
            # 展平为每个样本的解列表（顺序与原始 batch 一致）
            hids_np = [sol for sublist in nested_sols for sol in sublist]
        finally:
            # 无论是否异常，都要释放共享内存
            shm.close()
            shm.unlink()

        # =====================================================================
        # 3. 结果合并与 GPU 批量能量计算
        # =====================================================================
        valid_states = []
        split_sizes = []

        for i, hid_np in enumerate(hids_np):
            if hid_np.shape[0] == 0:
                split_sizes.append(0)
                continue

            # 将 numpy 解转回 torch，并复制可见单元拼成完整状态
            hid_tensor = torch.tensor(hid_np, dtype=dtype, device=device)
            vis_tensor = x[i:i+1].expand(hid_tensor.size(0), -1)
            full_state = torch.cat([vis_tensor, hid_tensor], dim=-1)

            valid_states.append(full_state)
            split_sizes.append(full_state.size(0))

        if sum(split_sizes) == 0:
            return torch.zeros((B, 1), dtype=dtype, device=device)

        giant_state = torch.cat(valid_states, dim=0)

        # 权重衰减项（保持与原代码一致）
        w_weight_decay = 0.02 * torch.sum(self.energy_head.quadratic_coef ** 2)
        b_weight_decay = 0.05 * torch.sum(self.energy_head.linear_bias ** 2)

        giant_energies = self.energy_head(giant_state) + w_weight_decay + b_weight_decay

        # 按原始样本拆分并计算每个样本的平均能量
        energy_splits = torch.split(giant_energies, split_sizes)
        final_energies = []
        for e in energy_splits:
            if e.numel() == 0:
                final_energies.append(torch.zeros((1, 1), dtype=dtype, device=device))
            else:
                final_energies.append(e.mean().view(1, 1))

        return torch.cat(final_energies, dim=0)

      

def _sample_categorical(categorical_probs, num_samples=1):
    """
    完全参考你提供的原版实现
    使用 Gumbel-Max 采样类别
    保证 shape / 逻辑 / 数值 完全一致
    """
    assert categorical_probs.ndim == 3
    B, L, V = categorical_probs.shape

    # 重复 num_samples 次（batch 维度扩张）
    # 原版：[B, L, V] → [B*K, L, V]
    categorical_probs = categorical_probs.repeat(num_samples, 1, 1)

    # Gumbel 噪声
    gumbel_noise = -torch.log(-torch.log(torch.rand_like(categorical_probs) + 1e-10) + 1e-10)

    # Gumbel-Max 采样
    samples = (categorical_probs + gumbel_noise).argmax(dim=-1)

    return samples

class GenMolEBM(GenMol):
    def __init__(self, config):
        super().__init__(config)

        hidden_size = self.config.model.hidden_size

        # === EBM ===
        self.ebm = BertEBM(
            bert_model=copy.deepcopy(self.backbone),
            hidden_size=hidden_size,
            vocab_size=self.tokenizer.vocab_size
        )

        # === 冻结 diffusion backbone ===
        # for p in self.backbone.parameters():
        #     p.requires_grad = False

        # self.backbone.eval()
        self.backbone.ebm = self.ebm
        # === EBM EMA ===
        if self.config.training.ema > 0:
            self.ebm_ema = ExponentialMovingAverage(
                self.ebm.parameters(),
                decay=self.config.training.ema
            )

        # if self.config.training.ema > 0:
        #     self.ema = ExponentialMovingAverage(
        #         self.ebm.parameters(),
        #         decay=self.config.training.ema
        #     )
        else:
            self.ebm_ema = None

    def ebm_forward(self, xt, x0, attention_mask=None,sigma=None):
        return self.ebm(xt, x0, attention_mask,sigma)

    # def training_step(self, batch, batch_idx):
        # input_ids = batch['input_ids']
        # attention_mask = batch['attention_mask']

        # # === sample time ===
        # t = self.mdlm.sample_time(input_ids.shape[0]).to(self.device)

        # # === diffusion forward ===
        # xt = self.mdlm.forward_process(input_ids, t)

        # # === frozen diffusion model ===
        # with torch.amp.autocast('cuda', dtype=torch.float32):
        #     logits = self.backbone(xt, attention_mask)["logits"]
        # logit_temperature = 1.0
        # probs = torch.softmax(log_p_x0 / logit_temperature, dim=-1)
        # preds = torch.distributions.Categorical(probs=probs).sample()
        # # === sample negative x0 ===
        # # x0_neg = torch.multinomial(
        # #     probs.view(-1, probs.size(-1)), 1
        # # ).view_as(input_ids)

        # # onely sample mask 
        # # mask_positions = (xt == self.mask_index)

        # # x0_neg = input_ids.clone()
        # # B,L,V = probs.shape
        # # sampled = torch.multinomial(probs.view(-1, V), 1).view(B, L)

        # # x0_neg[mask_positions] = sampled[mask_positions]

        # # x0_neg = _sample_categorical(logits.exp(), num_samples=1)

        # energy_pos = self.ebm_forward(xt, input_ids, attention_mask,t)
        # energy_neg = self.ebm_forward(xt, preds, attention_mask,t)


        # loss = (
        #     F.softplus(energy_pos) +   # = log(1 + exp(E_pos))
        #     F.softplus(-energy_neg)
        # ).mean()

        # # loss = F.softplus( energy_pos - energy_neg ).mean()
        # margin = 1.0
        # # loss = F.softplus(energy_pos - energy_neg + margin).mean()
        # # loss = -F.logsigmoid(energy_neg - energy_pos + margin).mean()
        # # reg = 1e-3 * (energy_pos.pow(2).mean() + energy_neg.pow(2).mean())
        # # loss = loss + reg
        
        # # energy_norm = (energy_pos**2 + energy_neg**2).mean() * 0.01  # 权重可调
        # self.log(name='train_loss',
        #          value=loss.item(),
        #          on_step=True,
        #          on_epoch=False,
        #          prog_bar=True,
        #          sync_dist=True)
        # self.log(name='energy_pos',
        #          value=energy_pos.mean().item(),
        #          on_step=True,
        #          on_epoch=False,
        #          prog_bar=True,
        #          sync_dist=True)
        # self.log(name='energy_neg',
        #          value=energy_neg.mean().item(),
        #          on_step=True,
        #          on_epoch=False,
        #          prog_bar=True,
        #          sync_dist=True)
        # # self.log(name='reg',
        # #          value=reg.item(),
        # #          on_step=True,
        # #          on_epoch=False,
        # #          prog_bar=True,
        # #          sync_dist=True)
        # return loss

    # def training_step(self, batch, batch_idx):

    #     target = batch['input_ids']
    #     attention_mask = batch['attention_mask']

    #     # === sample time ===
    #     t = self.mdlm.sample_time(input_ids.shape[0]).to(self.device)

    #     # === diffusion forward ===
    #     xt = self.mdlm.forward_process(input_ids, t)

    #     # === frozen diffusion model ===
    #     with torch.amp.autocast('cuda', dtype=torch.float32):
    #         logits = self.backbone(xt, attention_mask)["logits"]
    #     # ========================
    #     # 1. 从扩散模型采样负样本 x0_neg（保持原有逻辑）
    #     # ========================
    #     logprobs = self.mdlm._subs_parameterization(logits, xt)
    #     B, L, V = logprobs.shape

    #     # 采样负样本（与原版 _sample_categorical 一致）
    #     gumbel_noise = -torch.log(-torch.log(torch.rand_like(logprobs) + 1e-10) + 1e-10)
    #     x0_neg = (logprobs + gumbel_noise).argmax(dim=-1)
    #     x0_pos = target  # 正样本 = target

    #     # ========================
    #     # 2. 计算能量（关键：调用你的 EBM）
    #     # 这里调用你已经写好的 ebm_forward
    #     # ========================
    #     # 你需要确保 self 是 GenMolEBM，且有 ebm_forward
    #     sigma = self.mdlm.noise_schedule.calculate_sigma(time, target.device)  # 复用扩散的 sigma
    #     energy_pos = self.ebm_forward(xt, x0_pos, mask, sigma)  # [B, 1]
    #     energy_neg = self.ebm_forward(xt, x0_neg, mask, sigma)  # [B, 1]

    #     # ========================
    #     # 3. EBM NCE 损失（与你提供的 EBM 类完全一致）
    #     # ========================
    #     loss_pos = torch.log(torch.sigmoid(-energy_pos) + 1e-8)
    #     loss_neg = torch.log(torch.sigmoid(energy_neg) + 1e-8)
    #     loss = -(loss_pos + loss_neg).mean()  # [B, 1]


    #     self.log(name='energy_pos',
    #              value=energy_pos.mean().item(),
    #              on_step=True,
    #              on_epoch=False,
    #              prog_bar=True,
    #              sync_dist=True)
    #     self.log(name='energy_neg',
    #              value=energy_neg.mean().item(),
    #              on_step=True,
    #              on_epoch=False,
    #              prog_bar=True,
    #              sync_dist=True)
    #     self.log(name='loss',
    #              value=loss.mean().item(),
    #              on_step=True,
    #              on_epoch=False,
    #              prog_bar=True,
    #              sync_dist=True)

    #     return loss

    # def training_step(self, batch, batch_idx):

    #     # =========================================================
    #     # 0. batch
    #     # =========================================================
    #     target = batch["input_ids"]                 # [B, L]
    #     attention_mask = batch["attention_mask"]   # [B, L]

    #     B, L = target.shape

    #     # =========================================================
    #     # 1. sample diffusion time
    #     # =========================================================
    #     t = self.mdlm.sample_time(B).to(self.device)

    #     # =========================================================
    #     # 2. forward diffusion
    #     # xt ~ q(xt | x0)
    #     # =========================================================
    #     xt = self.mdlm.forward_process(target, t)

    #     # =========================================================
    #     # 3. frozen diffusion model
    #     # p_theta(x0 | xt)
    #     # =========================================================
    #     with torch.no_grad():
    #         with torch.amp.autocast("cuda", dtype=torch.float32):

    #             logits = self.backbone(
    #                 xt,
    #                 attention_mask=attention_mask
    #             )["logits"]                          # [B, L, V]

    #             logprobs = self.mdlm._subs_parameterization(
    #                 logits,
    #                 xt
    #             )                                   # [B, L, V]
    #     logit_temperature = 1.0
    #     probs = torch.softmax(logprobs / logit_temperature, dim=-1)
    #     preds = torch.distributions.Categorical(probs=probs).sample()
    #     # =========================================================
    #     # 4. construct negative samples
    #     #
    #     # IMPORTANT:
    #     # only replace masked positions
    #     # # =========================================================
    #     # gumbel_noise = -torch.log(
    #     #     -torch.log(torch.rand_like(logprobs) + 1e-10) + 1e-10
    #     # )

    #     # sampled_tokens = (logprobs + gumbel_noise).argmax(dim=-1)

    #     x0_neg = preds
    #     x0_pos = target

    #     # =========================================================
    #     # Hamming distance
    #     # =========================================================

    #     token_diff = (x0_pos != x0_neg).float()

    #     # ignore padding
    #     token_diff = token_diff * attention_mask.float()

    #     hamming_distance = (
    #         token_diff.sum(dim=1)
    #         / attention_mask.sum(dim=1).clamp(min=1)
    #     )

    #     mean_hamming = hamming_distance.mean()

    #     # =========================================================
    #     # 5. diffusion sigma
    #     # =========================================================
    #     sigma = self.mdlm.noise_schedule.calculate_sigma(
    #         t,
    #         target.device
    #     )

    #     dsigma = self.mdlm.noise_schedule.d_dt_sigma(
    #         t,
    #         target.device
    #     )

    #     # =========================================================
    #     # 6. EBM forward
    #     #
    #     # energy shape:
    #     # [B, 1]
    #     # =========================================================
    #     energy_pos = self.ebm_forward(
    #         xt=xt,
    #         x0=x0_pos,
    #         attention_mask=attention_mask,
    #         sigma=sigma,
    #     )

    #     energy_neg = self.ebm_forward(
    #         xt=xt,
    #         x0=x0_neg,
    #         attention_mask=attention_mask,
    #         sigma=sigma,
    #     )

    #     energy_pos = energy_pos.squeeze(-1)     # [B]
    #     energy_neg = energy_neg.squeeze(-1)     # [B]

    #     # =========================================================
    #     # 7. NCE / logistic EBM loss
    #     #
    #     # positive:
    #     #   low energy
    #     #
    #     # negative:
    #     #   high energy
    #     # =========================================================
    #     loss_pos = F.softplus(energy_pos)       # -log sigmoid(-E+)

    #     loss_neg = F.softplus(-energy_neg)      # -log sigmoid(E-)

    #     seq_loss = loss_pos + loss_neg          # [B]

    #     # =========================================================
    #     # 8. MDLM continuous-time weighting
    #     #
    #     # w(t) = dsigma / (exp(sigma)-1)
    #     # =========================================================
    #     time_weight = dsigma / torch.expm1(sigma)

    #     weighted_loss = seq_loss * time_weight

    #     # =========================================================
    #     # 9. final scalar loss
    #     # =========================================================
    #     loss = weighted_loss.mean()

    #     # =========================================================
    #     # 10. logging
    #     # =========================================================
    #     self.log(
    #         name="train/loss_weighted",
    #         value=loss,
    #         on_step=True,
    #         on_epoch=True,
    #         prog_bar=True,
    #         sync_dist=True,
    #     )

    #     self.log(
    #         name="train/loss_unweighted",
    #         value=seq_loss.mean().item(),
    #         on_step=True,
    #         on_epoch=True,
    #         prog_bar=True,
    #         sync_dist=True,
    #     )

    #     self.log(
    #         name="train/energy_pos",
    #         value=energy_pos.mean(),
    #         on_step=True,
    #         on_epoch=True,
    #         prog_bar=True,
    #         sync_dist=True,
    #     )

    #     self.log(
    #         name="train/energy_neg",
    #         value=energy_neg.mean(),
    #         on_step=True,
    #         on_epoch=True,
    #         prog_bar=True,
    #         sync_dist=True,
    #     )

    #     self.log(
    #         name="train/time_weight",
    #         value=time_weight.mean(),
    #         on_step=True,
    #         on_epoch=False,
    #         prog_bar=False,
    #         sync_dist=True,
    #     )
    #     self.log(
    #         name="train/mean_hamming",
    #         value=mean_hamming.item(),
    #         on_step=True,
    #         on_epoch=True,
    #         prog_bar=True,
    #         sync_dist=True,
    #     )


    #     return loss
    
     # =============================================================
    def training_step(
        self,
        batch,
        batch_idx
    ):

        # =========================================================
        # batch
        # =========================================================
        target = batch["input_ids"]
        attention_mask = batch["attention_mask"]

        B, L = target.shape

        # =========================================================
        # sample diffusion time
        # =========================================================
        t = self.mdlm.sample_time(B).to(
            self.device
        )

        # =========================================================
        # forward diffusion
        # =========================================================
        xt = self.mdlm.forward_process(
            target,
            t
        )

        # =========================================================
        # frozen diffusion model
        # =========================================================
        with torch.no_grad():

            with torch.amp.autocast(
                "cuda",
                dtype=torch.float32
            ):

                logits = self.backbone(
                    xt,
                    attention_mask=attention_mask
                )["logits"]

                logprobs = (
                    self.mdlm._subs_parameterization(
                        logits,
                        xt
                    )
                )


        # with torch.amp.autocast(
        #     "cuda",
        #     dtype=torch.float32
        # ):

        #     logits = self.backbone(
        #         xt,
        #         attention_mask=attention_mask
        #     )["logits"]

        #     logprobs = (
        #         self.mdlm._subs_parameterization(
        #             logits,
        #             xt
        #         )
        #     )


        if self.config.training.global_mean_loss:
            org_loss = self.mdlm.loss(logits, target, xt, t, mask=attention_mask, global_mean=True)
        else:
            org_loss = self.mdlm.loss(logits, target, xt, t, mask=attention_mask).mean()
        self.log(name='org_loss',
                 value=org_loss.item(),
                 on_step=True,
                 on_epoch=False,
                 prog_bar=True,
                 sync_dist=True)

        # =========================================================
        # sample negatives
        # =========================================================
        probs = torch.softmax(
            logprobs,
            dim=-1
        )

        x0_neg = torch.distributions.Categorical(
            probs=probs
        ).sample()

        x0_pos = target

        # =========================================================
        # only replace masked tokens
        # =========================================================
        mask_positions = (
            xt == self.mask_index
        )

        x0_neg = torch.where(
            mask_positions,
            x0_neg,
            target
        )

        # =========================================================
        # hamming distance
        # =========================================================
        token_diff = (
            x0_pos != x0_neg
        ).float()

        token_diff = (
            token_diff
            * attention_mask.float()
        )

        hamming_distance = (
            token_diff.sum(dim=1)
            / attention_mask.sum(dim=1).clamp(min=1)
        )

        mean_hamming = hamming_distance.mean()

        # =========================================================
        # optional:
        # hard negative mining
        # =========================================================
        hard_mask = (
            (hamming_distance > 0.05)
            &
            (hamming_distance < 0.50)
        )

        if hard_mask.sum() > 0:

            xt = xt[hard_mask]
            x0_pos = x0_pos[hard_mask]
            x0_neg = x0_neg[hard_mask]
            attention_mask = attention_mask[hard_mask]
            probs = probs[hard_mask]
            mask_positions = mask_positions[hard_mask]
            hamming_distance = hamming_distance[hard_mask]

            t = t[hard_mask]

            B = xt.size(0)

        # =========================================================
        # sigma
        # =========================================================
        sigma = (
            self.mdlm.noise_schedule.calculate_sigma(
                t,
                target.device
            )
        )

        dsigma = (
            self.mdlm.noise_schedule.d_dt_sigma(
                t,
                target.device
            )
        )

        # =========================================================
        # positive forward
        # =========================================================
        pos_out = self.ebm_forward(
            xt=xt,
            x0=x0_pos,
            attention_mask=attention_mask,
            sigma=sigma
        )

        # =========================================================
        # EBM FORWARD: 3 negative passes
        # =========================================================

        # 1. Hard negative: diffusion model prediction (x0_neg)
        neg_hard_out = self.ebm_forward(
            xt=xt,
            x0=x0_neg,
            attention_mask=attention_mask,
            sigma=sigma
        )

        # 2. Perturbed negative: ~30% adaptive perturbation on x0_neg
        # For discrete tokens, perturbation = resample a fraction of tokens
        perturb_ratio = 0.3
        perturb_mask = (
            torch.rand_like(x0_neg.float()) < perturb_ratio
        ) & mask_positions  # only perturb masked positions
        x0_neg_pert = torch.where(
            perturb_mask,
            torch.distributions.Categorical(probs=probs).sample(),
            x0_neg
        )
        neg_pert_out = self.ebm_forward(
            xt=xt,
            x0=x0_neg_pert,
            attention_mask=attention_mask,
            sigma=sigma
        )

        # 3. Random negative: fully random tokens at masked positions
        vocab_size = probs.size(-1)
        x0_neg_rand = torch.where(
            mask_positions,
            torch.randint(0, vocab_size, x0_neg.shape, device=x0_neg.device),
            x0_pos   # use x0_pos (already filtered) instead of unfiltered target
        )
        neg_rand_out = self.ebm_forward(
            xt=xt,
            x0=x0_neg_rand,
            attention_mask=attention_mask,
            sigma=sigma
        )

        # =========================================================
        # unpack energies
        # =========================================================
        energy_pos = pos_out["energy"]     # [B, 1]
        energy_neg_hard = neg_hard_out["energy"]   # [B, 1]
        energy_neg_pert = neg_pert_out["energy"]   # [B, 1]
        energy_neg_rand = neg_rand_out["energy"]   # [B, 1]

        # =========================================================
        # Multi-Negative InfoNCE Loss (aligned with ebm_v1.py)
        # =========================================================
        scores_pos = -energy_pos
        scores_neg_hard = -energy_neg_hard
        scores_neg_pert = -energy_neg_pert
        scores_neg_rand = -energy_neg_rand

        # Hard negative weight decay:
        # when x0_neg is very close to x0_pos, reduce its weight
        # to avoid positive/negative overlap in InfoNCE denominator
        hard_neg_tau = 0.25
        hard_neg_min_weight = 0.0
        # Use hamming_distance as the relative distance measure for discrete tokens
        hard_rel_dist = hamming_distance.detach().view(-1, 1)
        hard_neg_weight = (1.0 - torch.exp(-((hard_rel_dist / hard_neg_tau) ** 2))).clamp(
            min=hard_neg_min_weight,
            max=1.0,
        )
        hard_neg_log_weight = torch.where(
            hard_neg_weight > 0,
            torch.log(hard_neg_weight.clamp_min(1e-12)),
            torch.full_like(hard_neg_weight, -torch.inf),
        )
        # log-space addition => multiplicative weight in logsumexp
        scores_neg_hard = scores_neg_hard + hard_neg_log_weight

        # Concatenate: [B, 4] (1 positive + 3 negatives)
        all_scores = torch.cat([scores_pos, scores_neg_hard, scores_neg_pert, scores_neg_rand], dim=1)

        # InfoNCE: loss = logsumexp(all_scores) - score_positive
        ebm_loss = torch.logsumexp(all_scores, dim=1, keepdim=True) - scores_pos

        # =========================================================
        # continuous-time weighting
        # =========================================================
        time_weight = (
            dsigma
            / torch.expm1(sigma)
        ).unsqueeze(-1)   # [B, 1]

        ebm_loss = (
            ebm_loss
            * time_weight
        ).mean()

        # =========================================================
        # CONTRASTIVE LOSS (unchanged)
        # =========================================================
        # z_pos = pos_out["z"]

        # temperature = 0.1

        # Use all negatives for contrastive learning
        # z_neg_hard = neg_hard_out["z"]
        # z_neg_pert = neg_pert_out["z"]
        # z_neg_rand = neg_rand_out["z"]

        # Stack negatives: [B, 3, dim]
        # z_neg_all = torch.stack([z_neg_hard, z_neg_pert, z_neg_rand], dim=1)

        # InfoNCE contrastive: positive vs all negatives
        # logits shape: [B, 3]
        # logits = torch.einsum('bd,bkd->bk', z_pos, z_neg_all) / temperature

        # contrastive_labels = torch.zeros(B, dtype=torch.long, device=logits.device)

        # contrastive_loss = F.cross_entropy(
        #     logits,
        #     contrastive_labels
        # )

        # =========================================================
        # OPTIONAL:
        # embedding regularization
        # =========================================================
        # embedding_reg = (
        #     z_pos.pow(2).mean()
        #     +
        #     z_neg_hard.pow(2).mean()
        #     +
        #     z_neg_pert.pow(2).mean()
        #     +
        #     z_neg_rand.pow(2).mean()
        # )

        # =========================================================
        # FINAL LOSS
        # =========================================================
        # lambda_contrast = 0.1
        # lambda_reg = 1e-4

        loss = (
            ebm_loss
            # +
            # lambda_contrast
            # * contrastive_loss
            # +
            # lambda_reg
            # * embedding_reg
            # +org_loss
        )

        # =========================================================
        # logging
        # =========================================================
        self.log(
            "train/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True
        )

        # self.log(
        #     "train/contrastive_loss",
        #     contrastive_loss.item(),
        #     on_step=True,
        #     on_epoch=True,
        #     prog_bar=True,
        #     sync_dist=True
        # )

        self.log(
            "train/ebm_infonce",
            ebm_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True
        )

        self.log(
            "train/energy_pos",
            energy_pos.mean(),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True
        )

        self.log(
            "train/energy_neg_hard",
            energy_neg_hard.mean(),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True
        )

        self.log(
            "train/energy_neg_pert",
            energy_neg_pert.mean(),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True
        )

        self.log(
            "train/energy_neg_rand",
            energy_neg_rand.mean(),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True
        )

        self.log(
            "train/hard_neg_weight",
            hard_neg_weight.mean(),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True
        )


        return loss



    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.ebm.parameters(),
            lr=self.config.optim.lr,
            betas=(self.config.optim.beta1, self.config.optim.beta2),
            eps=self.config.optim.eps,
            weight_decay=self.config.optim.weight_decay
        )

        scheduler = hydra.utils.instantiate(
            {
                '_target_': 'transformers.get_constant_schedule_with_warmup',
                'num_warmup_steps': 2500
            },
            optimizer=optimizer
        )

        return [optimizer], [{
            'scheduler': scheduler,
            'interval': 'step'
        }]

    # def optimizer_step(self, *args, **kwargs):
    #     super().optimizer_step(*args, **kwargs)
    #     if self.ema:
    #         self.ema.update(self.ebm.parameters())

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        if self.ebm_ema:
            self.ebm_ema.update(self.ebm.parameters())
            
    # def load_pretrained_diffusion(self, ckpt_path):
    #     ckpt = torch.load(ckpt_path, map_location="cpu")
    #     state_dict = ckpt["state_dict"]

    #     new_state_dict = {
    #         k: v for k, v in state_dict.items()
    #         if k.startswith("backbone.")
    #     }

    #     self.load_state_dict(new_state_dict, strict=False)

    def load_pretrained_diffusion(self, ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu")

        state_dict = ckpt["state_dict"]
        print('state_dict-----------',state_dict)
        backbone_state = {}

        for k, v in state_dict.items():

            if k.startswith("backbone."):
                backbone_state[k] = v

            elif k.startswith("model.backbone."):
                new_k = k[len("model."):]
                backbone_state[new_k] = v

        missing, unexpected = super().load_state_dict(
            backbone_state,
            strict=False
        )

        print("missing:", missing)
        print("unexpected:", unexpected)


    # def load_state_dict(self, state_dict, strict=None):
    #     # 强行设置 strict=False，忽略 ebm 缺失的权重
    #     return super().load_state_dict(state_dict, strict=False)

    def on_save_checkpoint(self, checkpoint):
        super().on_save_checkpoint(checkpoint)

        if self.ebm_ema:
            checkpoint['ebm_ema'] = self.ebm_ema.state_dict()

    def on_load_checkpoint(self, checkpoint):
        super().on_load_checkpoint(checkpoint)
        if self.ebm_ema and 'ebm_ema' in checkpoint:
            self.ebm_ema.load_state_dict(checkpoint['ebm_ema'])

    def on_train_start(self):
        # 确保 ebm 在 train mode
        self.ebm.train()

        if self.ebm_ema:
            self.ebm_ema.move_shadow_params_to_device(self.device)

    def load_pretrained_diffusion_with_ema(
        self,
        ckpt_path
    ):
        ckpt = torch.load(
            ckpt_path,
            map_location="cpu",
            weights_only=False
        )

        state_dict = ckpt["state_dict"]

        # =====================================================
        # load backbone only
        # =====================================================
        backbone_state = {
            k[len("backbone."):]: v
            for k, v in state_dict.items()
            if k.startswith("backbone.")
        }

        missing, unexpected = self.backbone.load_state_dict(
            backbone_state,
            strict=True
        )

        print("✅ Backbone loaded")
        print("missing:", missing)
        print("unexpected:", unexpected)

        # =====================================================
        # load EMA
        # =====================================================
        if self.ema and "ema" in ckpt:

            print("🔥 Loading backbone EMA...")

            self.ema.load_state_dict(
                ckpt["ema"]
            )

            # EMA -> backbone
            self.ema.copy_to(
                itertools.chain(
                    self.backbone.parameters()
                )
            )

            print("✅ EMA weights copied to backbone")

        # =====================================================
        # sync EBM encoder
        # =====================================================
        self.ebm.encoder.load_state_dict(
            self.backbone.bert.encoder.state_dict()
        )

        self.ebm.embeddings.load_state_dict(
            self.backbone.bert.embeddings.state_dict()
        )

        self.ebm.token_embed = (
            self.ebm.embeddings.word_embeddings
        )

        print("✅ EBM encoder synced")
