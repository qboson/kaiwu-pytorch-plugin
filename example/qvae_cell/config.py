from dataclasses import dataclass
from torch import nn

@dataclass
class Config:
    """Configuration for training the QVAE model on single-cell data."""

    # dataset
    name = "immune"
    data_path = "./immune_processed.h5ad"

    # output
    output_dir = "./outputs_immune_notebook"

    # device
    device = "cuda:0"
    seed = 42

    # training
    batch_size = 512
    epochs = 2
    early_stopping_patience = 10
    checkpoint_every = 10
    train_log_every = 10
    disable_tqdm = True
    val_percentage = 0.1

    # model
    hidden_dim = 512
    latent_dim = 256
    num_latent_units = 256
    num_visible = 128
    num_hidden = 128
    activation_fct = nn.ReLU()
    normalization_method = "layer"

    # loss
    dist_beta = 10.0
    kl_beta = 1e-5

    # optimizer
    lr = 1e-4
    rbm_lr = 1e-3
    bm_weight_decay = 0.0

    # representation
    loss_type = "mse"
    feature_type = "q" 
    # representation = "q"

    # weights
    load_weights = False

    # sampler
    sampler_type = "sa"

    # SA
    sa_initial_temperature = 1000
    sa_alpha = 0.5
    sa_cutoff_temperature = 0.001
    sa_iterations_per_t = 10
    sa_size_limit = 10
    sa_rand_seed = 512

    # CIM
    project_no = "26035324"
    task_name = "demo2_qvae"
    task_mode = "sample"
    sample_number = 16
    precision = 8
    truncated_precision = 10
    target_bits = 550
    tmp_dir = "./tmp"

    # 允许通过字典更新（便于从命令行解析结果填充）
    def update(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self