import torch.nn as nn

class Config:
    """统一配置类，支持所有模型类型"""
    def __init__(self, model_type='AE'):
        self.type = model_type
        self.num_latent_units = 32
        self.activation_fct = nn.ReLU()

        if model_type == 'AE':
            self.encoder_hidden_nodes = [512, 256, 128]
            self.decoder_hidden_nodes = [128, 256, 512]

        elif model_type == 'VAE':
            self.encoder_hidden_nodes = [512, 256]
            self.decoder_hidden_nodes = [256, 512]

        elif model_type == 'DiVAE':
            self.num_latent_hierarchy_levels = 2
            self.num_det_units = 128
            self.num_det_layers = 2
            self.decoder_hidden_nodes = [256, 512]
            self.weight_decay_factor = 1e-4

        elif model_type == 'HiVAE':
            self.num_latent_hierarchy_levels = 2
            self.num_det_units = 128
            self.num_det_layers = 2
            self.decoder_hidden_nodes = [256, 512]
            self.encoder_hidden_nodes = [256, 128]   # 占位，实际 HiVAE 不使用

        elif model_type == 'RBM_VAE':
            self.num_latent_hierarchy_levels = 2
            self.num_det_units = 128
            self.num_det_layers = 2
            self.decoder_hidden_nodes = [256, 512]
            self.weight_decay_factor = 1e-4
            self.beta_kl = 0.001

        elif model_type == 'QVAE':
            self.encoder_hidden_nodes = [512] #[512, 256]
            self.decoder_hidden_nodes = [512] #[256, 512]
            self.num_latent_units = 256 #128      # 总潜变量维数，RBM 可见/隐藏各一半
            self.dist_beta = 10.0
            self.kl_beta = 0.000001
            self.sampler_type = 'sa'         # 'sa' or 'cim'
            self.loss_type = 'bernoulli'     # 'bernoulli' or'mse'
            self.weight_decay = 0.01
        #     # QVAE 训练相关参数
        #     self.lr = 1e-4
        #     self.rbm_lr = 1e-3
        #     self.bm_weight_decay = 0.0
        #     self.val_fraction = 0.1
        #     self.early_stopping_patience = 10
        #     self.checkpoint_every = 10

        elif model_type == 'CellQVAE':
            self.encoder_hidden_nodes = [512]   # 实际会被覆盖，但保留占位
            self.decoder_hidden_nodes = [512]
            self.num_latent_units = 256
            self.dist_beta = 10.0
            self.kl_beta = 1e-5
            self.sampler_type = 'sa'
            self.loss_type = 'bernoulli'   # 默认
            self.weight_decay = 0.01
            # 新增 CellQVAE 参数
            self.hidden_dim = 512
            self.normalization_method = 'layer'
            self.bm_weight_decay = 0.0
        else:
            raise ValueError(f"Unsupported model type: {model_type}")