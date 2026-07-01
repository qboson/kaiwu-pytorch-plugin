import torch.nn as nn

class Config:
    """统一配置类，支持所有模型类型及管道参数"""
    def __init__(self, model_type='MNISTQVAE', **kwargs):
        # 基础模型参数
        self.type = model_type
        self.activation_fct = nn.ReLU()

        if model_type == 'MNISTQVAE':
            self.encoder_hidden_nodes = [512]      # [512, 256] 可选
            self.decoder_hidden_nodes = [512]      # [256, 512] 可选
            self.num_latent_units = 256            # 总潜变量维数，RBM 可见/隐藏各一半
            self.dist_beta = 10.0
            self.kl_beta = 0.000001
            self.sampler_type = 'sa'               # 'sa' or 'cim'
            self.loss_type = 'bernoulli'           # 'bernoulli' or 'mse'
            self.weight_decay = 0.01
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        # ---- 训练管道参数（原 PiplineTransformer.__init__ 中的参数） ----
        self.name = kwargs.get('name', 'mnist')
        self.data_path = kwargs.get('data_path', './data')
        self.batch_size = kwargs.get('batch_size', 256)
        self.num_epochs = kwargs.get('num_epochs', 50)
        self.lr = kwargs.get('lr', 8e-4)
        self.bm_lr = kwargs.get('bm_lr', 8e-4)
        self.use_cuda = kwargs.get('use_cuda', False)
        self.feature_type = kwargs.get('feature_type', 'q')
        self.run_tsne = kwargs.get('run_tsne', False)
        self.compute_energy = kwargs.get('compute_energy', False)
        self.num_train_samples = kwargs.get('num_train_samples', 60000)
        self.num_test_samples = kwargs.get('num_test_samples', 10000)
        self.output_dir = kwargs.get('output_dir', None)
        self.classifier_kwargs = kwargs.get('classifier_kwargs', {})

        # 允许通过 kwargs 覆盖任何已有属性
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def __repr__(self):
        """返回可读的配置字符串"""
        attrs = {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
        return f"Config({attrs})"
