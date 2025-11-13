import torch
import torch.nn as nn
import torch.nn.functional as F

from kaiwu.torch_plugin import RestrictedBoltzmannMachine

# G(z)
class Generator(nn.Module):
    # initializers
    def __init__(self, input_size=32, n_class=10):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(self.fc1.out_features, 512)
        self.fc3 = nn.Linear(self.fc2.out_features, 1024)
        self.fc4 = nn.Linear(self.fc3.out_features, n_class)

    # forward method
    def forward(self, input):
        x = F.leaky_relu(self.fc1(input), 0.2)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.leaky_relu(self.fc3(x), 0.2)
        x = torch.tanh(self.fc4(x))

        return x

class Discriminator(nn.Module):
    # initializers
    def __init__(self, input_size=32, n_class=10):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(input_size, 1024)
        self.fc2 = nn.Linear(self.fc1.out_features, 512)
        self.fc3 = nn.Linear(self.fc2.out_features, 256)
        self.fc4 = nn.Linear(self.fc3.out_features, n_class)

    # forward method
    def get_feature(self, input):
        # input = zmap(input, alpha=1 1 1)
        x = F.leaky_relu(self.fc1(input), 0.2)
        x = F.dropout(x, 0.3)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.dropout(x, 0.3)
        x = F.leaky_relu(self.fc3(x), 0.2)
        return x

    def forward(self, input):
        x = self.get_feature(input)
        x = F.dropout(x, 0.3)
        x = torch.sigmoid(self.fc4(x))
        return x
    
class QGAN(nn.Module):
    def __init__(self, generator, discriminator, sampler, bm_visible, bm_hidden):
        super().__init__()

        self.G = generator
        self.D = discriminator
        self.sampler = sampler
        self.bm = RestrictedBoltzmannMachine(bm_visible, bm_hidden)

        self.bce_loss = nn.BCELoss()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def to(self, device=..., dtype=..., non_blocking=...):
        """Moves the model to the specified device.

        Args:
            device: Target device.
            dtype: Target data type.
            non_blocking: Whether the operation should be non-blocking.

        Returns:
            AbstractBoltzmannMachine: The model on the target device.
        """
        self.device = device
        return super().to(device)

    def discriminate(self, x):
        return self.D(x)

    def generate(self, z):
        return self.G(z)

    def sample_rbm_noise(self, only_visible=True):
        """从 RBM 中采样隐变量作为生成器输入"""
        start = self.bm.num_visible if only_visible else 0
        samples = self.bm.sample(self.sampler)[:, start:]  # 取 hidden 部分
        return (samples * 2 - 1).to(self.device)  # 映射到 [-1, 1]

    def d_loss(self, real_imgs):
        noise = self.sample_rbm_noise()
        fake_imgs = self.generate(noise)

        real_labels = torch.ones(real_imgs.size(0), device=self.device)
        fake_labels = torch.zeros(fake_imgs.size(0), device=self.device)

        real_pred = self.discriminate(real_imgs)
        fake_pred = self.discriminate(fake_imgs)

        loss_real = self.bce_loss(real_pred.squeeze(-1), real_labels)
        loss_fake = self.bce_loss(fake_pred.squeeze(-1), fake_labels)
        return loss_real + loss_fake

    def g_loss(self):
        noise = self.sample_rbm_noise()
        fake_imgs = self.generate(noise)
        labels = torch.ones(fake_imgs.size(0), device=self.device)
        pred = self.discriminate(fake_imgs)
        return self.bce_loss(pred.squeeze(-1), labels)

    def rbm_loss(self, real_features):
        # 提取真实图像的特征且不更新鉴别器梯度
        real_features_all = self.bm.get_hidden(real_features)
        sampled_states = self.sample_rbm_noise(False) # 采样玻尔兹曼分布更新BM机参数
        return self.bm.objective(real_features_all, sampled_states)
