import torch
import torch.nn as nn
import torch.nn.functional as F


class SmoothingDist:
    def pdf(self, zeta):
        """ Implements r(\zeta|z=0)"""
        raise NotImplementedError

    def cdf(self, zeta):
        """ Implements R(\zeta|z=0)"""
        raise NotImplementedError

    def sample(self, shape):
        """ Samples from r(\zeta|z=0)"""
        raise NotImplementedError

    def log_pdf(self, zeta):
        """ Computes log r(\zeta|z=0)"""
        raise NotImplementedError
    

class Exponential(SmoothingDist):
    """指数平滑分布类
    
    实现了指数平滑分布的PDF、CDF、采样和对数PDF计算。
    """
    
    def __init__(self, beta):
        self.beta = torch.tensor(beta, dtype=torch.float32)
        
    def pdf(self, zeta: torch.Tensor) -> torch.Tensor:
        return self.beta * torch.exp(-self.beta * zeta) / (1 - torch.exp(-self.beta))
    
    def cdf(self, zeta: torch.Tensor) -> torch.Tensor:
        return (1. - torch.exp(-self.beta * zeta)) / (1 - torch.exp(-self.beta))
    
    def sample(self, shape: tuple) -> torch.Tensor:
        rho = torch.rand(shape)
        zeta = -torch.log(1. - (1. - torch.exp(-self.beta)) * rho) / self.beta
        return zeta
    
    def log_pdf(self, zeta: torch.Tensor) -> torch.Tensor:
        return torch.log(self.beta) - self.beta * zeta - torch.log(1 - torch.exp(-self.beta))



class DistUtil:
    def reparameterize(self, is_training):
        raise NotImplementedError

    def kl_dist_from(self, dist_util_obj, aux):
        raise NotImplementedError

    def entropy(self):
        raise NotImplementedError

    def log_prob(self, samples):
        raise NotImplementedError



def sigmoid_cross_entropy_with_logits(logits, labels):
    """计算sigmoid交叉熵损失
    
    Args:
        logits (torch.Tensor): 对数几率
        labels (torch.Tensor): 标签
        
    Returns:
        torch.Tensor: sigmoid交叉熵损失
    """
    return logits - logits * labels + F.softplus(-logits)
    
class FactorialBernoulliUtil(DistUtil):
    """阶乘伯努利分布工具类
    
    用于处理二值随机变量的概率分布。
    """
    
    def __init__(self, param, smoothing_dist_beta=None):
        """初始化阶乘伯努利分布
        
        Args:
            param: 参数
            smoothing_dist_beta: 平滑分布的beta参数，可选
        """
        super().__init__()
        self.logit_mu = param

    def reparameterize(self, is_training: bool) -> torch.Tensor:
        """从伯努利分布中采样
        
        仅在测试时使用，因为伯努利分布的重参数化在训练时不可微分。
        
        Args:
            is_training (bool): 指示是否在构建训练计算图的标志
            
        Returns:
            torch.Tensor: 从伯努利分布采样的结果
            
        Raises:
            NotImplementedError: 当is_training为True时抛出，因为伯努利分布的重参数化在训练时不可微分
        """
        if is_training:
            raise NotImplementedError('伯努利分布的重参数化在训练时不可微分')
        else:
            device = self.logit_mu.device
            q = torch.sigmoid(self.logit_mu)
            rho = torch.rand_like(q, device=device)
            z = (rho < q).float()
            return z
        

    def entropy(self):
        """
        Computes the entropy of the bernoulli distribution using:
            x - x * z + log(1 + exp(-x)),  where x is logits, and z=sigmoid(x).
        Returns: 
            ent: entropy
        """
        mu = torch.sigmoid(self.logit_mu)
        ent = sigmoid_cross_entropy_with_logits(logits=self.logit_mu, labels=mu)
        return ent
    
    def log_prob_per_var(self, samples):
        """计算样本在分布下的对数概率
        
        Args:
            samples (torch.Tensor): 大小为(num_samples * num_vars)的矩阵
            
        Returns:
            torch.Tensor: 对数概率矩阵(num_samples * num_vars)
        """
        log_prob = - sigmoid_cross_entropy_with_logits(logits=self.logit_mu, labels=samples)
        return log_prob
        


class MixtureGeneric(FactorialBernoulliUtil):
    """混合分布类
    
    通过设置定义在z分量上的阶乘伯努利分布的对数几率，创建两个重叠分布的混合。
    这是一个通用类，可以与任何继承自SmoothingDist类的平滑分布一起工作。
    """
    
    num_param = 1
    
    def __init__(self, param, smoothing_dist_beta):
        """初始化混合分布
        
        Args:
            param: 参数
            smoothing_dist_beta: 平滑分布的beta参数
        """
        super().__init__(param)
        self.smoothing_dist = Exponential(smoothing_dist_beta)
        
    def reparameterize(self, is_training: bool) -> torch.Tensor:
        """使用祖先采样从两个重叠分布的混合中采样
        
        使用隐式梯度思想计算样本相对于logit_q的梯度。
        这个思想在DVAE# sec 3.4中提出。
        此函数不实现样本相对于beta或平滑变换的其他参数的梯度。
        
        Args:
            is_training (bool): 指示是否在构建训练计算图的标志
            
        Returns:
            torch.Tensor: 从重叠分布混合中采样的结果
        """
        q = torch.sigmoid(self.logit_mu)
        
        # 从伯努利分布采样
        z = super().reparameterize(is_training=False)
        shape = z.shape
        
        # 从平滑分布采样
        zeta = self.smoothing_dist.sample(shape)
        zeta = zeta.to(z.device)

        zeta = torch.where(z == 0., zeta, 1. - zeta)
        
        # 计算PDF和CDF
        pdf_0 = self.smoothing_dist.pdf(zeta)
        pdf_1 = self.smoothing_dist.pdf(1. - zeta)
        cdf_0 = self.smoothing_dist.cdf(zeta)
        cdf_1 = 1. - self.smoothing_dist.cdf(1. - zeta)
        
        # 计算梯度
        grad_q = (cdf_0 - cdf_1) / (q * pdf_1 + (1 - q) * pdf_0)
        grad_q = grad_q.detach()  # 相当于tf.stop_gradient
        grad_term = grad_q * q
        grad_term = grad_term - grad_term.detach()
        # 最终结果
        # 这相当于：zeta ≈ stop_gradient(zeta) + （ stop_gradient(∂zeta/∂q) * q - stop_gradient(∂zeta/∂q) * stop_gradient(q))
        # 目的是在反向传播时只让梯度流向 q，而不是 zeta 本身
        zeta = zeta.detach() + grad_term
        
        return zeta
    
    def log_prob_per_var(self, samples: torch.Tensor) -> torch.Tensor:
        """计算样本在重叠分布混合下的对数概率
        
        Args:
            samples (torch.Tensor): 大小为(num_samples * num_vars)的矩阵
            
        Returns:
            torch.Tensor: 对数概率矩阵(num_samples * num_vars)
        """
        q = torch.sigmoid(self.logit_mu)
        pdf_0 = self.smoothing_dist.pdf(samples)
        pdf_1 = self.smoothing_dist.pdf(1. - samples)
        log_prob = torch.log(q * pdf_1 + (1 - q) * pdf_0)
        return log_prob
    
    def log_prob(self, samples: torch.Tensor) -> torch.Tensor:
        """计算所有变量的对数概率之和
        
        Args:
            samples (torch.Tensor): 大小为(num_samples * num_vars)的矩阵
            
        Returns:
            torch.Tensor: 每个样本的对数概率向量
        """
        log_p = self.log_prob_per_var(samples)
        log_p = torch.sum(log_p, dim=1)
        return log_p
    
    def log_ratio(self, zeta: torch.Tensor) -> torch.Tensor:
        """计算KL梯度所需的log_ratio（在DVAE++中提出）
        
        Args:
            zeta (torch.Tensor): 近似后验样本
            
        Returns:
            torch.Tensor: log r(ζ|z=1) - log r(ζ|z=0)
        """
        log_pdf_0 = self.smoothing_dist.log_pdf(zeta)
        log_pdf_1 = self.smoothing_dist.log_pdf(1. - zeta)
        log_ratio = log_pdf_1 - log_pdf_0
        return log_ratio
