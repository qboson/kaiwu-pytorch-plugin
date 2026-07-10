# -*- coding: utf-8 -*-
"""
MNIST Data Loader
"""
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.datasets import MNIST, FashionMNIST, KMNIST

import logging
logger = logging.getLogger(__name__)

def loadMNIST(
    name="mnist", 
    data_path="./data", 
    batch_size=100, 
    num_evts_train=60000, 
    num_evts_test=10000,
    use_cuda=False
):
    """
    Load MNIST dataset with configurable parameters
    
    Args:
        name: dataset name ('mnist', 'fashion-mnist', 'kmnist')
        data_path: path to store/load MNIST data
        batch_size: batch size for data loaders
        num_evts_train: number of training samples to use
        num_evts_test: number of test samples to use  
        use_cuda: whether to use CUDA (affects DataLoader settings)
    
    Returns:
        train_loader, test_loader: PyTorch DataLoader objects
    """
    # Dataset mapping
    dataset_map = {
        "mnist": MNIST,
        "fashion-mnist": FashionMNIST,
        "kmnist": KMNIST,
    }

    # Select dataset class
    dataset_class = dataset_map[name]

    # Define transform to normalize data
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert PIL image to Tensor
        transforms.Lambda(lambda x: x.view(-1))  # Flatten the images
        # x.reshape(-1) or x.flatten() can also be used for flattening
    ])
    
    logger.info(f"Loading MNIST dataset from {data_path}")
    
    # Download and load training data
    train_data = dataset_class(
        root=data_path, 
        train=True, 
        download=True, 
        transform=transform
    )
    test_data = dataset_class(
        root=data_path, 
        train=False, 
        download=True, 
        transform=transform
    )
    
    # DataLoader configuration
    kwargs = {"num_workers": 0, "pin_memory": True} if use_cuda else {"num_workers": 0}

    # Limit dataset size if specified
    if num_evts_train < len(train_data):
        indices = torch.randperm(len(train_data))[:num_evts_train]
        train_data = Subset(train_data, indices)
        logger.info(f"Using subset of {num_evts_train} training samples")
    
    if num_evts_test < len(test_data):
        indices = torch.randperm(len(test_data))[:num_evts_test]  
        test_data = Subset(test_data, indices)
        logger.info(f"Using subset of {num_evts_test} test samples")
    
    # Create data loaders
    train_loader = DataLoader(
        dataset=train_data,  # 训练数据集
        batch_size=batch_size, # 每个批次的样本数
        shuffle=True, # 每个epoch打乱数据顺序，防止模型记忆顺序
        **kwargs # 解包上述配置参数
    )
    test_loader = DataLoader(
        dataset=test_data,  # 测试数据集
        batch_size=batch_size,  # 批次大小（通常与训练集相同）
        shuffle=False,  # 测试集不需要打乱，保证可重复性
        **kwargs,  # 解包配置参数
    )
    
    logger.info(f"Training samples: {len(train_loader.dataset)}")
    logger.info(f"Test samples: {len(test_loader.dataset)}")
    logger.info(f"Batch size: {batch_size}")
    
    return train_loader, test_loader

class MNISTDataLoader:
    """Alternative class-based MNIST data loader"""
    
    def __init__(self, batch_size=100, num_evts_train=60000, num_evts_test=10000, data_path="./data"):
        self.batch_size = batch_size
        self.num_evts_train = num_evts_train
        self.num_evts_test = num_evts_test
        self.data_path = data_path
        
        self.train_loader = None
        self.test_loader = None
        
    def load(self):
        """Load MNIST data"""
        self.train_loader, self.test_loader = loadMNIST(
            batch_size=self.batch_size,
            num_evts_train=self.num_evts_train, 
            num_evts_test=self.num_evts_test,
            data_path=self.data_path
        )
        return self.train_loader, self.test_loader
    
    def get_input_size(self):
        """Get input dimension (flattened MNIST image size)"""
        if self.train_loader is None:
            raise ValueError("Data not loaded yet. Call load() first.")
        
        # Get first batch to determine input size
        sample_batch, _ = next(iter(self.train_loader))
        if isinstance(sample_batch, list):
            # For multi-input models, return list of dimensions
            return [data.shape[1] for data in sample_batch]
        else:
            # For single input models, return single dimension
            return sample_batch.shape[1]

if __name__ == "__main__":
    # Test the data loader
    train_loader, test_loader = loadMNIST(batch_size=64, num_evts_train=1000, num_evts_test=200)
    
    # Check a batch
    for data, labels in train_loader:
        print(f"Batch data shape: {data.shape}")
        print(f"Batch labels shape: {labels.shape}")
        print(f"Data range: [{data.min():.3f}, {data.max():.3f}]")
        break