import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from .trainer import Trainer
from .classifier import MLPClassifier
# from .feature_extractor import FeatureExtractor

def get_full_pipeline(
    name='mnist', 
    data_path='./data', 
    batch_size=256,
    num_epochs=50, 
    lr=8e-4, 
    bm_lr=8e-4,
    use_cuda=False,
    run_tsne=False,
    output_dir=None,
    feature_type='q',
    **classifier_kwargs
):
    """
    Returns a sklearn Pipeline that first trains a QVAE + extracts features,
    then trains an MLPClassifier on the extracted features.

    All outputs (QVAE reconstructions, t-SNE, MLP curves, model weights)
    are saved under `output_dir` (auto-generated if None).
    """
    if output_dir is None:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"./output/pipeline_{timestamp}"

    # Create the transformer
    qvae_transformer = PiplineTransformer(
        name=name,
        data_path=data_path,
        batch_size=batch_size,
        num_epochs=num_epochs,
        lr=lr,
        bm_lr=bm_lr,
        use_cuda=use_cuda,
        feature_type=feature_type,
        run_tsne=run_tsne,
        output_dir=output_dir
    )

    # Create MLPClassifier
    clf = MLPClassifier(save_path=output_dir, **classifier_kwargs)
    return Pipeline([('qvae', qvae_transformer), ('classifier', clf)])

class PiplineTransformer:
    """
    A transformer that trains a QVAE and extracts features using the trained model.
    """
    def __init__(
        self, 
        name='mnist', 
        data_path='./data', 
        batch_size=256,
        num_epochs=50, 
        lr=8e-4, 
        bm_lr=1e-4,
        use_cuda=False, 
        feature_type='q',
        run_tsne=False,
        output_dir=None
    ):
        self.name = name
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.lr = lr
        self.bm_lr = bm_lr
        self.use_cuda = use_cuda
        self.feature_type = feature_type
        self.run_tsne = run_tsne
        self.output_dir = output_dir
        
        self.trainer = None
        self.extractor = None
        self.model = None
        
    def fit(self, X, y=None):
        """
        训练 QVAE（忽略 y，因为 QVAE 是无监督的）
        X: 原始图像数据，用于训练 QVAE
        """
        X_train, X_val = train_test_split(X, test_size=0.2, random_state=42)
        y_train = np.zeros(len(X_train), dtype=int)
        y_val = np.zeros(len(X_val), dtype=int)

        # 创建 Trainer
        self.trainer = Trainer(
            name=self.name,
            data_path=self.data_path,
            model_type='QVAE',
            batch_size=self.batch_size,
            num_epochs=self.num_epochs,
            lr=self.lr,
            bm_lr=self.bm_lr,
            use_cuda=self.use_cuda,
            output_dir=self.output_dir,
           # Pass custom data
            custom_train_data=(X_train, y_train),
            custom_test_data=(X_val, y_val)
        )
        # 训练 QVAE
        self.model, _, _ = self.trainer.train(run_tsne=self.run_tsne)

        # # 创建特征提取器
        # self.extractor = FeatureExtractor(self.model, self.feature_type)
        return self
    
    # def transform(self, X):
    #     if self.extractor is None:
    #         raise ValueError("Model not trained. Call fit() first.")
    #     return self.extractor.transform(X)
    def transform(self, X):
        if self.trainer is None:
            raise ValueError("Model not trained. Call fit() first.")
        # Convert X to a DataLoader with dummy labels
        X_tensor = torch.FloatTensor(X)
        # Create dummy labels (zeros)
        dummy_labels = torch.zeros(len(X), dtype=torch.long)
        dataset = TensorDataset(X_tensor, dummy_labels)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        # Extract features
        features, _ = self.trainer.extract_features(loader, feature_type=self.feature_type)
        return features.numpy()