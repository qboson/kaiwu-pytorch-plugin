import os
from tqdm import tqdm
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from utils.helpers import plot_training_curves

# 设置日志配置
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MLPClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self, 
        input_dim=None, 
        hidden_dims=[256, 128], 
        output_dim=10,
        weight_decay=1e-4, 
        lr_mlp=1e-3, 
        batch_size_mlp=64, 
        epochs_mlp=100,
        device=None, 
        save_path=None,
        random_state=42
    ):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.weight_decay = weight_decay
        self.lr = lr_mlp
        self.batch_size = batch_size_mlp
        self.epochs = epochs_mlp
        self.device = device
        self.random_state = random_state
        self.save_path = save_path

        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(self.random_state)

        self.model = None
        self.classes_ = None

    def _create_model(self):
        """创建 MLP PyTorch 模型"""
        layers = []
        prev_dim = self.input_dim
        for h in self.hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, self.output_dim))
        return nn.Sequential(*layers).to(self.device)

    def _train_mlp_epoch(self, model, data_loader, optimizer, criterion, device):
        """训练单个epoch
    
        Args:
            model: MLP模型
            data_loader: 训练数据加载器
            optimizer: 优化器
            criterion: 损失函数
             device: 计算设备
        
        Returns:
            tuple: (训练准确率, 平均训练损失)
        """
        train_loss, train_correct, train_total = 0, 0, 0
        model.train()

        for batch_x, batch_y in data_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()

            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += batch_y.size(0)
            train_correct += (predicted == batch_y).sum().item()

        # 计算训练集准确率
        train_acc = 100 * train_correct / train_total
        avg_train_loss = train_loss / len(data_loader)

        return train_acc, avg_train_loss

    def _eval_mlp_epoch(self, model, data_loader, criterion, device):
        """验证单个epoch
    
        Args:
            model: MLP模型
            data_loader: 验证数据加载器
            criterion: 损失函数
            device: 计算设备
        Returns:
            tuple: (验证准确率, 平均验证损失)
        """
        val_loss, val_correct, val_total = 0, 0, 0
        model.eval()

        with torch.no_grad():
            for batch_x, batch_y in data_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)

                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)  # 计算验证损失
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                val_total += batch_y.size(0)
                val_correct += (predicted == batch_y).sum().item()

            # 计算测试集准确率
            val_acc = 100 * val_correct / val_total
            avg_val_loss = val_loss / len(data_loader)

        return val_acc, avg_val_loss

    def fit(self, X, y, validation_split=0.2):
        """训练MLP模型"""
        if self.input_dim is None:
            self.input_dim = X.shape[1]
            logger.info(f"Auto-detected input_dim: {self.input_dim}")
        # 数据划分
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, 
            test_size=validation_split, 
            random_state=self.random_state,
            stratify=y
        )
        # 转为 Tensor
        X_train = torch.FloatTensor(X_train).to(self.device)
        y_train = torch.LongTensor(y_train).to(self.device)
        X_val = torch.FloatTensor(X_val).to(self.device)
        y_val = torch.LongTensor(y_val).to(self.device)

        self.classes_ = np.unique(y)

        # 创建数据加载器
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        # 创建模型
        self.model = self._create_model()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        criterion = nn.CrossEntropyLoss()

        # 记录训练历史
        train_loss_history = []
        val_loss_history = []
        train_acc_history = []
        val_acc_history = []

        best_val_acc = 0.0
        best_state = None
        epoch_pbar = tqdm(range(1, self.epochs + 1), desc="Training MLP")
        for epoch in epoch_pbar:
            # 训练
            train_acc, avg_train_loss = self._train_mlp_epoch(
                model=self.model,
                data_loader=train_loader,
                optimizer=optimizer,
                criterion=criterion,
                device=self.device
            )

            # 验证
            val_acc, avg_val_loss = self._eval_mlp_epoch(
                model=self.model,
                data_loader=val_loader,
                criterion=criterion,
                device=self.device
            )

            # 记录历史
            train_loss_history.append(avg_train_loss)
            val_loss_history.append(avg_val_loss)
            train_acc_history.append(train_acc)
            val_acc_history.append(val_acc)

            # 打印进度
            if epoch % 10 == 0:
                # print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Val Acc={val_acc:.2f}%")
                logger.info(
                    f"Epoch {epoch}/{epoch}: "
                    f"Train Loss={avg_train_loss:.4f}, Train Acc={train_acc:.2f}% | "
                    f"Val Loss={avg_val_loss:.4f}, Val Acc={val_acc:.2f}%"
                )

            # 选择最佳模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = self.model.state_dict()
                if self.save_path is not None:
                    model_save_path = os.path.join(self.save_path, "best_mlp_classifier.pth")
                    torch.save(best_state, model_save_path)

        # 加载最佳模型
        self.model.load_state_dict(best_state)
        logger.info(f"Best Validation Accuracy: {best_val_acc:.2f}%")

        # 绘制训练曲线
        if self.save_path is not None:
            curves_save_path = os.path.join(
                self.save_path, f"mlp_training_curves_epochs_{self.epochs}.png"
            )
        else:
            curves_save_path = None   # let plot_training_curves generate default
        plot_training_curves(
            train_loss_history=train_loss_history,
            val_loss_history=val_loss_history,
            train_acc_history=train_acc_history,
            val_acc_history=val_acc_history,
            save_path=curves_save_path,
            show=True,
        )
        return self

    def predict(self, X):
        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)
        with torch.no_grad():
            out = self.model(X_tensor)
            _, pred = out.max(1)
        return pred.cpu().numpy()

    def predict_proba(self, X):
        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)
        with torch.no_grad():
            out = self.model(X_tensor)
            probs = torch.softmax(out, dim=1)
        return probs.cpu().numpy()

    def score(self, X, y):
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)