import torch
import numpy as np
import matplotlib.pyplot as plt
import random
import time

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline

seed = 17171
# PyTorch
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)  # 为当前GPU设置
    torch.cuda.manual_seed_all(seed)  # 为所有GPU设置
# Python
random.seed(seed)
# NumPy
np.random.seed(seed)

from rbm_digits import RBMRunner
import kaiwu as kw


# 添加license相关信息
# kw.license.init(user_id="", sdk_code="")
def train_classifier():
    logistic = LogisticRegression(random_state=42)

    # 初始化RBM
    rbm = RBMRunner(
        n_components=128,
        learning_rate=0.1,
        batch_size=32,
        n_iter=2,
        verbose=True,
        plot_img=False,
        random_state=seed,
    )

    # 加载数据
    X_train, X_test, y_train, y_test = rbm.load_data(plot_img=True)

    classifier = Pipeline(steps=[("rbm", rbm), ("logistic", logistic)])

    ########## 训练模型 ##########
    logistic.C = 500.0
    logistic.max_iter = 1000

    # 训练 RBM-Logistic Pipeline
    start_time = time.time()
    classifier.fit(X_train, y_train)
    training_time = time.time() - start_time
    print(f"RBM Pipline training completed in {training_time:.2f} seconds")

    # 训练 Logistic regression
    logistic_classifier = LogisticRegression(C=500.0, max_iter=1000, random_state=42)

    start_time = time.time()
    logistic_classifier.fit(X_train, y_train)
    training_time = time.time() - start_time
    print(f"Logistic regression training completed in {training_time:.2f} seconds")

    ########## 评估模型 ##########
    pip_pred = classifier.predict(X_test)
    pip_acc = accuracy_score(y_test, pip_pred)
    print(
        "\nLogistic regression using RBM features:\n%s\n"
        % (classification_report(y_test, pip_pred))
    )
    print(f"Test Accuracy: {pip_acc:.4f}")

    log_pred = logistic_classifier.predict(X_test)
    log_acc = accuracy_score(y_test, log_pred)
    print(
        "\nLogistic regression using raw pixel features:\n%s\n"
        % (classification_report(y_test, log_pred))
    )
    print(f"Test Accuracy: {log_acc:.4f}")
    return rbm, y_test, log_pred
