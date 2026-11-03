# Digits 上的均衡传播

在 `sklearn.datasets.load_digits` 上训练 64-32-10 的均衡传播网络，并用同宽度的反向传播 MLP 作对照。像素除以 16。每个单元使用 1 个比特，QUBO 由 Kaiwu 1.3.1 模拟退火求解。

1 比特单元的场强不超过 1/2 时会停在 0，所以第一层权重的初始尺度是 0.2，读出层接近 0.02，输出偏置为 -0.1。微扰系数 beta 取 2：beta 为 1 时，目标比特和全 0 状态能量打平，两相经常不分开。优化器是学习率 0.2 的 SGD，训练 6 个 epoch。

冒烟（100 张训练、100 张测试、6 个 epoch）：

```powershell
python example/ep_digits/train_digits.py --smoke
```

冒烟测试准确率：均衡传播 0.4800，反向传播 0.6300。

默认命令使用 `random_state=0` 的分层 75/25 划分，epoch 数和学习率与冒烟相同：

```powershell
python example/ep_digits/train_digits.py
```

完整划分（1347 张训练、450 张测试）的测试准确率：均衡传播 0.8222，反向传播 0.9667。

两次运行都把结果写到 `example/ep_digits/results/summary.json`，该目录已加入 `.gitignore`。
