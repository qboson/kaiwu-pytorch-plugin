# Digits 上的均衡传播

在 `sklearn.datasets.load_digits` 上训练 64-32-10 的均衡传播网络。像素除以 16。每个单元都是 ±1 自旋，自由相和微扰相的 Ising 矩阵都由 Kaiwu 模拟退火求解，Kaiwu 1.3.1 和 1.4.1 都可以运行。

测试集是 `random_state=0` 的分层 25%。剩下的图片里再分层留出 20% 作验证集，超参数只按验证准确率选择：

- 第一层权重在默认初始化基础上放大 4 倍（`--input-scale 4`），否则隐藏层从输入得到的场和输出层反馈的场一样大，隐藏层学不到输入的特征。
- 微扰系数 beta 取 2，SGD 学习率取 0.05，批大小 10，训练 6 个 epoch。
- 预测时只跑自由相，取输出层场强最大的类别。

冒烟（100 张训练、100 张验证、100 张测试）：

```powershell
python example/ep_digits/train_digits.py --smoke
```

冒烟测试准确率：0.4700。

完整划分（1077 张训练、270 张验证、450 张测试）：

```powershell
python example/ep_digits/train_digits.py
```

验证准确率：0.8407；测试准确率：0.8156。单进程每个 epoch 约 80 秒。

两次运行都把结果写到 `example/ep_digits/results/summary.json`，该目录已加入 `.gitignore`。
