# MNIST 上的均衡传播

本示例支持完整数据集训练和 MNIST 小样本验证。快速验证时使用 `--limit-train 20 --limit-eval 20`，并用 `--output-dir` 指定独立输出目录，完整命令见下文。

在完整的原版 MNIST 上训练均衡传播网络。数据通过 `torchvision.datasets.MNIST` 读取，像素除以 255。每个单元都是 ±1 自旋，自由相和微扰相的 Ising 矩阵都由 Kaiwu 1.3.1 模拟退火求解。

测试集是官方的 10000 张测试图。从 60000 张训练图里随机留出 5000 张作验证集，剩下 55000 张用于训练。超参数在 2000 张训练图上按验证准确率选择。

## 全连接网络（默认）

784-128-10，第一层初始化放大 2 倍，beta 取 2，SGD 学习率 0.05，批大小 20。退火每个温度迭代 50 次：迭代 10 次时，单点翻转还能让能量继续下降，说明解没有收敛。

```powershell
python example/ep_mnist/train_mnist.py
```

首次运行会把 MNIST 下载到 `example/ep_mnist/data/`；已有数据时用 `--data-dir` 指定所在目录。以下结果使用种子 0，训练了 1 个 epoch。

验证准确率：0.8788；测试准确率：0.8872。

## 卷积网络

`--conv` 把第一层换成 `EPConv2d`：4 个通道，卷积核 7，步长 3，输出 4×8×8 = 256 个自旋，之后接 10 个输出自旋。自旋更多，退火每个温度迭代 100 次；第一层初始化放大 4 倍。

```powershell
python example/ep_mnist/train_mnist.py --conv --input-scale 4 --sa-iterations 100
```

以下结果使用种子 0，训练了 1 个 epoch。

验证准确率：0.6966；测试准确率：0.7155。

## 耗时

每个样本每个相位调用一次退火，所以每个 epoch 要做 110000 次求解，验证和测试另需 15000 次。单进程下全连接网络每次求解约 31 毫秒，一个 epoch 约 1 小时。上面两组结果是两个任务并行跑的，实测分别约 72 和 174 毫秒/次。正式运行前可以先用 `--limit-train 2000 --limit-eval 1000` 看一下本机速度。

使用 Python 3.10、Kaiwu 1.3.1、PyTorch 2.7.0 和 torchvision 0.22.0。
如果本地 SDK 要求授权，请在运行前初始化授权。

默认输出目录为 `example/ep_mnist/results/`。全连接写入 `summary.json`，
卷积写入 `summary_conv.json`。快速验证时用 `--output-dir` 指定单独的目录。

```powershell
python example/ep_mnist/train_mnist.py --limit-train 20 --limit-eval 20 --output-dir runs/ep-mnist-smoke
```
