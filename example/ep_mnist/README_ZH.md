# MNIST 均衡传播示例

## 安装与运行

使用 Python 3.10、Kaiwu 1.3.1、PyTorch 2.7.0 和 torchvision 0.22.0。
实际调用模拟退火前，在本地初始化 SDK 授权。

```sh
python -m pip install -e .
python -m pip install -r example/ep_mnist/requirements.txt
python example/ep_mnist/train_mnist.py --device cpu --threads 1 --epochs 3 --seed 0 --split-seed 0 --data-dir data --output-dir runs/ep-cpu
python example/ep_mnist/train_mnist.py --device cuda --threads 1 --epochs 3 --seed 0 --split-seed 0 --data-dir data --output-dir runs/ep-cuda
```

默认 CPU、训练一轮；`--device cuda:N` 可选择 GPU，需要安装对应的 CUDA PyTorch。
参数先在 CPU 初始化后迁移，关闭 TF32。复现实验时设置 OMP_NUM_THREADS=1、MKL_NUM_THREADS=1。
快速检查可增加 `--limit-train 20 --limit-eval 20`；调参使用 `--validation-only`，不加载测试集。

MNIST 像素除以 255；官方 60000 张训练图划出 5000 张验证图，留下 55000 张训练。
10000 张测试图仅在最后一轮完成后评估。`--split-seed` 与训练用的 `--seed` 分开控制。

## 模型与输出

默认 EPLinear 网络为 784-128-10，使用 ±1 自旋；beta=2、首层初始化缩放=2、
SGD 学习率=0.05、batch size=20、SA 每温度迭代=50。每批计算自由相与微扰相，
通过两相能量差求导更新参数。候选解经过参考自旋解码和能量排序，预测使用自由相的输出层场。
`--conv` 将首层改为 EPConv2d：4 通道、核大小 7、步长 3。

```sh
python example/ep_mnist/train_mnist.py --conv --input-scale 4 --sa-iterations 100 --output-dir runs/ep-conv
python example/ep_mnist/train_mnist.py --device cpu --data-dir data --evaluate runs/ep-cuda/checkpoint.pt --output-dir runs/ep-evaluation
```

每次使用新输出目录；每轮保存 `checkpoint.pt` 和 `summary.json`（卷积为 `summary_conv.json`）。
评估恢复模型配置和采样设置，设备由当前命令指定，输出 `evaluation.json`。
checkpoint 支持跨设备评估，不支持断点续训；此处不报告新的完整卷积训练结果。

## 参考结果与耗时

| 执行方式 | 测试准确率 | 总耗时（分钟） | SA（分钟） | Non-SA（分钟） |
| --- | ---: | ---: | ---: | ---: |
| CPU EP | 90.30% | 353.004 | 341.725 | 11.279 |
| CUDA EP + CPU SA | 89.06% | 453.329 | 386.931 | 66.398 |

上述为三轮单种子结果。CPU EP 最终训练/验证准确率为 90.06% / 89.54%。
CPU BP 同划分、种子、层宽、批大小、学习率及轮数的测试准确率为 96.20%，
但使用 ReLU、交叉熵及不同初始化。EP 能量差可能为负，不能与 BP 交叉熵直接比较。

![CPU EP/BP 训练和验证错误率](../../docs/_static/ep-mnist/training_curves.png)

[逐轮 CSV](../../docs/_static/ep-mnist/training_curves.csv) ·
[参考结果、环境与来源](reference_results/README.md) · [验证说明](../../tests/README.validation.md)

曲线展示分类错误率（1－准确率），不是逐轮测试准确率。表中耗时包含三轮训练、
训练/验证集评估及最终测试。每阶段记录总耗时、SA、非 SA 及调用次数，CUDA 阶段边界同步。
SA 仍在 CPU 执行；非 SA 包含矩阵构建、传输、梯度及 Python 开销，并非纯 GPU 时间。
完整 EP 每轮训练调用 SA 110000 次，评估 60000 次，最终测试 10000 次。
不同执行时段与入口不构成严格加速比，清理归档后未重跑完整训练。
