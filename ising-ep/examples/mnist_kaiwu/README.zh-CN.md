# MNIST/100 Ising EP 实验

[English](README.md) · [中文](README.zh-CN.md)

用 Kaiwu 的模拟退火器完成两个弛豫相，在 MNIST/100 上训练 784-64-40 的 Ising 均衡传播网络。算法本身见[项目 README](../../README.zh-CN.md)；本文件说明如何运行实验，以及归档运行产出了什么。

## 数据

MNIST/100 是 Ising EP 文献使用的缩小划分：**按原始 IDX 文件顺序，每个数字取前 100 张训练图像和前 10 张测试图像**。按位置选取而不是随机抽取，因此只凭源文件就能复现这一划分，不需要额外记录种子。像素缩放到 `[0, 1]`，按 C 顺序展平，没有数据增强。

把四个压缩包放到 `data/MNIST/raw/`：

```
train-images-idx3-ubyte.gz   train-labels-idx1-ubyte.gz
t10k-images-idx3-ubyte.gz    t10k-labels-idx1-ubyte.gz
```

也可以加上 `--download-data`，在第一次使用时下载。下载是显式选项，训练不会意外访问网络；不加该选项时，缺少压缩包会抛出 `FileNotFoundError`，而不是悄悄编造一个子集。

## 命令

以下命令假定已经从项目根目录执行 `pip install -e ".[dev]"`，从而可以导入 `ising_ep`。见[安装](../../README.zh-CN.md#安装)。在项目根目录下运行。

冒烟运行，几分钟内把整条路径走通：

```powershell
python -m examples.mnist_kaiwu.experiment `
  --output runs/smoke --seeds 0 --n-hidden 8 --epochs 1 `
  --candidate-count 2 --iterations-per-t 4
```

产生 `results/primary/` 的那次运行：

```powershell
python -m examples.mnist_kaiwu.experiment `
  --output runs/primary --seeds 0,1,2 `
  --n-hidden 64 --epochs 5 --batch-size 10 --lr 0.01 --beta 1.0
```

### 选项

| 选项 | 默认值 | 含义 |
|---|---|---|
| `--output` | 必填 | 接收产物的目录 |
| `--data-root` | `data` | 含有 `MNIST/raw` 的目录 |
| `--download-data` | 关闭 | 缺少压缩包时下载，而不是失败 |
| `--seeds` | `0,1,2` | 逗号分隔的实验种子 |
| `--n-hidden` | `64` | 单个隐层的宽度 |
| `--hidden-sizes` | 未设置 | 逗号分隔的各隐层宽度，如 `128,64`；设置后覆盖 `--n-hidden` |
| `--epochs` | `5` | 遍历训练集的次数 |
| `--batch-size` | `10` | 每次更新前共用同一组参数的样本数 |
| `--lr` | `0.01` | SGD 学习率，未单独指定时用于每一层 |
| `--lr-weights` | 未设置 | 权重学习率，每个连接一个（共 `len(hidden) + 1` 个） |
| `--lr-biases` | 未设置 | 偏置学习率，每个自旋层一个（共 `len(hidden) + 1` 个） |
| `--init-gains` | 未设置 | 每个连接的初始范围缩放，默认 `1` |
| `--weight-decay` | `0.0` | 只作用于权重矩阵的 L2 衰减 |
| `--beta` | `1.0` | 微扰系数，必须为正 |
| `--initial-temperature` | `1.25` | 退火起始温度 |
| `--alpha` | `0.95` | 几何冷却因子 |
| `--cutoff-temperature` | `0.001` | 退火停止温度 |
| `--iterations-per-t` | `10` | 每个温度的扫描次数 |
| `--candidate-count` | `8` | 每次求解返回的候选数 |

开销按 `2 × n_train × epochs + n_test` 次求解器调用增长，因此默认运行每个种子发出 10,100 次调用。降低 `--candidate-count` 和 `--iterations-per-t` 是缩短冒烟运行最快的办法。

## 产物

```
<output>/config.json              完整配置，含退火日程
<output>/summary.json             跨种子的 mean/std/min/max
<output>/seeds/<seed>/
    initial-model.npz             训练前的 weights.N、biases.N
    checkpoint.npz                最后一个 epoch 之后的同一组参数
    epochs/<e>/model.npz          每个 epoch 的参数快照
    epochs/<e>/dynamics.json      hidden_flip、output_flip、zero_update 等
    test/metrics.json             accuracy、tie_rate、confusion、per_class
    test/predictions.json         每张测试图像一条记录
    status.json                   该种子的摘要、学习证据、调用次数
```

JSON 以排序后的键写出，且不含无意义空白，因此同一配置的重复运行产生逐字节相同、可以干净比较的文件。

### 阅读 `dynamics.json`

| 字段 | 含义 |
|---|---|
| `hidden_flip` | 两个相之间发生变化的隐层自旋比例 |
| `output_flip` | 输出自旋的同一比例 |
| `zero_update` | 两个相完全一致的样本比例，这些样本贡献的梯度恰好为零 |
| `hidden_zero_update` | 仅隐层状态没有变化的样本比例 |

`zero_update` 接近 1.0 表示微扰太弱，或者退火预算太小：运行会正常结束，但什么也学不到。`status.json` 里的 `learning_evidence.passed` 专门检查这种失败：两个耦合都要移动，并且至少一个 epoch 要在不少于 5% 的样本上翻转隐层自旋。

## 归档结果

来自 [`results/primary/`](../../results/primary/README.zh-CN.md)。784-64-40，5 个 epoch，批大小 10，`lr = 0.01`，`β = 1.0`，种子 0/1/2，默认退火日程。

### 各种子

| 种子 | 准确率 | 平局率 | ‖ΔW‖ | ‖ΔV‖ | 各 epoch 的隐层翻转率 |
|---|---|---|---|---|---|
| 0 | 0.15 | 0.62 | 2.493 | 2.657 | 0.685, 0.412, 0.318, 0.240, 0.232 |
| 1 | 0.54 | 0.06 | 3.498 | 3.349 | 0.930, 0.829, 0.664, 0.612, 0.524 |
| 2 | 0.46 | 0.18 | 3.000 | 3.049 | 0.885, 0.728, 0.635, 0.551, 0.522 |

### 跨种子

| 指标 | 均值 | 标准差 | 最小 | 最大 |
|---|---|---|---|---|
| 准确率 | 0.3833 | 0.1682 | 0.15 | 0.54 |
| 平局率 | 0.2867 | 0.2407 | 0.06 | 0.62 |

### 逐类

| 类别 | 均值 | 标准差 | 最小 | 最大 |
|---|---|---|---|---|
| 0 | 0.800 | 0.000 | 0.8 | 0.8 |
| 1 | 0.500 | 0.283 | 0.1 | 0.7 |
| 2 | 0.167 | 0.125 | 0.0 | 0.3 |
| 3 | 0.233 | 0.205 | 0.0 | 0.5 |
| 4 | 0.300 | 0.216 | 0.0 | 0.5 |
| 5 | 0.233 | 0.205 | 0.0 | 0.5 |
| 6 | 0.300 | 0.294 | 0.0 | 0.7 |
| 7 | 0.767 | 0.125 | 0.6 | 0.9 |
| 8 | 0.200 | 0.283 | 0.0 | 0.6 |
| 9 | 0.333 | 0.287 | 0.0 | 0.7 |

每个种子的求解器调用：5,000 次自由相 + 5,000 次微扰相 + 100 次评估 = 10,100。

### 观察

- 数字 2 是最弱的类别（0.17）。数字 0 和 7 最强（0.80 和 0.77），也是每个种子都高于 0.5 的仅有类别。
- 类别 6 的离散最大（标准差 0.29），类别 1、8、9 紧随其后，这些类别的准确率强烈依赖种子。
- 隐层状态发生变化的样本比例从 0.69–0.93 起步，五个 epoch 后降到种子 0 的 0.23 和另外两个种子的约 0.52。每个 epoch 都高于 0.05 的学习阈值。
- 平局率与准确率反向变化：种子 0 既最弱（0.15），平局也最多（0.62，其余为 0.06 和 0.18）。它的大部分测试图像上，输出块得分相同，预测来自局域场回退，而不是一个明确的输出状态。

### 记录的基线

| 运行 | 准确率 |
|---|---|
| 论文风格的软件退火器 | 0.15 |
| 平凡门 | 0.10 |
| 同一划分上的反向传播 MLP | 0.91 |
| **本实验** | **0.3833** |

高于退火器基线，低于反向传播。网络、求解器和数据规模见[项目局限](../../README.zh-CN.md#局限)。
