# 基于 Kaiwu 的 Ising 均衡传播

[English](README.md) · [中文](README.zh-CN.md)

均衡传播（equilibrium propagation，EP）只用网络的两次弛豫来训练基于能量的模型，不需要反向传播。本项目在**离散 Ising 自旋**上实现 EP，并把每一次弛豫交给 [Kaiwu](https://platform.qboson.com/) SDK 的模拟退火器，因此“弛豫”是一次真正的组合优化求解，而不是梯度下降。

主实验用 64-64-40 网络对 sklearn Digits 做分类。默认配置三次种子的测试准确率是 **10.0% ± 0.2%**（平局率 100%）；在验证集上把输出层学习率减半之后是 **49.9% ± 1.3%**。同一划分上的反向传播对照是 93.52% ± 0.65%。归档在 [`results/digits/`](results/digits/tuned/README.zh-CN.md)，数字复述于下方[结果](#结果)。MNIST/100 仍作为第二个实验保留，见 [`results/primary/`](results/primary/README.zh-CN.md)。

---

## 算法

### 网络与编码

输入 `x` 是被钳制的像素值。隐层自旋 `h` 和输出自旋 `o` 都是双极的，`s ∈ {-1, +1}`。10 个类别各占 4 个输出自旋，因此 `n_output = 40`。类别 `c` 的目标把对应的那一块设为 `+1`，其余自旋设为 `-1`。

### 能量与代价

```
E(x, h, o) = -x·W·h - h·V·o - b_h·h - b_o·o
C(o, y)    = n_output - y·o            # 对 ±1 自旋等于 0.5*||o - y||^2
F_β        = E + β·C
```

输出的平方误差因此塌缩成 `o` 上的一个*线性*项。整个目标保持二次，从而可以写成 QUBO 来求解。

### 两个相

把输入项和偏置收进局域场

```
H_h = x·W + b_h
H_o = b_o + β·y
```

目标就只剩下自旋：

```
F_β(h, o) = -h·V·o - H_h·h - H_o·o + β·n_output
```

**自由相**在 `β = 0` 下求解，标签信息不进入问题。**微扰相**在 `β > 0` 下求解，标签只通过 `H_o` 出现。微扰相从自由相的解热启动。这是本项目对量子硬件上反向退火的替代：两个状态保持接近，它们的差反映的是微扰，而不是求解器噪声。

### 梯度

EP 估计量是两个相的关联之差。这里通过对下式做自动微分得到：

```
loss = (E(x, h_nudged, o_nudged) - E(x, h_free, o_free)) / β
```

自旋是求解器返回的常数，对上式求导正好得到

```
dW   = (x ⊗ h_free - x ⊗ h_nudged) / β
dV   = (h_free ⊗ o_free - h_nudged ⊗ o_nudged) / β
db_h = (h_free - h_nudged) / β
db_o = (o_free - o_nudged) / β
```

因此由 PyTorch 代替手写公式。一个批次里的每个样本都对着*同一组*参数求解，每个批次做一次优化器更新。

### QUBO 映射

Kaiwu 接受上三角 QUBO。代入 `s = 2q - 1`，其中 `q ∈ {0, 1}`，并用 `q² = q` 把所有线性项折到对角线上：

```
Q[:n_h, :n_h] = diag(2·V.sum(axis=1) - 2·H_h)
Q[n_h:, n_h:] = diag(2·V.sum(axis=0) - 2·H_o)
Q[:n_h, n_h:] = -4·V
constant      = β·n_output + H_h.sum() + H_o.sum() - V.sum()
```

`constant` 只平移目标值，不改变最小化点，所以和 `Q` 一起返回，不折进矩阵。

### 更深的网络

模型是一叠层：`--hidden-sizes 128,64` 得到 784-128-64-40。第 `l` 层有自己的权重 `W_l` 和偏置 `b_l`，能量只对相邻层求和：

```
E = -Σ_l s_{l-1}·W_l·s_l - Σ_l b_l·s_l        （s_0 = x，最后一个 s 是输出）
```

`IsingEPModel.qubo(x, target, β)` 把它折成一个块三对角的 `Q`。每个耦合在两层之间的块上加 `-4·W`，在两个对角块上各加 `2·W.sum(axis)`；每个场加 `-2·H`。上面单隐层的情形就是隐层到输出的耦合 `V`，以及场 `H_h`、`H_o`。学习率按层设置（`--lr-weights`、`--lr-biases`，每个连接一个值），另有只作用于权重的 `--weight-decay`，以及缩放各层初始均匀分布范围的 `--init-gains`。

默认的 `--n-hidden 64` 构造出的网络和初始权重，与归档运行完全相同。

### 辅助自旋

SDK 的 QUBO 到 Ising 转换会追加一个规范自旋（Ising 耦合矩阵没有线性项，线性部分需要一个参考自旋）。求解器返回的每一行因此有 `n + 1` 个元素。按 Ising 目标排序候选并做负尾翻转之后，这个规范自旋被固定为 `+1`。两种规范选择变成同一个物理状态，也就是前 `n` 个元素。Ising 目标与 EP 能量只差一个常数，所以最小化点相同。退火由种子固定候选，因此同一种子、同一配置的运行逐位可复现。

`kaiwu_solver.py` 在调用时选择 SDK 入口，同一条代码路径覆盖 Kaiwu 1.3.1 到 1.4.3。1.4.x 的辅助函数在 `kaiwu.core` 上（`qubo_model_to_ising_model`、`get_sorted_solutions`）；1.3.1 里它们是 `kaiwu.qubo.qubo_matrix_to_qubo_model` 和 `kaiwu.conversion`，排序与负尾翻转按同一规则在本地完成。

---

## 目录结构

```
src/ising_ep/            可复用核心，不含实验细节
  model.py               层堆叠、参数、EP 能量和 QUBO（torch.nn.Module）
  kaiwu_solver.py        唯一导入 kaiwu 的模块
examples/mnist_kaiwu/    MNIST/100 实验
  data.py                确定性的 MNIST/100 子集
  train.py               两相训练循环
  evaluate.py            只用自由相的评估
  metrics.py             准确率、动力学与学习证据
  artifacts.py           JSON/NPZ 写入，以及可选的凭证读取
  experiment.py          命令行入口
examples/digits_kaiwu/   sklearn Digits 实验与反向传播对照
tests/unit/              不需要 SDK，也不需要凭证
tests/integration/       没有授权 SDK 时自动跳过
results/digits/          Digits 主实验：参考运行与调参运行
results/primary/         MNIST/100 的三次种子运行（只读证据）
results/tuned/           MNIST/100 的调参运行
```

只有 `kaiwu_solver.py` 接触 SDK，而且是懒导入，所以其余部分在没有企业版 wheel 时也能导入和测试。

---

## 安装

支持的 Kaiwu 版本为 1.3.1 到 1.4.3（`pyproject.toml` 只在可选的 `kaiwu` 额外项里写了 `kaiwu>=1.3.1,<=1.4.3`，因为 wheel 不在 PyPI 上；请按下文手动安装 wheel，没有它时 `pip install -e` 也能成功）。归档结果产生于 1.4.1。1.4 系列在第一次退火前会要求用户 ID 和 SDK 码（见 `.env.example`），1.3.1 不需要。这些 wheel 面向 **Python 3.10**。

集成测试已在真实的 1.3.1 和 1.4.1 wheel 上跑过；1.4.2 和 1.4.3 在声明范围内，但没有实测。1.4 的 wheel 依赖 `kaiwu-community` 提供 `kaiwu/__init__.py` 和 `kaiwu.core`。在同一个环境里把 1.3.1 换成 1.4.x 会删掉这些共用文件，换完后要执行 `pip install --force-reinstall --no-deps kaiwu-community==1.0.7`，或者直接新建环境。

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

如果不想安装本包，`requirements.txt` 固定了同样的运行和开发依赖（numpy、torch、pytest、pylint、pydocstyle）：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

它不包含 Kaiwu wheel，wheel 需要另外安装。

<details>
<summary>没有网络时安装</summary>

`pip install -e .` 默认在隔离环境里构建，也就是要从 PyPI 拉取 `setuptools>=68`。离线时改用解释器自带的构建工具：

```powershell
.\.venv\Scripts\python.exe -m pip install wheel
.\.venv\Scripts\python.exe -m pip install -e . --no-build-isolation --no-deps
```

需要 `wheel`，是因为 70 以下的 setuptools 没有内置 `bdist_wheel` 命令，否则构建会停在 `invalid command 'bdist_wheel'`。使用 `--no-deps` 时，请确认 `numpy`、`torch` 和开发工具已经装好。

如果完全无法安装，测试仍然可以跑：`pyproject.toml` 设置了 `pythonpath = [".", "src"]`，裸检出目录里 `pytest` 也能解析 `ising_ep`。直接运行实验模块则需要包可导入，要么安装，要么设置 `PYTHONPATH=src`。

</details>

Kaiwu 的 wheel 不在 PyPI 上。从量子云平台下载对应操作系统和架构的 `cp310` wheel，装进同一个解释器：

```powershell
.\.venv\Scripts\python.exe -m pip install <kaiwu-1.4.1-cp310-...>.whl
.\.venv\Scripts\python.exe -c "import importlib.metadata as m; print(m.version('kaiwu'))"
```

最后一条命令会打印已安装的版本。1.4.x 上 `kaiwu.__version__` 报的是另一个过期值，以发行版版本为准。

1.4 系列在第一次退火前就需要凭证：把 `.env.example` 复制为 `.env`，填入 `KAIWU_USER_ID` 和 `KAIWU_SDK_CODE`。1.3.1 不需要。查找顺序：先看环境变量，再看项目目录下的 `.env`，最后看上一级目录（工作区根目录）的 `.env`。不要提交 `.env` 或许可证文件。

### MNIST 数据

数据不随仓库提供。把四个 IDX 压缩包放到 `data/MNIST/raw/`，或者在第一次使用时加上 `--download-data` 下载。下载是显式选项，训练不会意外访问网络。

---

## 运行

Digits 主实验。产生 [`results/digits/tuned/lrout05/`](results/digits/tuned/lrout05) 的命令：

```powershell
.\.venv\Scripts\python.exe -m examples.digits_kaiwu.experiment `
  --output runs/digits-lrout05 --seeds 0,1,2 --epochs 5 `
  --validation-fraction 0.2 --eval-every-epoch `
  --lr-weights 0.01,0.005 --lr-biases 0.01,0.005
```

同一划分的反向传播对照不需要 Kaiwu：

```powershell
.\.venv\Scripts\python.exe -m examples.digits_kaiwu.backprop `
  --output runs/digits-backprop --seeds 0,1,2
```

MNIST/100 是第二个实验。冒烟运行（一个种子，小网络，几分钟）：

```powershell
.\.venv\Scripts\python.exe -m examples.mnist_kaiwu.experiment `
  --output runs/smoke --seeds 0 --n-hidden 8 --epochs 1 `
  --candidate-count 2 --iterations-per-t 4
```

产生 `results/primary/` 的完整三次种子运行：

```powershell
.\.venv\Scripts\python.exe -m examples.mnist_kaiwu.experiment `
  --output runs/primary --seeds 0,1,2 `
  --n-hidden 64 --epochs 5 --batch-size 10 --lr 0.01 --beta 1.0
```

带逐层学习率的更深网络（三个连接，所以各给三个值）：

```powershell
.\.venv\Scripts\python.exe -m examples.mnist_kaiwu.experiment `
  --output runs/deep --seeds 0 --hidden-sizes 128,64 `
  --lr-weights 0.01,0.01,0.02 --lr-biases 0.005,0.005,0.01
```

本仓库没有记录更深网络在 MNIST 上的准确率。Digits 上 `--hidden-sizes 128` 和 `128,64` 在筛选阶段没有超过参考，见 [`results/digits/tuned/`](results/digits/tuned/README.zh-CN.md)。

`--help` 列出全部选项，包括退火日程（`--initial-temperature`、`--alpha`、`--cutoff-temperature`、`--iterations-per-t`、`--candidate-count`）。

### 输出

```
<output>/config.json              完整配置，用于复现
<output>/summary.json             跨种子统计
<output>/seeds/<seed>/
    initial-model.npz             训练前的参数
    checkpoint.npz                最后一个 epoch 之后的参数
    epochs/<e>/model.npz          每个 epoch 的快照
    epochs/<e>/dynamics.json      该 epoch 的翻转率
    test/metrics.json             准确率、平局率、混淆矩阵、逐类结果
    test/predictions.json         逐样本预测
    status.json                   该种子的摘要与学习证据
```

`.npz` 文件保存 `weights.0 … weights.K` 和 `biases.0 … biases.K`，输入层在前。归档运行早于这个布局，用的是单隐层的旧名字：`W`、`V`、`b_h`、`b_o` 分别对应 `weights.0`、`weights.1`、`biases.0`、`biases.1`，数值相同。

提交进 `results/` 的只有 `config.json`、`summary.json`、各种子的 `status.json` 和 `test/metrics.json`，以及带验证集的运行里每个 epoch 的 `validation.json`。参数快照、翻转率和逐样本预测留在本地的 `runs/`。

所有随机量都从种子派生，因此给定的 `(种子, 配置)` 逐位复现：初始化、每个 epoch 的打乱、每个样本成对退火的种子，以及评估种子。

---

## 结果

### Digits

归档于 [`results/digits/`](results/digits/tuned/README.zh-CN.md)。网络 64-64-40，5 个 epoch，批大小 10，种子 0/1/2，划分种子 0（训练 1149 / 验证 288 / 测试 360）。退火与 MNIST 相同：`T₀ = 1.25`，`α = 0.95`，截止温度 `0.001`，每个温度 10 次迭代，每次求解 8 个候选。超参数只在验证集上选，测试集每个种子只读一次。

| 配置 | 测试准确率 | 平局率 |
|------|------------|--------|
| 默认（`lr = 0.01`，`β = 1`） | 10.0% ± 0.2% | 100% |
| 输出层学习率 ×0.5 | **49.9% ± 1.3%** | 6.6% |
| 再加 `β = 2` | 49.3% ± 0.6% | 5.0% |
| 反向传播 64-64-10（`lr = 0.05`） | 93.52% ± 0.65% | |

标准差是三个种子的总体标准差。默认配置在测试集上每张图都是平局。两个调参配置之间差 0.6 个点，小于种子之间的波动。反向传播用同一划分和同样的 epoch 数、批大小，学习率是它自己的默认值。

### MNIST/100

归档于 [`results/primary/`](results/primary/README.zh-CN.md)。配置：784-64-40，5 个 epoch，批大小 10，`lr = 0.01`，`β = 1.0`，种子 0/1/2；退火 `T₀ = 1.25`，`α = 0.95`，截止温度 `0.001`，每个温度 10 次迭代，每次求解 8 个候选。

后续的调参研究（超参数在验证集上选，测试集只用一次）见 [`results/tuned/`](results/tuned/README.zh-CN.md)：输出层初始增益 0.5 的测试准确率为 62.7% ± 1.9%。上面的基线数字不变。

每个种子做 **10,100 次求解器调用**：训练中 5,000 次自由相 + 5,000 次微扰相（1,000 个样本 × 5 个 epoch × 2 个相），再加上评估的 100 次。

| 种子 | 准确率 | 平局率 | ‖ΔW‖ | ‖ΔV‖ |
|------|--------|--------|------|------|
| 0 | 0.15 | 0.62 | 2.49 | 2.66 |
| 1 | 0.54 | 0.06 | 3.50 | 3.35 |
| 2 | 0.46 | 0.18 | 3.00 | 3.05 |
| **均值** | **0.3833** | 0.2867 | | |
| 标准差 | 0.1682 | 0.2407 | | |

最差的种子也是平局最多的：种子 0 在 62% 的测试图像上退回到局域场打破平局，另外两个种子是 6% 和 18%。

逐类准确率（各种子的均值）：

| 类别 | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|------|---|---|---|---|---|---|---|---|---|---|
| 准确率 | 0.80 | 0.50 | 0.17 | 0.23 | 0.30 | 0.23 | 0.30 | 0.77 | 0.20 | 0.33 |

数字 2 最弱，为 0.17；数字 0 和 7 最强。两个相之间隐层状态发生变化的样本比例，随种子和 epoch 落在 0.23 到 0.93；每个 epoch 都高于 0.05，所以微扰没有塌缩成自由相状态。

同一划分上记录的基线：论文风格的软件退火器 0.15，平凡门 0.10，反向传播 MLP 0.91。见[局限](#局限)。

---

## 测试与检查

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q
.\.venv\Scripts\python.exe -m pytest tests -m integration
.\.venv\Scripts\python.exe -m pylint src/ising_ep examples tests
.\.venv\Scripts\python.exe -m pydocstyle --convention=google src examples
```

单元测试不需要 SDK，也不需要凭证：注入一个假的 `kaiwu` 模块来断言官方调用链。没有 wheel 时，集成测试自动跳过。

文档字符串是强制的：启用了 `pylint.extensions.docparams`，且 `accept-no-param-doc = false`，缺少 `Args`、`Returns` 或 `Raises` 会让检查失败。

---

## 局限

- **网络。** 归档运行用 64 个隐层自旋，文献中是 120。更深的堆叠已支持，但这里没有做基准测试。
- **求解器。** Kaiwu 经典模拟退火，微扰相热启动。文献使用 D-Wave 反向退火。
- **数据。** 主实验是 sklearn Digits：1149 张训练、288 张验证、360 张测试。MNIST/100 仍在：1,000 张训练图像，100 张测试图像。
- **准确率。** Digits 上，调参后的 EP 是 **49.9% ± 1.3%**，同一划分的反向传播是 93.52% ± 0.65%，默认 EP 配置是 10.0% ± 0.2%。MNIST/100 的参考运行是 **38.3% ± 16.8%**；同一划分上的基线：软件退火器 0.15，平凡门 0.10，反向传播 MLP 0.91。

## 文献

论文不包含在本仓库中。

- Laydevant、Marković、Grollier，*Training an Ising machine with equilibrium propagation*（2024）。离散 Ising 均衡传播，±1 自旋，每个类别一块输出自旋。
- Scellier、Bengio，*Equilibrium Propagation: Bridging the Gap between Energy-Based Models and Backpropagation*（2017）。``F_β`` 目标，以及两相梯度。
- Ernoult 等，*Equilibrium Propagation with Continual Weight Updates*（2020）。Laborieux 等，*Scaling Equilibrium Propagation to Deep ConvNets by Drastically Reducing Its Gradient Estimator Bias*（2021）。Scellier 等，*Quantum Equilibrium Propagation*（2024）。
