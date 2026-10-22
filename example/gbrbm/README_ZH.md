**语言版本**：[中文](README_ZH.md) | [English](README.md)

### 在连续值数据上训练高斯-伯努利 RBM

本示例演示 `GaussianBernoulliRestrictedBoltzmannMachine` 在实值数据上的完整工作流——当输入是连续值（传感器读数、图像灰度、物理量测量）而非二值时，应使用这一模型。

运行方式：

```bash
python example/gbrbm/run_gbrbm.py
```

脚本完成以下工作：

* 构造一份带相关性的合成连续值数据集；
* 使用对比散度（Contrastive Divergence）训练模型：正相用 `infer_from_gaussian` 将数据补全为含采样伯努利单元的完整状态，负相用 `gibbs_sample` 从同一批数据出发做一次模型扫描，随后用普通 SGD 最小化 `bm.objective(s_positive, s_negative)`（其梯度与负对数似然的梯度一致）；
* 用带 burn-in 的 Gibbs 链（`gibbs_sample` 的 `n_burnin` 参数）从训练好的模型生成新样本；
* 对比数据与生成样本逐维的均值和标准差，并用 `marginal_energy`（高斯态的自由能）为两者打分。

预期输出（固定种子，CPU）：

```text
training on continuous data (local Gibbs negative phase)...
epoch  100  objective     0.1418
epoch  200  objective    -0.0345
epoch  300  objective     0.0182
objective: start 1.9143 -> end 0.0182

per-dimension mean (data vs generated):
  dim 0:   0.9981    1.0140
  dim 1:  -0.3382   -0.2790
  dim 2:   0.3734    0.3448
  dim 3:  -1.2853   -1.3006

per-dimension std (data vs generated):
  dim 0:   1.0176    1.0236
  dim 1:   1.1600    1.1729
  dim 2:   1.1005    1.1170
  dim 3:   1.0358    1.0590

marginal energy (free energy of Gaussian states): data -0.8093, generated -0.7373
```

模型的高斯能量在可见单元上是对角的，因此拟合目标是逐维的均值与方差，而不是数据的非对角相关性。

**依赖项**：除包自身要求（`torch`、`kaiwu`）外无额外依赖。默认路径的负相使用本地 Gibbs 采样，**无需 Kaiwu SDK license**。

#### 使用 Kaiwu SDK 求解器

量子插件路径将负相的本地 Gibbs 扫描替换为 `bm.sample(sampler)`：把模型伯努利一侧转成 Ising 问题并交给 Kaiwu SDK 求解。将 `run_gbrbm.py` 顶部的 `USE_SOLVER` 设为 `True`，安装 SDK（`pip install kaiwu==1.3.1`），并参照 `example/run_rbm.py` 初始化 license：

```python
import os
import kaiwu as kw

kw.license.init(os.getenv("USER_ID"), os.getenv("SDK_CODE"))
```

凭据从 [platform.qboson.com](https://platform.qboson.com) 获取（详见[安装指南](https://kaiwu-pytorch-plugin.readthedocs.io/en/latest/source/getting_started/installation.html)）。
