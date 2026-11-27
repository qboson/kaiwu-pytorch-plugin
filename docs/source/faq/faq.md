# 常见问题

本章汇总了使用 Kaiwu-PyTorch-Plugin 时的常见问题和解决方案。

## 安装相关

### Q: 安装时提示 Python 版本不兼容？

A: KPP 0.4.0 支持 Python 3.10、3.12 和 3.14。可以用 conda 创建对应版本的环境，下面以 Python 3.10 为例。

```{code-block} bash

 conda create -n quantum_env python=3.10
 conda activate quantum_env

```
### Q: 无法导入 kaiwu.torch_plugin 模块？

A: 请检查以下几点：

1. 运行 `python -m pip show kaiwu-torch-plugin` 检查当前环境中的 KPP。
2. 确认当前环境已激活：`conda activate quantum_env`
3. 确认 Kaiwu SDK 已正确安装

如果问题仍然存在，尝试重新安装：

```{code-block} bash

 python -m pip uninstall kaiwu-torch-plugin
 python -m pip install kaiwu-torch-plugin

```
### Q: Kaiwu SDK 安装失败？

A: pip 会随 KPP 安装兼容的 Kaiwu SDK。如果安装失败，请按安装指南检查 Python 版本和平台。需要手动安装时，按以下步骤操作。

1. 访问 [Kaiwu SDK 下载页面](https://platform.qboson.com/sdkDownload)
2. 下载对应系统的安装包
3. 按照页面说明进行安装

## 模型相关

### Q: RBM 和 BM 有什么区别？应该使用哪个？

A: 主要区别如下：

```{list-table}
:widths: 30 35 35
:header-rows: 1

* - 特性
  - RBM（受限玻尔兹曼机）
  - BM（全连接玻尔兹曼机）
* - 连接结构
  - 层内无连接
  - 全连接
* - 推荐场景
  - 简单场景
  - 生成任务以及需要建模复杂依赖时

```
### Q: 如何选择隐藏层节点数？

A: 隐藏层节点数的选择取决于：

- **数据复杂度**：复杂数据需要更多隐藏节点
- **计算资源**：节点越多，计算量越大

### Q: 训练损失不下降怎么办？

A: 尝试以下方法：

1. **调整学习率**：尝试更小（如 0.001）或更大（如 0.1）的学习率
2. **增加权重衰减**：添加 L2 正则化防止过拟合
3. **检查数据预处理**：确保数据已归一化到合适范围
4. **增加训练轮次**：某些情况下需要更长的训练时间

## 采样相关

### Q: 如何获取量子计算机访问权限？

A: 请按以下步骤操作：

1. 在 [QBoson 平台](https://platform.qboson.com/) 注册账号
2. 通过平台联系官方工作人员
3. 联系邮箱：developer@boseq.com

在获取真机权限前，您可以使用模拟退火优化器进行开发和测试。

### Q: 模拟退火采样器参数如何调整？

A: `SimulatedAnnealingOptimizer` 的主要参数：

- `alpha`：降温系数。初始温度和截止温度相同时，`alpha` 越接近 1，降温越慢。
- `iterations_per_t`：每个温度下的迭代次数。
- `size_limit`：输出解的个数。

模拟退火的结果具有随机性。可用下面两组配置作为起点，在实际问题上比较耗时和解的质量。

```{code-block} python

 from kaiwu.classical import SimulatedAnnealingOptimizer

 # Faster cooling, fewer output solutions
 sampler = SimulatedAnnealingOptimizer(alpha=0.9, size_limit=5)

 # Slower cooling, more output solutions
 sampler = SimulatedAnnealingOptimizer(alpha=0.995, size_limit=100)

```
## 性能相关

### Q: 如何加速训练？

A: 几种加速方法：

1. **增大批大小**：在 GPU 内存允许的情况下增大 batch_size

2. **调整退火计算量**：用 `alpha` 调整降温速度，用 `iterations_per_t` 调整每个温度的迭代次数。

### Q: 显存不足怎么办？

A: 尝试以下方法：

1. 减小批大小（batch_size）
2. 减少隐藏层节点数
3. 使用梯度累积

## 其他问题

### Q: 如何引用这个项目？

A: 请使用以下 BibTeX 格式：

```{code-block} bibtex

 @software{KaiwuPyTorchPlugin,
     title = {Kaiwu-PyTorch-Plugin},
     author = {{QBoson Inc.}},
     year = {2024},
     url = {https://github.com/QBoson/Kaiwu-pytorch-plugin}
 }

```
### Q: 在哪里可以获取更多帮助？

A: 您可以通过以下渠道获取帮助：

1. **GitHub Issues**：[提交问题](https://github.com/QBoson/Kaiwu-pytorch-plugin/issues)
2. **开发者社区**：玻色量子开发者社区
3. **联系邮箱**：developer@boseq.com
