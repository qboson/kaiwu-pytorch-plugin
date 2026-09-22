# 贡献指南

感谢你对 Kaiwu-PyTorch-Plugin 的关注。欢迎提交代码、文档、示例、测试和问题反馈。提交前，请先搜索现有的 [Issues](https://github.com/QBoson/Kaiwu-pytorch-plugin/issues) 和 Pull Requests，避免重复工作。

## 报告问题与提出建议

请通过 [GitHub Issues](https://github.com/QBoson/Kaiwu-pytorch-plugin/issues) 报告可复现的问题或提出功能建议。请提供所用的 Python、PyTorch、Kaiwu SDK 和 Kaiwu-PyTorch-Plugin 版本，以及最小复现步骤、实际结果和完整报错信息。

## 开始开发

请按项目安装文档使用 Python 3.10 和 conda 环境：

```bash
conda create -n quantum_env python=3.10
conda activate quantum_env
git clone https://github.com/QBoson/Kaiwu-pytorch-plugin.git
cd Kaiwu-pytorch-plugin
pip install -r requirements/requirements.txt
pip install .
```

Kaiwu SDK 为必需依赖。可直接安装 1.3.1 版本：

```bash
pip install kaiwu==1.3.1
```

其他版本的安装方式请参阅[安装指南](../getting_started/installation.md)。参与开发时，额外安装开发依赖：

```bash
pip install -r requirements/devel.txt
```

## 代码、测试与文档

- 为代码改动补充或更新对应的测试。提交前运行完整测试或受影响的测试：

```bash
pytest tests/
pytest tests/test_rbm.py
```

- 运行项目已有的代码风格检查：

```bash
pylint src/kaiwu/
```

- 遵循现有代码风格、类型标注和文档字符串格式；不要通过新增或扩大 pylint 的 `disable` 配置来规避检查。
- 用户可见行为变化时，请同步更新对应的 README、示例或 Sphinx 文档。

## 更新或新增示例

已有的基础脚本位于 `example/run_bm.py` 和 `example/run_rbm.py`。Digits、DBN、BM 生成和 Q-VAE 示例通过各自目录中的 Notebook 运行，具体入口见 [example/README_ZH.md](https://github.com/qboson/kaiwu-pytorch-plugin/blob/main/example/README_ZH.md)。

Q-Diffusion 的最小训练和生成示例可在仓库根目录执行：

```bash
pip install -r example/qdiffusion/requirements.txt
python example/qdiffusion/simple/simple_train_example.py
python example/qdiffusion/simple/simple_generate_example.py
```

新增示例时，请在 `example/` 中提供运行入口和依赖说明，并同步更新 `example/README_ZH.md` 与 `example/README.md`。

## 提交 Pull Request

请在 Pull Request 中说明改动目的、测试方式，以及对 README、示例或文档的更新。维护者会根据代码质量、测试覆盖和兼容性进行审阅。
