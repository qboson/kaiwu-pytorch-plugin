# Contributing

Thanks for helping out. The full guide (环境搭建、代码与测试、示例更新、PR 要求) lives in the documentation:

- 中文: https://kaiwu-pytorch-plugin.readthedocs.io/zh-cn/latest/source/about/contributing.html
- English: https://kaiwu-pytorch-plugin.readthedocs.io/en/latest/source/about/contributing.html

Quick version:

```bash
conda create -n quantum_env python=3.10
conda activate quantum_env
git clone https://github.com/QBoson/Kaiwu-pytorch-plugin.git
cd Kaiwu-pytorch-plugin
pip install -r requirements/requirements.txt
pip install kaiwu==1.3.1
pip install .
pip install -r requirements/devel.txt
```

Before opening a pull request:

- `pytest tests/ -q` and `pylint src/ --disable=C0114,C0116` both pass
- code changes come with tests
- user-visible changes update the README / example docs / Sphinx docs
- the PR description says what changed, why, and how you tested it (there is a template)

For bug reports, please use the issue template — the versions and the full traceback are what maintainers need to reproduce the problem. Search [existing issues and PRs](https://github.com/qboson/kaiwu-pytorch-plugin/issues) first to avoid duplicated work.
