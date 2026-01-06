# 【论文复现：将论文中的基于能量的奖励模型（EBRM）替换为QBM】

## 背景描述

奖励模型（RMs）对于将大语言模型（LLMs）与人类偏好对齐至关重要，然而它们往往难以捕捉复杂的人类偏好，并难以泛化至未见数据。本任务需将论文《Energy-Based Reward Models for Robust Language Model Alignment》中提到的基于能量的奖励模型（EBRM）中的Energy Score模块替换为QBM，并使用文章中提到的数据集进行结果对比验证。

详情见[issue #78](https://github.com/qboson/kaiwu-pytorch-plugin/issues/78)。

项目文件：

```
├── README.md
├── imgs
│   ├── pairwire-training_plots.png
│   └── training_plots.png
├── model_final.pth
├── prepare_rmb_dataset.py
├── rmb_dataset.pt
├── rmb_dataset2.pt
├── rmb_dataset_pairwise.pt
├── rmb_dataset_train.pt
├── rmb_dataset_val.pt
├── run_diagnostic_train.py
├── run_qbm_ebm.py # smoke text
├── run_train_qbm_ebm.py
├── save_and_plot_results.py
└── training_metrics.npz
```

我们使用[RMB](https://github.com/Zhou-Zoey/RMB-Reward-Model-Benchmark)数据集进行测试，如果该数据集对你有帮助请引用：

```
@inproceedings{lambert2025rewardbench,
  title={Rewardbench: Evaluating reward models for language modeling},
  author={Lambert, Nathan and Pyatkin, Valentina and Morrison, Jacob and Miranda, Lester James Validad and Lin, Bill Yuchen and Chandu, Khyathi and Dziri, Nouha and Kumar, Sachin and Zick, Tom and Choi, Yejin and others},
  booktitle={Findings of the Association for Computational Linguistics: NAACL 2025},
  pages={1755--1797},
  year={2025}
}
```
## 复现步骤

### 数据管线的搭建

我们首先构建了[prepare_rmb_dataset.py](example/qbm_ebrm_results/prepare_rmb_dataset.py)将json格式的数据集转化为pt格式文件用于适配dataloader。

### 算法的搭建

然后将EBRM中的energy score算法替换为QBM算法，原论文中energy score算法接受两个参数传统RM算法输出的特征值'embedding'以及RM算法的打分'r'，并返回修改后的打分'r*'。新的QBM算法替代energy score算法也采取相同的输入输出参数。

### 训练及验证结果

我们分别对[RMB-Reward-Model-Benchmark](https://github.com/Zhou-Zoey/RMB-Reward-Model-Benchmark)数据集中的BoN_set以及Pairwise_set分别训练五个epochs进行测试。测试结果可视化如下：

![BoN_set](https://github.com/YuzeHao2023/kaiwu-pytorch-plugin/blob/main/example/qbm_ebrm_results/imgs/training_plots.png "BoN_set效果可视化结果")

BoN_set效果可视化结果

![Pairwise_set](https://github.com/YuzeHao2023/kaiwu-pytorch-plugin/blob/main/example/qbm_ebrm_results/imgs/pairwire-training_plots.png "Pairwise_set效果可视化结果")

Pairwise_set效果可视化结果

## 快速开始

### 安装

首先将kaiwu-pytorch-plugin项目fork，然后再您本地运行下面代码将其下载到本地：

```bash
git clone https://github.com/qboson/kaiwu-pytorch-plugin.git
```

安装KPP:

```bash
cd kaiwu-pytorch-plugin
pip3 install -r requirements/requirements.txt
pip3 install .
```

### 数据集转化

使用下面指令下载数据集：

```bash
git clone https://github.com/Zhou-Zoey/RMB-Reward-Model-Benchmark.git
```

将[prepare_rmb_dataset.py](example/qbm_ebrm_results/prepare_rmb_dataset.py)中的路径修改为实际需要的路径：

```python
src = os.path.join(repo_root, 'RMB-Reward-Model-Benchmark', 'RMB_dataset', 'BoN_set', 'Harmlessness', 'S2.json')
```

然后运行：

```bash
python3 example/qbm_ebrm_results/prepare_rmb_dataset.py
```

即可生成对应的pt格式数据集文件。

### 训练及可视化

在wandb离线状态下进行训练，

```bash
python3 example/qbm_ebrm_results/run_train_qbm_ebm.py
WANDB_MODE=offline
python3 example/qbm_ebrm_results/save_and_plot_results.py
```

可视化部分见训练及验证结果部分图片。

---

如果这篇论文对您的研究有帮助，请引用下面的论文：

```
@article{lochab2025energy,
  title={Energy-Based Reward Models for Robust Language Model Alignment},
  author={Lochab, Anamika and Zhang, Ruqi},
  journal={arXiv preprint arXiv:2504.13134},
  year={2025}
}
```

