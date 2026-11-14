# sklearn Digits 上的 Ising EP 实验

[English](README.md) · [中文](README.zh-CN.md)

用 Kaiwu 的模拟退火器完成两个弛豫相，在 sklearn Digits 上训练 64-64-40 的 Ising 均衡传播网络。同一划分上还有一个反向传播对照。算法见[项目 README](../../README.zh-CN.md)。

## 数据

`sklearn.datasets.load_digits`：1797 张 8×8 手写数字，像素除以 16 缩放到 `[0, 1]`。按类别分层，`--split-seed`（默认 0）独立于实验种子：测试集 360 张（20%），其余 1437 张用于训练。`--validation-fraction 0.2` 再从训练部分切出 288 张做验证，测试集不动。

不需要下载。`config.json` 里的 `data` 块记录了划分种子、比例和各份大小，只凭这个文件就能重建划分。

## 命令

在项目根目录运行，并保证能导入 `ising_ep`（`pip install -e ".[dev]"`，或设置 `PYTHONPATH=src;.`）。

均衡传播，默认配置：

```powershell
python -m examples.digits_kaiwu.experiment --output runs/digits-primary --seeds 0,1,2
```

先在验证集上选超参数，不读测试集：

```powershell
python -m examples.digits_kaiwu.experiment --output runs/digits-val `
  --validation-fraction 0.2 --eval-every-epoch --skip-test --seeds 0 `
  --lr-weights 0.01,0.005 --lr-biases 0.01,0.005
```

反向传播对照不需要 Kaiwu。网络是 64-64-10（tanh 隐层，交叉熵），epoch 数、批大小和数据划分与 EP 相同，学习率用它自己的默认值 0.05：

```powershell
python -m examples.digits_kaiwu.backprop --output runs/digits-backprop --seeds 0,1,2
```

QUBO 只含隐层和输出自旋，输入宽度不影响求解规模，所以一次 Digits 求解和同结构的 MNIST 求解开销相同。

## 归档结果

| 运行 | 测试准确率 | 说明 |
|---|---|---|
| EP 默认配置 | 10.0% ± 0.2% | [`results/digits/primary/`](../../results/digits/primary/README.zh-CN.md)，平局率 100% |
| EP，输出层学习率 ×0.5 | **49.9% ± 1.3%** | [`results/digits/tuned/`](../../results/digits/tuned/README.zh-CN.md) |
| EP，再加 `β = 2` | 49.3% ± 0.6% | 同上 |
| 反向传播 | 93.52% ± 0.65% | 默认配置，3 个种子，每个 epoch 约 0.04 秒 |

筛选过程和复现命令写在调参说明里。
