# MNIST closeout evidence / MNIST 结项证据

The full-data EP and BP jobs completed three epochs, along with 15 validation-only subset comparisons. Raw summaries and published figures are retained unchanged. Full-data results use seed 0; the subset study uses seeds 0/1/2. BP is an experimental comparison script, not a plugin API or a second EP implementation.

完整 EP、BP 三轮训练和 15 个子集对照均已完成。原始 JSON 与发布曲线保持不变。完整数据结果为单种子结果，子集对照仅使用验证集。BP 脚本作为结项实验附件交付，不加入插件导出接口。

## Environment and provenance / 环境与来源

Use Python 3.10.11, Kaiwu 1.3.1, PyTorch 2.7.0, torchvision 0.22.0, NumPy 2.2.6 and matplotlib 3.10.3; see `environment.json`. Install the Kaiwu wheel separately and initialize its local license if required. From the repository root, install this checkout into that environment:

```bash
python -m pip install -e .
```

Check the imported package location before reproduction, especially if another plugin installation exists. The runner fixes `OMP_NUM_THREADS=1` and `MKL_NUM_THREADS=1`, as in the archived experiments. Measured times depend on the machine. Datasets, checkpoints, credentials and virtual environments are not included.

`provenance.json` identifies base commit `93a536b` plus local changes and the training-script hash. The BP and dispatcher sources were adapted from the actual experiment scripts: paths and dispatch interfaces changed, while the BP computation and experiment configurations were retained. Historical source hashes and adapted-script hashes are recorded separately. The convolutional result remains a historical one-epoch run.

使用前安装当前检出及对应 SDK，必要时初始化本地授权。脚本经过路径与调度接口适配，不声称与历史版本逐字节一致；原始脚本与交付脚本的哈希分别记录。调度程序固定上述两个线程环境变量。数据集、检查点、授权信息和虚拟环境不随附件交付。

## Run experiments / 运行实验

Commands below run from the repository root. Supply an MNIST data directory and a new output directory. Paths containing spaces must be quoted. The default selection runs BP, the one-epoch probe, full EP, then the 15 subset jobs. Use `--dry-run` first: it prints commands without creating directories, loading data or starting training.

以下命令从仓库根目录执行；数据路径可自行指定，输出目录必须尚不存在。`--dry-run` 只列出命令，无训练和写入副作用。默认依次运行 BP、单轮测速、完整 EP 和 15 个子集任务；不需要重新训练时直接使用下方图表再生成命令。

```bash
python example/ep_mnist/closeout/run_experiments.py --data-dir data --output-dir runs/closeout-reproduction --dry-run
python example/ep_mnist/closeout/run_experiments.py --data-dir data --output-dir runs/closeout-reproduction
python example/ep_mnist/closeout/run_experiments.py --data-dir data --output-dir runs/bp-reproduction --only bp
python example/ep_mnist/closeout/run_experiments.py --data-dir data --output-dir runs/ep-reproduction --only ep-full
python example/ep_mnist/closeout/run_experiments.py --data-dir data --output-dir runs/sweep-reproduction --only sweep
```

The official MNIST data has 60,000 training and 10,000 test images. Split seed 0 reserves 5,000 of the training images for validation. Full EP and BP use seed 0, three epochs, batch size 20, SGD learning rate 0.05 and widths 784-128-10. BP uses ReLU and cross entropy; EP uses discrete spins, beta 2, initialization scale 2 and 50 annealing iterations per temperature.

The subset matrix is `(beta, iterations per temperature) = (2,50), (1,50), (4,50), (2,20), (2,100)`, each with seeds 0/1/2, 2,000 training images, 1,000 validation images and three epochs. Every subset job passes `--validation-only`; it never evaluates test accuracy. The job names match the archived summaries.

官方 MNIST 为 60,000 张训练图和 10,000 张测试图；项目以种子 0 留出 5,000 张验证图。完整 EP/BP 使用种子 0、三轮、批大小 20、学习率 0.05 和 784-128-10 层宽。子集参数矩阵及种子如上，均使用 2,000 张训练图、1,000 张验证图和三轮训练，显式跳过测试集。

Each task records its command, status and exit code in `status.json`, with separate task logs. Training failures do not prevent later jobs from running; any failed task or report generation makes the runner exit nonzero. Inspect `summary_returncode` and `summarize.log` for report failures. Existing output directories are never overwritten.

任务命令、状态与退出码写入 `status.json`，每项有独立日志。训练失败后继续后续任务，但只要任一任务或汇总失败，整个调度程序就返回非零退出码；汇总失败可查看 `summary_returncode` 和 `summarize.log`。已存在的输出目录不会被覆盖。

## Regenerate figures without training / 无需训练的图表再生成

```bash
python example/ep_mnist/closeout/summarize.py example/ep_mnist/closeout --output-dir runs/closeout-figures
```

This creates a new directory containing CSV, PNG, PDF and `results.md`. Published copies are in [static assets](../../../docs/_static/ep-mnist/). Omitting `--output-dir` retains the original in-place reporting behavior used by the dispatcher; use the command above when reading the immutable archive.

该命令在新目录中生成 CSV、PNG、PDF 和结果说明，不改动归档。省略 `--output-dir` 时保留调度程序使用的原地汇总行为。曲线为训练和验证分类错误率，不是逐轮测试准确率；EP 两相能量差与 BP 交叉熵不应直接比较数值大小。
