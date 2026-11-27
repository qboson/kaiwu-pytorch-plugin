# Reference MNIST results / MNIST 参考结果

One complete CPU EP run, one CUDA EP run and one CPU BP comparison are retained.
All use three epochs, seed/split seed 0, 55000 training / 5000 validation / 10000
test images, widths 784-128-10, batch size 20 and SGD learning rate 0.05.
EP uses beta 2, input scale 2 and 50 SA iterations per temperature; full sampler
settings are in the summaries. BP uses ReLU/cross entropy and default initialization.
BP test accuracy is 96.20%; its objective is not comparable to the EP energy difference.

| Execution | Test accuracy | Total (min) | SA (min) | Non-SA (min) |
| --- | ---: | ---: | ---: | ---: |
| CPU EP | 90.30% | 353.004 | 341.725 | 11.279 |
| CUDA EP + CPU SA | 89.06% | 453.329 | 386.931 | 66.398 |

Totals include three training and train/validation evaluation phases plus final
testing, excluding startup and file-output overhead. Non-SA = total minus SA;
it includes transfers, matrix construction, gradients and Python overhead.
CPU fields come from its summary; CUDA totals use device-report phase boundaries
consistently. Both EP runs made 520000 SA calls. The model can use CUDA but SA
remains CPU based. These single-run observations used different execution times
and entrypoints, so they do not establish a controlled speedup.

All results predate the cleanup. [Provenance](provenance.json) records file hashes
and immutable links to exact executed source snapshots in the earlier commit.
The CPU/CUDA EP reports contain device/environment information; the CPU BP
summary was produced by the same archived harness and CUDA-enabled PyTorch 2.7.0
environment, with model tensors on CPU. No full training rerun is claimed.

本目录只保留 CPU/GPU 各一组 EP 完整结果及一份 CPU BP 对照。表中耗时为三轮训练、
训练/验证集评估及最终测试的阶段合计；非 SA 并非纯 GPU 运算时间。结果为单种子，
执行入口和时段不同，不能据此宣称严格加速比。源码及原始文件通过固定提交链接追溯。

To regenerate the CPU EP/BP training and validation error curves from this data:

```sh
python example/ep_mnist/plot_reference_results.py --output-dir runs/ep-reference-figures
```

The script creates only `training_curves.png` and `training_curves.csv`.
It does not train models, run parameter searches, or plot per-epoch test accuracy.
