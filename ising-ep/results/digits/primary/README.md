# Digits reference run

[English](README.md) · [中文](README.zh-CN.md)

Read-only three-seed run of the default configuration on sklearn Digits. The
tuned runs are in [`../tuned/`](../tuned/README.md).

Do not overwrite this directory. Point `--output` at a new path instead.

The archive keeps JSON only. Parameter snapshots, flip rates and per-sample
predictions stay in the local run directory.

## Provenance

Produced on 2026-11-14 with Kaiwu 1.4.1 by the current code (`config.json`
records `2026-11-13T18:05:57Z`). Command:

```powershell
python -m examples.digits_kaiwu.experiment --output runs/digits-final/primary `
  --seeds 0,1,2 --epochs 5 --validation-fraction 0.2 --eval-every-epoch
```

Split seed 0, the default: all 1,797 images are stratified, 360 are held out
for testing, then 288 of the remainder become validation, leaving 1,149 for
training. Each seed reads the test split once.

## Headline numbers

64-64-40, 5 epochs, batch 10, `lr = 0.01`, `β = 1.0`. The standard deviation is
the population standard deviation of the three seeds.

| Seed | Test accuracy | Tie rate | Validation accuracy after epoch 5 |
|------|---------------|----------|-------------------------------------|
| 0 | 0.0972 | 1.00 | 0.097 |
| 1 | 0.1028 | 1.00 | 0.101 |
| 2 | 0.1000 | 1.00 | 0.101 |
| **mean** | **0.1000 ± 0.0023** | 1.00 | 0.100 |

13,290 solver calls per seed: 5,745 free + 5,745 nudged in training
(1149 × 5 × 2), 1,440 validation (288 × 5), and 360 test.

Every test image was a tie, so the accuracy is chance. At the 3-epoch
screening stage this configuration also sat near 11% and assigned almost every
validation image to one class. It did not learn.

A backpropagation network on the same split and the same training budget
reaches 93.52% ± 0.65%. See
[`examples/digits_kaiwu/`](../../../examples/digits_kaiwu/README.md).
