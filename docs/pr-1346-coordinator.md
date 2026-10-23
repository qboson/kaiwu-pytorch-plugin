# PR #1346 coordinator — `fix(example/qvae_mnist): unify trainer device placement`

Upstream: https://github.com/qboson/kaiwu-pytorch-plugin/pull/1346  
Fork branch: `cursor/qvae-mnist-trainer-device-placement-a8f8`

If `ManagePullRequest` cannot edit qboson, paste the body with:

```bash
gh pr edit 1346 --repo qboson/kaiwu-pytorch-plugin --body-file docs/pr-1346-body.md
```

(Strip this header and keep the section below `---` in `docs/pr-1346-body.md`.)

## Artifacts on fork

| Item | Location |
|------|----------|
| PR body | `docs/pr-1346-body.md` |
| Tests | `tests/test_mnist_device_placement.py` |
| Example fix | `example/qvae_mnist/trainer/trainer.py`, `model_tuner.py` |

## Validation (run from repo root)

```bash
PYTHONPATH=src pytest tests/test_mnist_device_placement.py -v
python3 -m black --check \
  example/qvae_mnist/trainer/trainer.py \
  example/qvae_mnist/trainer/model_tuner.py \
  tests/test_mnist_device_placement.py
PYTHONPATH=src python3 -m pylint tests/test_mnist_device_placement.py
```

CUDA hosts additionally execute the four `TestCudaTrainerPlacement` cases (skipped on CPU-only CI).
