# Coordinator notes — upstream PR #1343

Upstream: https://github.com/qboson/kaiwu-pytorch-plugin/pull/1343  
Fork branch: `cursor/feature-extractor-parameter-device-aa20` on `TNHTH/kaiwu-pytorch-plugin`

Cloud agents cannot edit `qboson/kaiwu-pytorch-plugin` PR metadata. After merging the fork branch tip into the upstream PR branch, refresh the PR body from `docs/pr-1343-body.md`:

```bash
gh pr edit 1343 --repo qboson/kaiwu-pytorch-plugin --body-file docs/pr-1343-body.md
```

Suggested title (unchanged unless coordinator prefers otherwise):

```
fix(example): align FeatureExtractor device with QVAE parameters after .to()
```

Merge gate checklist (#164):

- [ ] Failing repro documented via `_LegacyFeatureExtractor` stale-snapshot tests
- [ ] Parent `nn.Module.to()` path covered
- [ ] CUDA `construct-on-CUDA / move-to-CPU` covered when GPU available
- [ ] `black` + `pylint` clean on touched paths
- [ ] No edits under excluded open-PR themes (BM/RBM/qdiffusion library, etc.)
