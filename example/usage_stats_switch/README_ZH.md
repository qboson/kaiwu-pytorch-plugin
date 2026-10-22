# KPP 用量统计开关

KPP 只在自己的采样调用路径上上报匿名用量统计。全局开关可用
`is_usage_stats_enabled()` 读取，用 `enable_usage_stats()` /
`disable_usage_stats()` 在运行时切换，或在进程启动前通过环境变量
`KPP_STATS_ENABLED` 设置（`true`、`1`、`yes`、`on` 为开启，其他值为关闭）。
代码级设置优先于环境变量。

本示例打印当前状态、切换开关，并在子进程里演示环境变量的效果。

## 内容

- `usage_stats_switch.py`：`state()`、`child_state()`，以及依次验证运行时与
  环境变量两种情况的主函数 `main()`。

## 运行

```bash
python example/usage_stats_switch/usage_stats_switch.py
```

预期输出：

```
current: enabled
after disable: disabled
after enable: enabled
child with KPP_STATS_ENABLED=true: enabled
child with KPP_STATS_ENABLED=false: disabled
usage-statistics switch OK
```

## 依赖

只需要已安装的 `kaiwu-torch-plugin`（会带入 `torch`）。不需要采样器、数据集、
checkpoint 或 Kaiwu 许可证。

## 示例校验的契约

- 环境变量未设置时默认开启；
- `KPP_STATS_ENABLED=false` 在新进程中关闭统计；
- `KPP_STATS_ENABLED` 接受 `true`、`1`、`yes`、`on`；
- 运行时调用 `enable_usage_stats()` / `disable_usage_stats()` 会覆盖环境变量。

上述不变量由 `tests/test_usage_stats_switch_example.py` 固定。
