"""Build reproducible tables and figures from completed MNIST runs."""

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, pstdev

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def dump_csv(path, rows):
    """Save a table when at least one result exists."""
    if not rows:
        return
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def save_figure(figure, output, name):
    """Export for both quick inspection and report inclusion."""
    figure.savefig(output / f"{name}.png", dpi=180)
    figure.savefig(output / f"{name}.pdf")
    plt.close(figure)


def training_curves(output, summaries):
    """Plot shared classification errors for EP and BP."""
    runs = [(name, summaries[name]) for name in ("ep-full", "bp") if name in summaries]
    if not runs:
        return
    figure, ax = plt.subplots(figsize=(7.2, 4.2), layout="constrained")
    rows = []
    for name, summary in runs:
        history = summary["history"]
        epochs = [row["epoch"] for row in history]
        for split, style in (("train", "-"), ("val", "--")):
            ax.plot(epochs, [100 * row[f"{split}_error"] for row in history],
                    linestyle=style, marker="o", label=f"{name} {split}")
        rows.extend({"run": name, **row} for row in history)
    ax.set(xlabel="Epoch", ylabel="Classification error (%)", xticks=[1, 2, 3],
           title="MNIST classification error by epoch")
    ax.grid(alpha=0.25)
    ax.legend()
    dump_csv(output / "training_curves.csv", rows)
    save_figure(figure, output, "training_curves")


def summarize_sweep(output, summaries):
    """Aggregate completed three-seed configurations, with population SD."""
    groups = {}
    seed_rows = []
    for name, summary in summaries.items():
        if not name.startswith("beta-"):
            continue
        beta, budget = summary["beta"], summary["sa_iterations"]
        row = {"beta": beta, "sa_iterations": budget, "seed": summary["seed"],
               "val_accuracy": summary["history"][-1]["val_accuracy"],
               "mean_train_seconds": mean(x["train_seconds"] for x in summary["history"]),
               "mean_eval_seconds": mean(x["eval_seconds"] for x in summary["history"])}
        groups.setdefault((beta, budget), []).append(row)
        seed_rows.append(row)
    dump_csv(output / "sweep_seeds.csv", seed_rows)
    rows = []
    for (beta, budget), runs in sorted(groups.items()):
        if len(runs) != 3:
            continue
        rows.append({
            "beta": beta, "sa_iterations": budget, "seeds": len(runs),
            "val_accuracy_mean": mean(x["val_accuracy"] for x in runs),
            "val_error_mean": 1 - mean(x["val_accuracy"] for x in runs),
            "val_accuracy_sd": pstdev(x["val_accuracy"] for x in runs),
            "train_seconds_mean": mean(x["mean_train_seconds"] for x in runs),
            "train_seconds_sd": pstdev(x["mean_train_seconds"] for x in runs),
            "eval_seconds_mean": mean(x["mean_eval_seconds"] for x in runs),
            "eval_seconds_sd": pstdev(x["mean_eval_seconds"] for x in runs),
        })
    dump_csv(output / "sweep_summary.csv", rows)
    for name, x_key, fixed, label in (
        ("beta", "beta", {"sa_iterations": 50}, "Nudging coefficient β"),
        ("budget", "sa_iterations", {"beta": 2}, "Annealing iterations per temperature"),
    ):
        points = [row for row in rows if all(row[key] == val for key, val in fixed.items())]
        if not points:
            continue
        x = [row[x_key] for row in points]
        figure, (left, right) = plt.subplots(1, 2, figsize=(9, 4), layout="constrained")
        left.errorbar(x, [100 * row["val_accuracy_mean"] for row in points],
                      yerr=[100 * row["val_accuracy_sd"] for row in points], fmt="o-", capsize=4)
        right.errorbar(x, [row["train_seconds_mean"] for row in points],
                       yerr=[row["train_seconds_sd"] for row in points], fmt="o-", capsize=4)
        left.set(ylabel="Validation accuracy (%)", xlabel=label, xticks=x)
        right.set(ylabel="Training time per epoch (s)", xlabel=label, xticks=x)
        figure.suptitle("MNIST validation subset: mean ± population SD (3 seeds)")
        left.grid(alpha=0.25)
        right.grid(alpha=0.25)
        save_figure(figure, output, f"sweep_{name}")
    return rows


def write_report(output, status, summaries, sweep_rows):
    """Write the report appendix with released results and measured timings."""
    lines = ["# 最新 MNIST 实验结果", "", "| 任务 | 状态 | 训练集 | 验证集 | 测试集 |",
             "|---|---|---:|---:|---:|"]
    labels = {"passed": "完成", "failed": "失败", "pending": "排队", "running": "进行中"}
    for name, row in status.items():
        summary = summaries.get(name)
        values = ["—"] * 3
        if summary:
            final = summary["history"][-1]
            values = [f"{final['train_accuracy']:.2%}", f"{final['val_accuracy']:.2%}",
                      f"{summary['test_accuracy']:.2%}" if "test_accuracy" in summary else "—"]
        lines.append(f"| {name} | {labels[row['state']]} | " + " | ".join(values) + " |")
    lines.extend(["", "## 完整数据集上的 EP 与 BP", "",
                  "55,000 张训练图、5,000 张验证图和 10,000 张官方测试图。"
                  "两组使用相同划分、种子 0、批量 20、学习率 0.05、784-128-10 结构和三轮训练。"
                  "EP 使用离散自旋和两相退火更新，BP 使用 ReLU、交叉熵和默认 PyTorch 初始化。",
                  "", "| 模型 | 最终测试准确率 | 平均每轮训练秒数 | 平均每轮评估秒数 |",
                  "|---|---:|---:|---:|"])
    for name in ("ep-full", "bp"):
        if name not in summaries:
            continue
        run = summaries[name]
        train_seconds = mean(row["train_seconds"] for row in run["history"])
        eval_seconds = mean(row["eval_seconds"] for row in run["history"])
        lines.append(f"| {name} | {run['test_accuracy']:.2%} | {train_seconds:.2f} | "
                     f"{eval_seconds:.2f} |")
    if "ep-full" in summaries and "bp" in summaries:
        gap = 100 * (summaries["ep-full"]["test_accuracy"] - summaries["bp"]["test_accuracy"])
        lines.extend(["", f"EP 相对 BP 的测试准确率差为 {gap:+.2f} 个百分点。"
                      "这组完整数据结果来自固定种子的同一划分。"])
    if "ep-full" in summaries or "bp" in summaries:
        lines.extend(["", "[训练曲线 PDF](training_curves.pdf) · [逐轮指标 CSV](training_curves.csv)"])
    lines.extend(["", "## β 与退火预算对照", "",
                  "各组固定 2,000 张训练图和 1,000 张验证图，训练三轮，使用种子 0、1、2。"
                  "表中准确率与误差来自第三轮，训练耗时先按轮平均再按种子汇总。"])
    if sweep_rows:
        lines.extend(["", "| β | 每温迭代 | 验证准确率 | 验证误差 | 训练秒数/轮 |",
                      "|---:|---:|---:|---:|---:|"])
        for row in sweep_rows:
            lines.append(f"| {row['beta']:g} | {row['sa_iterations']} | "
                         f"{row['val_accuracy_mean']:.2%} ± {row['val_accuracy_sd']:.2%} | "
                         f"{row['val_error_mean']:.2%} | "
                         f"{row['train_seconds_mean']:.2f} ± {row['train_seconds_sd']:.2f} |")
        lines.extend(["", "[各随机种子指标](sweep_seeds.csv) · [汇总 CSV](sweep_summary.csv)"])
    if len(sweep_rows) == 5:
        beta = max((row for row in sweep_rows if row["sa_iterations"] == 50),
                   key=lambda row: row["val_accuracy_mean"])
        budget = sorted((row for row in sweep_rows if row["beta"] == 2),
                        key=lambda row: row["sa_iterations"])
        high, low = budget[-1], budget[0]
        accuracy_change = 100 * (high["val_accuracy_mean"] - low["val_accuracy_mean"])
        time_ratio = high["train_seconds_mean"] / low["train_seconds_mean"]
        lines.extend(["", f"固定退火预算为 50 时，本子集验证均值最高的 β 为 {beta['beta']:g}。"
                      f"固定 β=2 时，预算从 {low['sa_iterations']} 增至 {high['sa_iterations']}，"
                      f"准确率变化 {accuracy_change:+.2f} 个百分点，平均每轮训练耗时为原来的 {time_ratio:.2f} 倍。",
                      "", "[β 对照图](sweep_beta.pdf) · [退火预算对照图](sweep_budget.pdf)"])
    pending = output / "results.tmp"
    pending.write_text("\n".join(lines) + "\n", encoding="utf-8")
    pending.replace(output / "results.md")


def main():
    """Regenerate the artifact set from successful jobs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output-dir", type=Path, help="new directory for regenerated figures")
    args = parser.parse_args()
    source = args.source.resolve()
    output = args.output_dir.resolve() if args.output_dir is not None else source
    status = json.loads((source / "status.json").read_text(encoding="utf-8"))
    summaries = {
        name: json.loads((source / name / "summary.json").read_text(encoding="utf-8"))
        for name, row in status.items() if row["state"] == "passed"
    }
    if args.output_dir is not None:
        output.mkdir(parents=True, exist_ok=False)
    training_curves(output, summaries)
    sweep_rows = summarize_sweep(output, summaries)
    write_report(output, status, summaries, sweep_rows)


if __name__ == "__main__":
    main()
