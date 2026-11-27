"""Plot CPU EP/BP reference errors without training or parameter searches."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def main():
    """Write the two published assets from the adjacent reference summaries."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).resolve().parent / "reference_results"
    rows = []
    figure, axis = plt.subplots(figsize=(7.2, 4.2), layout="constrained")
    for name, label in (("ep-full-cpu", "CPU EP"), ("bp-full-cpu", "CPU BP")):
        summary = json.loads((source / name / "summary.json").read_text(encoding="utf-8"))
        history = summary["history"]
        for epoch in history:
            rows.append({"model": label, "epoch": epoch["epoch"],
                         "train_error": 1 - epoch["train_accuracy"],
                         "val_error": 1 - epoch["val_accuracy"]})
        for split, style in (("train", "-"), ("val", "--")):
            axis.plot([row["epoch"] for row in history],
                      [100 * (1 - row[split + "_accuracy"]) for row in history],
                      linestyle=style, marker="o", label=f"{label} {split}")
    axis.set(xlabel="Epoch", ylabel="Classification error (%)",
             title="MNIST: CPU EP and BP (seed 0)", xticks=[1, 2, 3])
    axis.grid(alpha=0.25)
    axis.legend()
    figure.savefig(args.output_dir / "training_curves.png", dpi=180)
    plt.close(figure)
    with (args.output_dir / "training_curves.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
