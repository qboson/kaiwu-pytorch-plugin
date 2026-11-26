"""Local device/timing harness; imports the unchanged final EP implementation.

CUDA runs are hybrid: model tensors/gradients on CUDA, Kaiwu SA on CPU.
All timed phases synchronize CUDA at their boundaries. The per-solve timers
measure host wall time; they are not CUDA-kernel timings.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys
import time
import traceback
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import kaiwu
kaiwu.__path__.insert(0, str(ROOT / "src/kaiwu"))
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_json(path, value):
    pending = path.with_suffix(".tmp")
    pending.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    pending.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "cuda"], required=True)
    parser.add_argument("--kind", choices=["ep", "bp"], default="ep")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--limit-train", type=int)
    parser.add_argument("--limit-eval", type=int)
    parser.add_argument("--validation-only", action="store_true")
    parser.add_argument("--conv", action="store_true")
    parser.add_argument("--channels", type=int, default=4)
    parser.add_argument("--sa-iterations", type=int, default=50)
    parser.add_argument("--evaluate", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device(args.device)

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    # Initialize CUDA before phase timers, and report setup separately.
    setup = time.perf_counter()
    if device.type == "cuda":
        (torch.ones((32, 32), device=device) @ torch.ones((32, 32), device=device)).sum().item()
        sync()
        torch.cuda.reset_peak_memory_stats()
    entry = load_module(ROOT / "example/ep_mnist/train_mnist.py", "device_ep_entry")
    paths = [ROOT / "example/ep_mnist/train_mnist.py",
             ROOT / "src/kaiwu/torch_plugin/equilibrium_propagation.py", Path(__file__)]
    report = {
        "status": "running", "pid": os.getpid(),
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "platform": platform.platform(), "cpu": platform.processor(),
        "torch_threads": torch.get_num_threads(), "sampler_processes": 1,
        "execution": "CUDA model plus CPU Kaiwu SA" if args.device == "cuda" and args.kind == "ep" else args.device,
        "timing": "synchronized phase wall time including transfers; evaluation includes train and validation sets",
        "initialization_seconds": time.perf_counter() - setup,
        "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        "phases": [],
    }
    current = {}
    last_progress = [0.0]

    def progress(force=False):
        now = time.perf_counter()
        if not args.output.exists() or (not force and now - last_progress[0] < 30):
            return
        last_progress[0] = now
        row = {**current, "pid": os.getpid(), "updated_at_utc": datetime.now(timezone.utc).isoformat()}
        if "started" in row:
            row["elapsed_seconds"] = now - row.pop("started")
        write_json(args.output / "progress.json", row)
        write_json(args.output / "device_report.json", report)
        print(json.dumps({"progress": row}), flush=True)

    class DeviceLoader:
        def __init__(self, loader):
            self.loader, self.dataset = loader, loader.dataset

        def __iter__(self):
            for inputs, labels in self.loader:
                yield inputs.to(device), labels.to(device)
                current["processed"] += len(inputs)
                progress()

    def phase(name, loader, callback):
        sync()
        current.clear()
        current.update(phase=name, processed=0, total=len(loader.dataset), started=time.perf_counter(),
                       sampler_calls=0, sampler_seconds=0.0, ising_seconds=0.0)
        progress(True)
        result = callback(DeviceLoader(loader))
        sync()
        row = {**current, "elapsed_seconds": time.perf_counter() - current["started"]}
        row.pop("started")
        report["phases"].append(row)
        progress(True)
        return result

    original_build = entry._build_model
    def build(config):
        # Initialize on CPU with the original seed before moving to CUDA.
        model = original_build(config).to(device)
        original_matrix = model.ising_matrix
        def matrix(*a, **kw):
            started = time.perf_counter()
            result = original_matrix(*a, **kw)
            current["ising_seconds"] = current.get("ising_seconds", 0.0) + time.perf_counter() - started
            return result
        model.ising_matrix = matrix
        return model
    entry._build_model = build
    original_sampler = entry.SimulatedAnnealingOptimizer
    class TimedSampler:
        def __init__(self, **settings):
            self.inner = original_sampler(**settings)
        def solve(self, matrix):
            started = time.perf_counter()
            result = self.inner.solve(matrix)
            current["sampler_calls"] = current.get("sampler_calls", 0) + 1
            current["sampler_seconds"] = current.get("sampler_seconds", 0.0) + time.perf_counter() - started
            return result
    entry.SimulatedAnnealingOptimizer = TimedSampler
    original_train, original_accuracy = entry._train_epoch, entry._accuracy
    entry._train_epoch = lambda m, s, o, loader, beta: phase(
        "train", loader, lambda batches: original_train(m, s, o, batches, beta))
    entry._accuracy = lambda m, s, loader: phase(
        "accuracy", loader, lambda batches: original_accuracy(m, s, batches))
    started = time.perf_counter()
    try:
        if args.kind == "ep":
            sys.argv = ["train_mnist", "--data-dir", str(ROOT.parents[1] / "data"),
                        "--output-dir", str(args.output), "--epochs", str(args.epochs),
                        "--sa-iterations", str(args.sa_iterations), "--channels", str(args.channels),
                        "--seed", "0", "--split-seed", "0"]
            for name in ["limit_train", "limit_eval", "evaluate"]:
                value = getattr(args, name)
                if value is not None:
                    sys.argv.extend(["--" + name.replace("_", "-"), str(value)])
            for name in ["validation_only", "conv"]:
                if getattr(args, name):
                    sys.argv.append("--" + name.replace("_", "-"))
            entry.main()
        else:
            args.output.mkdir(parents=True, exist_ok=False)
            config = argparse.Namespace(n_val=5000, split_seed=0, batch_size=20,
                                        limit_train=args.limit_train, limit_eval=args.limit_eval)
            train, val, test = entry._loaders(config, ROOT.parents[1] / "data", not args.validation_only)
            torch.manual_seed(0)
            model = nn.Sequential(nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 10)).to(device)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
            order = np.arange(len(train.dataset))
            rng = np.random.default_rng(0)
            batches = DataLoader(train.dataset, batch_size=20, sampler=order)
            summary = {"model": "784-128-10 ReLU MLP", "device": args.device, "epochs": args.epochs,
                       "seed": 0, "split_seed": 0, "batch_size": 20, "lr": 0.05,
                       "n_train": len(train.dataset), "n_val": len(val.dataset),
                       "n_test": 0 if test is None else len(test.dataset), "history": []}
            def fit(loader):
                total = 0.0
                for x, y in loader:
                    optimizer.zero_grad()
                    loss = nn.functional.cross_entropy(model(x), y)
                    loss.backward()
                    optimizer.step()
                    total += float(loss.detach()) * len(x)
                return total / len(loader.dataset)
            def accuracy(loader):
                with torch.no_grad():
                    return sum(int((model(x).argmax(1) == y).sum()) for x, y in loader) / len(loader.dataset)
            for epoch in range(1, args.epochs + 1):
                rng.shuffle(order)
                loss = phase("train", batches, fit)
                train_seconds = report["phases"][-1]["elapsed_seconds"]
                sync()
                eval_start = time.perf_counter()
                train_acc = phase("train_accuracy", train, accuracy)
                val_acc = phase("val_accuracy", val, accuracy)
                sync()
                row = {"epoch": epoch, "train_seconds": train_seconds, "eval_seconds": time.perf_counter()-eval_start,
                       "train_loss": loss, "train_accuracy": train_acc, "val_accuracy": val_acc,
                       "train_error": 1-train_acc, "val_error": 1-val_acc}
                summary["history"].append(row)
                write_json(args.output / "summary.json", summary)
                print(json.dumps(row), flush=True)
            if test is not None:
                summary["test_accuracy"] = phase("test_accuracy", test, accuracy)
                summary["test_seconds"] = report["phases"][-1]["elapsed_seconds"]
            write_json(args.output / "summary.json", summary)
            torch.save(model.state_dict(), args.output / "checkpoint.pt")
        report["status"] = "completed"
    except BaseException:
        report["status"] = "failed"
        report["error"] = traceback.format_exc()
        raise
    finally:
        sync()
        report["run_wall_seconds"] = time.perf_counter() - started
        report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        if device.type == "cuda":
            report["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated()
        if args.output.exists():
            write_json(args.output / "device_report.json", report)
    print("DEVICE RUN COMPLETED", flush=True)


if __name__ == "__main__":
    main()
