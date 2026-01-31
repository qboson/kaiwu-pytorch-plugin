from __future__ import annotations

import argparse
import platform
import subprocess
import sys
from pathlib import Path


def find_repo_root(start: Path) -> Path:
    for p in [start] + list(start.parents):
        if (p / "src" / "kaiwu").exists():
            return p
    return start.parent


def run(cmd: list[str], cwd: Path) -> None:
    print("\n$ " + " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare Issue #26 submission: run smoke pipeline, run pytest, and write a small report. "
            "Does NOT download datasets or run heavy training by default."
        )
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run run_all.py smoke (requires dataset already present).",
    )
    parser.add_argument(
        "--pytest",
        action="store_true",
        help="Run pytest -q.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("example/mts_qvae/ISSUE_26_RUN_REPORT.md"),
        help="Where to write the run report (relative to repo root by default).",
    )
    args = parser.parse_args()

    repo_root = find_repo_root(Path(__file__).resolve())
    out = args.out
    if not out.is_absolute():
        out = (repo_root / out).resolve()

    # Collect environment info
    py = sys.executable
    py_ver = sys.version.replace("\n", " ")
    os_info = f"{platform.system()} {platform.release()}"

    try:
        import torch  # type: ignore

        torch_ver = getattr(torch, "__version__", "unknown")
        cuda = bool(torch.cuda.is_available())
    except Exception:
        torch_ver = "not-importable"
        cuda = False

    # Optional checks
    if args.pytest:
        run([py, "-m", "pytest", "-q"], cwd=repo_root)

    if args.smoke:
        run(
            [
                py,
                str(repo_root / "example" / "mts_qvae" / "run_all.py"),
                "--skip-download",
                "--vae-epochs",
                "1",
                "--qvae-epochs",
                "1",
                "--n-samples",
                "50",
                "--force",
            ],
            cwd=repo_root,
        )

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        "\n".join(
            [
                "# Issue #26 本地运行报告（自动生成）",
                "",
                f"- python: `{py}`",
                f"- python_version: `{py_ver}`",
                f"- os: `{os_info}`",
                f"- torch: `{torch_ver}`",
                f"- cuda_available: `{cuda}`",
                "",
                "说明：此报告仅记录运行环境与命令是否成功，不包含或暗示论文结论。",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"\nWrote report: {out}")
    print("\nNext: follow example/mts_qvae/ISSUE_26_SUBMISSION.md for git/PR steps.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
