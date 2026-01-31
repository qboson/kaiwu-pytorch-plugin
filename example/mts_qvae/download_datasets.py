import argparse
import hashlib
import os
import sys
import urllib.request
import zipfile
from pathlib import Path

from typing import Optional

ZENODO_DATASETS_URL = (
    "https://zenodo.org/api/records/14590156/files/Datasets.zip/content"
)
ZENODO_DATASETS_MD5 = "e24a1dd48aef7bd5bb416d1e1fa9d257"


def find_repo_root(start: Path) -> Path:
    for p in [start] + list(start.parents):
        if (p / "src" / "kaiwu").exists():
            return p
    return start.parent


def md5sum(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path) -> None:
    """Download a large file robustly.

    Uses `requests` with streaming + resume when available. Falls back to urllib.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)

    try:
        import requests  # type: ignore
    except Exception:
        requests = None

    tmp = dest.with_suffix(dest.suffix + ".part")

    if requests is None:
        # Fallback: urllib without resume (kept for environments without requests)
        def report(block_num: int, block_size: int, total_size: int):
            if total_size <= 0:
                return
            downloaded = block_num * block_size
            pct = min(100.0, downloaded * 100.0 / total_size)
            sys.stdout.write(
                f"\rDownloading: {pct:6.2f}% ({downloaded/1e6:.1f}/{total_size/1e6:.1f} MB)"
            )
            sys.stdout.flush()

        if tmp.exists():
            tmp.unlink()
        urllib.request.urlretrieve(url, tmp, reporthook=report)
        sys.stdout.write("\n")
        sys.stdout.flush()
        tmp.replace(dest)
        return

    # requests path: resume + retries
    headers = {
        "User-Agent": "kaiwu-mts-qvae-downloader/1.0 (+https://github.com/qboson/kaiwu-pytorch-plugin)",
    }

    def _stream_once(resume_from: int) -> Optional[int]:
        req_headers = dict(headers)
        if resume_from > 0:
            req_headers["Range"] = f"bytes={resume_from}-"

        with requests.get(url, headers=req_headers, stream=True, timeout=(15, 120), allow_redirects=True) as r:
            r.raise_for_status()
            content_len = r.headers.get("Content-Length")
            total = None
            if content_len is not None:
                total = int(content_len)
                if resume_from > 0 and r.status_code == 206:
                    total = total + resume_from

            mode = "ab" if resume_from > 0 else "wb"
            downloaded = resume_from
            with tmp.open(mode) as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if not chunk:
                        continue
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        pct = min(100.0, downloaded * 100.0 / total)
                        sys.stdout.write(
                            f"\rDownloading: {pct:6.2f}% ({downloaded/1e6:.1f}/{total/1e6:.1f} MB)"
                        )
                        sys.stdout.flush()

            sys.stdout.write("\n")
            sys.stdout.flush()
            return total

    max_attempts = 8
    attempt = 0
    while True:
        attempt += 1
        resume_from = tmp.stat().st_size if tmp.exists() else 0
        try:
            total = _stream_once(resume_from)

            # If server provided total size, ensure we got it
            if total is not None:
                got = tmp.stat().st_size
                if got < total:
                    raise IOError(f"retrieval incomplete: got only {got} out of {total} bytes")

            tmp.replace(dest)
            return
        except Exception as e:
            if attempt >= max_attempts:
                raise
            print(f"Download interrupted ({e}). Retrying ({attempt}/{max_attempts})...")



def extract_zip(zip_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_dir)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download MTS-VAE datasets (Zenodo 14590156) with md5 verification, then unzip."  # noqa: E501
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("example/mts_qvae/data/zenodo"),
        help="Output directory to place Datasets.zip and extracted folders.",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip download if you already have Datasets.zip.",
    )
    args = parser.parse_args()

    repo_root = find_repo_root(Path(__file__).resolve())
    out_dir: Path = args.out_dir
    if not out_dir.is_absolute():
        out_dir = (repo_root / out_dir).resolve()
    zip_path = out_dir / "Datasets.zip"

    if not args.skip_download:
        print(f"Downloading Zenodo dataset to: {zip_path}")
        download(ZENODO_DATASETS_URL, zip_path)

    if not zip_path.exists():
        raise FileNotFoundError(f"Missing {zip_path}. Use --skip-download only if file exists.")

    print("Verifying md5...")
    got = md5sum(zip_path)
    if got.lower() != ZENODO_DATASETS_MD5:
        raise RuntimeError(
            f"MD5 mismatch for {zip_path}: expected {ZENODO_DATASETS_MD5}, got {got}"
        )
    print("MD5 OK")

    marker = out_dir / ".extracted"
    if marker.exists():
        print(f"Already extracted (marker exists): {marker}")
        return 0

    print(f"Extracting to: {out_dir}")
    extract_zip(zip_path, out_dir)

    # lightweight sanity check
    expected = [
        out_dir / "MTS" / "data" / "tv_sim_split_train.pkl",
        out_dir / "Datasets" / "MTS" / "data" / "tv_sim_split_train.pkl",
    ]
    if not any(p.exists() for p in expected):
        found = list(out_dir.glob("**/tv_sim_split_train.pkl"))
        if found:
            print(f"Sanity check OK: found {found[0]}")
        else:
            print(
                "Warning: expected file not found. The zip structure may have changed; "
                "inspect the extracted folders under: "
                f"{out_dir}"
            )

    marker.write_text("ok", encoding="utf-8")
    print("Done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
