"""Download the real dataset from Kaggle into ``data/raw/``.

Prerequisites
-------------
1. ``pip install kaggle``
2. Create an API token at https://www.kaggle.com/settings -> "Create New Token"
   and save ``kaggle.json`` to ``~/.kaggle/kaggle.json`` (Linux/macOS) or
   ``%USERPROFILE%\\.kaggle\\kaggle.json`` (Windows). Alternatively set the
   ``KAGGLE_USERNAME`` and ``KAGGLE_KEY`` environment variables.

Usage:
    python scripts/download_data.py

Manual alternative: download the zip from the dataset page, extract the CSV,
and drop it into ``data/raw/``. The dashboard picks up the first CSV it finds
there and falls back to the synthetic sample otherwise.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ai_adoption import config as C  # noqa: E402


def main() -> int:
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        print("The `kaggle` package is not installed. Run: pip install kaggle")
        return 1

    api = KaggleApi()
    try:
        api.authenticate()
    except Exception as exc:  # noqa: BLE001 - surface any auth problem plainly
        print(f"Kaggle authentication failed: {exc}")
        print("See the docstring at the top of this script for setup steps.")
        return 1

    C.RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {C.KAGGLE_DATASET} -> {C.RAW_DIR}")
    api.dataset_download_files(C.KAGGLE_DATASET, path=str(C.RAW_DIR), unzip=True, quiet=False)

    csvs = sorted(C.RAW_DIR.glob("*.csv"))
    if not csvs:
        print("Download finished but no CSV was found in data/raw/.")
        return 1
    for f in csvs:
        print(f"  ✓ {f.name} ({f.stat().st_size / 1024:.1f} KB)")
    print("Done. Restart the dashboard to load the real data.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
