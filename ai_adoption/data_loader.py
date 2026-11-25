"""Load, validate and clean the AI & LLM tools adoption dataset.

Resolution order for the source file:
1. An explicit path passed by the caller.
2. The first ``*.csv`` found in ``data/raw/`` (the real Kaggle download).
3. The bundled synthetic sample in ``data/sample/``.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C


class SchemaError(ValueError):
    """Raised when the input file lacks the columns the dashboard needs."""


@dataclass(frozen=True)
class LoadedData:
    df: pd.DataFrame
    source_path: Path
    is_sample: bool


def find_source_file(explicit: str | Path | None = None) -> tuple[Path, bool]:
    """Return ``(path, is_sample)`` for the CSV the app should read."""
    if explicit:
        path = Path(explicit)
        if not path.exists():
            raise FileNotFoundError(path)
        return path, False

    raw_files = sorted(C.RAW_DIR.glob("*.csv"))
    if raw_files:
        return raw_files[0], False

    if C.SAMPLE_FILE.exists():
        return C.SAMPLE_FILE, True

    raise FileNotFoundError(
        "No dataset found. Run `python scripts/download_data.py` to fetch the "
        "Kaggle data, or `python scripts/generate_sample_data.py` for a sample."
    )


def _normalise_header(name: str) -> str:
    return str(name).strip().replace(" ", "_")


def validate_schema(df: pd.DataFrame) -> None:
    missing = [c for c in C.REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise SchemaError(
            f"Dataset is missing required column(s): {', '.join(missing)}. "
            f"Found: {', '.join(map(str, df.columns))}"
        )


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Type-coerce, de-duplicate and enrich the raw frame."""
    df = df.rename(columns=_normalise_header).copy()
    validate_schema(df)

    # --- numeric columns ---------------------------------------------------
    for col in (C.COL_DOWNLOADS, C.COL_LIKES):
        df[col] = (
            pd.to_numeric(
                df[col].astype(str).str.replace(r"[,\s]", "", regex=True),
                errors="coerce",
            )
            .fillna(0)
            .clip(lower=0)
            .astype("int64")
        )

    # --- categorical columns ----------------------------------------------
    for col in (C.COL_NAME, C.COL_DEV, C.COL_TASK, C.COL_FRAMEWORK):
        df[col] = (
            df[col]
            .astype("string")
            .str.strip()
            .replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
            .fillna(C.UNKNOWN)
        )
    df[C.COL_TASK] = df[C.COL_TASK].str.lower()

    # The published Kaggle file has Developer_Org = "Community/Independent" on
    # every row. When the column carries no information, recover the publisher
    # from the "org/model" repository id instead.
    if df[C.COL_DEV].nunique() <= 1 and df[C.COL_NAME].str.contains("/", regex=False).mean() > 0.5:
        df[C.COL_DEV] = df[C.COL_NAME].str.split("/", n=1).str[0].where(
            df[C.COL_NAME].str.contains("/", regex=False), C.UNKNOWN
        )

    if C.COL_SOURCE not in df.columns:
        df[C.COL_SOURCE] = C.UNKNOWN

    # --- de-duplicate & IDs -----------------------------------------------
    df = df.drop_duplicates(subset=[C.COL_NAME, C.COL_DEV]).reset_index(drop=True)
    if C.COL_ID not in df.columns or df[C.COL_ID].isna().any():
        df[C.COL_ID] = np.arange(1, len(df) + 1)
    df[C.COL_ID] = pd.to_numeric(df[C.COL_ID], errors="coerce").astype("int64")

    # --- derived metrics ---------------------------------------------------
    downloads = df[C.COL_DOWNLOADS].replace(0, np.nan)
    df[C.COL_LIKES_PER_1K] = (df[C.COL_LIKES] / downloads * 1_000).fillna(0).round(3)

    # The Kaggle page does not publish the score formula. Keep the supplied
    # score when present; otherwise fall back to likes per 1K downloads.
    if C.COL_SCORE in df.columns:
        df[C.COL_SCORE] = pd.to_numeric(df[C.COL_SCORE], errors="coerce")
        df[C.COL_SCORE] = df[C.COL_SCORE].fillna(df[C.COL_LIKES_PER_1K])
    else:
        df[C.COL_SCORE] = df[C.COL_LIKES_PER_1K]

    df[C.COL_DOWNLOAD_TIER] = pd.cut(
        df[C.COL_DOWNLOADS],
        bins=C.DOWNLOAD_TIER_BINS,
        labels=C.DOWNLOAD_TIER_LABELS,
        right=False,
    )

    ordered = [c for c in C.ALL_COLUMNS if c in df.columns]
    extras = [c for c in df.columns if c not in ordered]
    return df[ordered + extras]


def load_dataset(path: str | Path | None = None) -> LoadedData:
    source, is_sample = find_source_file(path)
    raw = pd.read_csv(source)
    return LoadedData(df=clean(raw), source_path=source, is_sample=is_sample)


def save_processed(df: pd.DataFrame, path: Path = C.PROCESSED_FILE) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path
