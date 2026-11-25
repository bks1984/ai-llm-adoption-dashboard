"""Project-wide configuration: paths, schema, and the chart palette."""
from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
SAMPLE_DIR = DATA_DIR / "sample"
PROCESSED_DIR = DATA_DIR / "processed"

SAMPLE_FILE = SAMPLE_DIR / "ai_llm_tools_sample.csv"
PROCESSED_FILE = PROCESSED_DIR / "ai_llm_tools_clean.csv"

KAGGLE_DATASET = "muhammadahmeddata/global-ai-and-llm-tools-adoption-dataset-2026"
KAGGLE_URL = f"https://www.kaggle.com/datasets/{KAGGLE_DATASET}"

# --------------------------------------------------------------------------- #
# Schema (as documented on the Kaggle dataset page)
# --------------------------------------------------------------------------- #
COL_ID = "Record_ID"
COL_NAME = "Model_Or_Tool_Name"
COL_DEV = "Developer_Org"
COL_TASK = "Primary_Task_Category"
COL_FRAMEWORK = "Framework_Library"
COL_DOWNLOADS = "Total_Downloads"
COL_LIKES = "Community_Likes"
COL_SCORE = "Adoption_Engagement_Score"
COL_SOURCE = "Data_Source"

REQUIRED_COLUMNS = [
    COL_NAME,
    COL_DEV,
    COL_TASK,
    COL_FRAMEWORK,
    COL_DOWNLOADS,
    COL_LIKES,
]
ALL_COLUMNS = [
    COL_ID,
    COL_NAME,
    COL_DEV,
    COL_TASK,
    COL_FRAMEWORK,
    COL_DOWNLOADS,
    COL_LIKES,
    COL_SCORE,
    COL_SOURCE,
]
NUMERIC_COLUMNS = [COL_ID, COL_DOWNLOADS, COL_LIKES, COL_SCORE]
CATEGORICAL_COLUMNS = [COL_DEV, COL_TASK, COL_FRAMEWORK, COL_SOURCE]

# Derived columns created by the cleaning step
COL_LIKES_PER_1K = "Likes_per_1K_Downloads"
COL_DOWNLOAD_TIER = "Download_Tier"

DOWNLOAD_TIER_BINS = [0, 1_000, 10_000, 100_000, 1_000_000, 10_000_000, float("inf")]
DOWNLOAD_TIER_LABELS = ["<1K", "1K–10K", "10K–100K", "100K–1M", "1M–10M", "10M+"]

UNKNOWN = "unknown"

# --------------------------------------------------------------------------- #
# Chart palette (validated categorical order, light mode)
# --------------------------------------------------------------------------- #
SERIES = [
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#e87ba4",  # magenta
    "#008300",  # green
    "#6250d6",  # violet
    "#e34948",  # red
]
OTHER_GRAY = "#a3a29c"
PRIMARY = SERIES[0]
SEQUENTIAL_BLUE = ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"]
