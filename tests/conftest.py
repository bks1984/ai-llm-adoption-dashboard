import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ai_adoption import config as C  # noqa: E402


@pytest.fixture
def raw_df() -> pd.DataFrame:
    """Small hand-built frame with the messy cases the cleaner must handle."""
    return pd.DataFrame({
        "Record_ID": [1, 2, 3, 4, 5, 6],
        "Model_Or_Tool_Name": ["org/gpt2", "org/bert", "x/whisper", "y/llama-GGUF",
                               "org/gpt2", " z/tiny "],
        "Developer_Org": ["org", "org", "x", "y", "org", None],
        "Primary_Task_Category": ["Text-Generation", "fill-mask",
                                  "automatic-speech-recognition", "text-generation",
                                  "text-generation", "text-classification"],
        "Framework_Library": ["transformers", "transformers", "pytorch", "gguf",
                              "transformers", "onnx"],
        "Total_Downloads": ["1,000,000", 500_000, 20_000, 300_000, 1_000_000, "n/a"],
        "Community_Likes": [2_000, 400, 150, 30, 2_000, 5],
        "Adoption_Engagement_Score": [2.0, 0.8, None, 0.1, 2.0, 0.0],
        "Data_Source": ["HuggingFace Live Hub API"] * 6,
    })


@pytest.fixture
def clean_df(raw_df):
    from ai_adoption.data_loader import clean
    return clean(raw_df)


@pytest.fixture
def sample_df():
    from ai_adoption.data_loader import clean
    from scripts.generate_sample_data import generate
    return clean(generate(rows=400, seed=7))


__all__ = ["C"]
