import pandas as pd
import pytest

from ai_adoption import config as C
from ai_adoption.data_loader import SchemaError, clean, find_source_file, load_dataset


def test_clean_parses_numbers_and_handles_bad_values(clean_df):
    assert clean_df[C.COL_DOWNLOADS].dtype == "int64"
    assert clean_df.loc[clean_df[C.COL_NAME] == "org/gpt2", C.COL_DOWNLOADS].item() == 1_000_000
    # "n/a" downloads -> 0, never NaN
    assert clean_df.loc[clean_df[C.COL_NAME] == "z/tiny", C.COL_DOWNLOADS].item() == 0


def test_clean_drops_duplicates_and_normalises_text(clean_df):
    assert len(clean_df) == 5
    assert "z/tiny" in set(clean_df[C.COL_NAME])          # whitespace stripped
    assert clean_df[C.COL_TASK].str.islower().all()        # task lower-cased
    assert (clean_df[C.COL_DEV] == C.UNKNOWN).sum() == 1   # None -> unknown


def test_derived_columns(clean_df):
    row = clean_df[clean_df[C.COL_NAME] == "org/gpt2"].iloc[0]
    assert row[C.COL_LIKES_PER_1K] == pytest.approx(2.0)
    assert row[C.COL_DOWNLOAD_TIER] == "1M–10M"
    zero = clean_df[clean_df[C.COL_NAME] == "z/tiny"].iloc[0]
    assert zero[C.COL_LIKES_PER_1K] == 0  # no division by zero


def test_missing_score_falls_back_to_likes_per_1k(clean_df):
    row = clean_df[clean_df[C.COL_NAME] == "x/whisper"].iloc[0]
    assert row[C.COL_SCORE] == pytest.approx(row[C.COL_LIKES_PER_1K])


def test_score_column_optional(raw_df):
    out = clean(raw_df.drop(columns=[C.COL_SCORE, C.COL_SOURCE, C.COL_ID]))
    assert {C.COL_SCORE, C.COL_SOURCE, C.COL_ID} <= set(out.columns)


def test_headers_with_spaces_are_normalised(raw_df):
    out = clean(raw_df.rename(columns=lambda c: c.replace("_", " ")))
    assert C.COL_DOWNLOADS in out.columns


def test_missing_required_column_raises(raw_df):
    with pytest.raises(SchemaError, match="Total_Downloads"):
        clean(raw_df.drop(columns=[C.COL_DOWNLOADS]))


def test_find_source_prefers_explicit_then_raw(tmp_path, monkeypatch, raw_df):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    monkeypatch.setattr(C, "RAW_DIR", raw_dir)
    path, is_sample = find_source_file()
    assert is_sample  # falls back to bundled sample

    real = raw_dir / "real.csv"
    raw_df.to_csv(real, index=False)
    path, is_sample = find_source_file()
    assert path == real and not is_sample

    data = load_dataset()
    assert not data.is_sample and len(data.df) == 5


def test_bundled_sample_loads():
    df = clean(pd.read_csv(C.SAMPLE_FILE))
    assert len(df) == 1000
    assert set(C.ALL_COLUMNS) <= set(df.columns)
