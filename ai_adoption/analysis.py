"""Pure-pandas analysis functions behind each dashboard view.

Each function maps to one of the use cases stated on the Kaggle page:

* AI trend analysis           -> :func:`task_summary`
* Framework popularity        -> :func:`framework_summary`, :func:`task_framework_matrix`
* Engagement vs. downloads    -> :func:`engagement_correlation`, :func:`hidden_gems`
* Top developers & tools      -> :func:`developer_summary`, :func:`top_models`
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config as C


# --------------------------------------------------------------------------- #
# Filtering
# --------------------------------------------------------------------------- #
def apply_filters(
    df: pd.DataFrame,
    tasks: Iterable[str] | None = None,
    frameworks: Iterable[str] | None = None,
    developers: Iterable[str] | None = None,
    min_downloads: int = 0,
    search: str = "",
) -> pd.DataFrame:
    mask = pd.Series(True, index=df.index)
    if tasks:
        mask &= df[C.COL_TASK].isin(list(tasks))
    if frameworks:
        mask &= df[C.COL_FRAMEWORK].isin(list(frameworks))
    if developers:
        mask &= df[C.COL_DEV].isin(list(developers))
    if min_downloads:
        mask &= df[C.COL_DOWNLOADS] >= min_downloads
    if search:
        needle = search.strip().lower()
        mask &= (
            df[C.COL_NAME].str.lower().str.contains(needle, regex=False)
            | df[C.COL_DEV].str.lower().str.contains(needle, regex=False)
        )
    return df[mask]


# --------------------------------------------------------------------------- #
# KPIs
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class KPIs:
    n_models: int
    n_developers: int
    n_tasks: int
    n_frameworks: int
    total_downloads: int
    total_likes: int
    median_downloads: float
    likes_per_1k: float
    top1_share: float  # share of downloads held by the single top model
    top10_share: float  # share of downloads held by the top 10 models


def compute_kpis(df: pd.DataFrame) -> KPIs:
    total_dl = int(df[C.COL_DOWNLOADS].sum())
    sorted_dl = df[C.COL_DOWNLOADS].sort_values(ascending=False)
    share = (lambda n: float(sorted_dl.head(n).sum() / total_dl)) if total_dl else (lambda n: 0.0)
    total_likes = int(df[C.COL_LIKES].sum())
    return KPIs(
        n_models=len(df),
        n_developers=df[C.COL_DEV].nunique(),
        n_tasks=df[C.COL_TASK].nunique(),
        n_frameworks=df[C.COL_FRAMEWORK].nunique(),
        total_downloads=total_dl,
        total_likes=total_likes,
        median_downloads=float(df[C.COL_DOWNLOADS].median()) if len(df) else 0.0,
        likes_per_1k=(total_likes / total_dl * 1_000) if total_dl else 0.0,
        top1_share=share(1),
        top10_share=share(10),
    )


# --------------------------------------------------------------------------- #
# Group summaries
# --------------------------------------------------------------------------- #
def _group_summary(df: pd.DataFrame, by: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            columns=[by, "models", "downloads", "likes", "median_downloads",
                     "likes_per_1k", "download_share", "model_share"]
        )
    g = (
        df.groupby(by, observed=True)
        .agg(
            models=(C.COL_NAME, "count"),
            downloads=(C.COL_DOWNLOADS, "sum"),
            likes=(C.COL_LIKES, "sum"),
            median_downloads=(C.COL_DOWNLOADS, "median"),
        )
        .reset_index()
    )
    g["likes_per_1k"] = np.where(g["downloads"] > 0, g["likes"] / g["downloads"] * 1_000, 0.0)
    g["download_share"] = g["downloads"] / max(g["downloads"].sum(), 1)
    g["model_share"] = g["models"] / max(g["models"].sum(), 1)
    return g.sort_values("downloads", ascending=False, ignore_index=True)


def task_summary(df: pd.DataFrame) -> pd.DataFrame:
    return _group_summary(df, C.COL_TASK)


def framework_summary(df: pd.DataFrame) -> pd.DataFrame:
    return _group_summary(df, C.COL_FRAMEWORK)


def developer_summary(df: pd.DataFrame) -> pd.DataFrame:
    return _group_summary(df, C.COL_DEV)


def top_n_with_other(series: pd.Series, n: int, other: str = "Other") -> pd.Series:
    """Keep the ``n`` most frequent labels and fold the rest into ``other``."""
    keep = series.value_counts().head(n).index
    return series.where(series.isin(keep), other)


def task_framework_matrix(
    df: pd.DataFrame, top_tasks: int = 10, top_frameworks: int = 8, value: str = "models"
) -> pd.DataFrame:
    """Pivot of task x framework (model count or downloads), top-N on each axis."""
    if df.empty:
        return pd.DataFrame()
    tasks = task_summary(df).head(top_tasks)[C.COL_TASK]
    fws = framework_summary(df).head(top_frameworks)[C.COL_FRAMEWORK]
    sub = df[df[C.COL_TASK].isin(tasks) & df[C.COL_FRAMEWORK].isin(fws)]
    agg = ("count" if value == "models" else "sum")
    col = C.COL_NAME if value == "models" else C.COL_DOWNLOADS
    mat = sub.pivot_table(index=C.COL_TASK, columns=C.COL_FRAMEWORK, values=col,
                          aggfunc=agg, fill_value=0)
    return mat.reindex(index=list(tasks), columns=list(fws), fill_value=0)


# --------------------------------------------------------------------------- #
# Rankings
# --------------------------------------------------------------------------- #
def top_models(df: pd.DataFrame, n: int = 15, by: str = C.COL_DOWNLOADS) -> pd.DataFrame:
    return df.nlargest(n, by)


# --------------------------------------------------------------------------- #
# Engagement vs downloads
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Correlation:
    pearson_log: float
    spearman: float
    slope: float       # elasticity of likes w.r.t. downloads (log-log)
    intercept: float
    n: int

    @property
    def strength(self) -> str:
        r = abs(self.spearman)
        if np.isnan(r):
            return "n/a"
        if r >= 0.7:
            return "strong"
        if r >= 0.4:
            return "moderate"
        if r >= 0.2:
            return "weak"
        return "negligible"


def engagement_correlation(df: pd.DataFrame) -> Correlation:
    sub = df[(df[C.COL_DOWNLOADS] > 0) & (df[C.COL_LIKES] > 0)]
    if len(sub) < 3:
        return Correlation(np.nan, np.nan, np.nan, np.nan, len(sub))
    x = np.log10(sub[C.COL_DOWNLOADS].to_numpy(dtype=float))
    y = np.log10(sub[C.COL_LIKES].to_numpy(dtype=float))
    pearson = float(np.corrcoef(x, y)[0, 1])
    spearman = float(sub[C.COL_DOWNLOADS].rank().corr(sub[C.COL_LIKES].rank()))
    slope, intercept = np.polyfit(x, y, 1)
    return Correlation(pearson, spearman, float(slope), float(intercept), len(sub))


def add_engagement_residual(df: pd.DataFrame, corr: Correlation | None = None) -> pd.DataFrame:
    """Add ``Engagement_Residual``: how many log10 units a model's likes sit
    above (+) or below (−) what its download count predicts."""
    corr = corr or engagement_correlation(df)
    out = df.copy()
    if np.isnan(corr.slope):
        out["Engagement_Residual"] = 0.0
        return out
    with np.errstate(divide="ignore"):
        x = np.log10(out[C.COL_DOWNLOADS].clip(lower=1).astype(float))
        y = np.log10(out[C.COL_LIKES].clip(lower=1).astype(float))
    out["Engagement_Residual"] = (y - (corr.slope * x + corr.intercept)).round(3)
    return out


def hidden_gems(df: pd.DataFrame, n: int = 15, min_downloads: int = 1_000) -> pd.DataFrame:
    """Models liked far more than their download count predicts."""
    scored = add_engagement_residual(df)
    scored = scored[scored[C.COL_DOWNLOADS] >= min_downloads]
    return scored.nlargest(n, "Engagement_Residual")


def under_engaged(df: pd.DataFrame, n: int = 15, min_downloads: int = 100_000) -> pd.DataFrame:
    """Heavily downloaded models that attract few likes (likely pipeline deps)."""
    scored = add_engagement_residual(df)
    scored = scored[scored[C.COL_DOWNLOADS] >= min_downloads]
    return scored.nsmallest(n, "Engagement_Residual")


def concentration_curve(df: pd.DataFrame) -> pd.DataFrame:
    """Cumulative share of downloads vs. share of models (a Lorenz-style curve)."""
    dl = df[C.COL_DOWNLOADS].sort_values(ascending=False).to_numpy(dtype=float)
    if dl.size == 0 or dl.sum() == 0:
        return pd.DataFrame({"model_share": [], "download_share": []})
    return pd.DataFrame({
        "model_share": np.arange(1, dl.size + 1) / dl.size,
        "download_share": np.cumsum(dl) / dl.sum(),
    })


def gini(values: pd.Series) -> float:
    v = np.sort(values.to_numpy(dtype=float))
    if v.size == 0 or v.sum() == 0:
        return 0.0
    n = v.size
    return float((2 * np.arange(1, n + 1) - n - 1).dot(v) / (n * v.sum()))
