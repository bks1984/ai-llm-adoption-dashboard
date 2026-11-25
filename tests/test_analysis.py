import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from ai_adoption import analysis as A
from ai_adoption import charts as CH
from ai_adoption import config as C


def test_filters(clean_df):
    assert len(A.apply_filters(clean_df, tasks=["text-generation"])) == 2
    assert len(A.apply_filters(clean_df, frameworks=["gguf"])) == 1
    assert len(A.apply_filters(clean_df, min_downloads=400_000)) == 2
    assert len(A.apply_filters(clean_df, search="WHISPER")) == 1
    assert len(A.apply_filters(clean_df, search="org")) == 2  # matches developer too
    assert A.apply_filters(clean_df, tasks=["nope"]).empty


def test_kpis(clean_df):
    k = A.compute_kpis(clean_df)
    assert k.n_models == 5
    assert k.total_downloads == 1_820_000
    assert k.top1_share == pytest.approx(1_000_000 / 1_820_000)
    assert k.top10_share == pytest.approx(1.0)


def test_kpis_empty(clean_df):
    k = A.compute_kpis(clean_df.iloc[0:0])
    assert k.n_models == 0 and k.likes_per_1k == 0 and k.top10_share == 0


def test_group_summary_shares_sum_to_one(sample_df):
    for fn in (A.task_summary, A.framework_summary, A.developer_summary):
        s = fn(sample_df)
        assert s["download_share"].sum() == pytest.approx(1.0)
        assert s["models"].sum() == len(sample_df)
        assert s["downloads"].is_monotonic_decreasing


def test_matrix_shape(sample_df):
    m = A.task_framework_matrix(sample_df, top_tasks=5, top_frameworks=4)
    assert m.shape == (5, 4)
    assert (m.to_numpy() >= 0).all()


def test_correlation_on_perfect_power_law():
    dl = np.logspace(2, 7, 50)
    df = pd.DataFrame({C.COL_DOWNLOADS: dl.astype(int), C.COL_LIKES: (2 * dl ** 0.7).round().astype(int)})
    corr = A.engagement_correlation(df)
    assert corr.spearman > 0.99
    assert corr.slope == pytest.approx(0.7, abs=0.05)
    assert corr.strength == "strong"


def test_correlation_too_few_points(clean_df):
    corr = A.engagement_correlation(clean_df.head(2))
    assert np.isnan(corr.spearman) and corr.strength == "n/a"


def test_hidden_gems_and_workhorses(sample_df):
    gems = A.hidden_gems(sample_df, n=5)
    work = A.under_engaged(sample_df, n=5)
    assert len(gems) == 5 and (gems[C.COL_DOWNLOADS] >= 1_000).all()
    assert gems["Engagement_Residual"].min() > work["Engagement_Residual"].max()


def test_top_n_with_other():
    s = pd.Series(list("aaabbc"))
    assert set(A.top_n_with_other(s, 1)) == {"a", "Other"}


def test_gini_bounds():
    assert A.gini(pd.Series([5, 5, 5, 5])) == pytest.approx(0)
    assert A.gini(pd.Series([0, 0, 0, 100])) == pytest.approx(0.75)
    assert A.gini(pd.Series([], dtype=float)) == 0


def test_concentration_curve_ends_at_one(sample_df):
    curve = A.concentration_curve(sample_df)
    assert curve["download_share"].iloc[-1] == pytest.approx(1.0)
    assert curve["download_share"].is_monotonic_increasing


@pytest.mark.parametrize("builder", [
    lambda d: CH.ranked_bar(d, C.COL_NAME, C.COL_DOWNLOADS, "Downloads"),
    CH.download_tier_bar,
    CH.concentration_chart,
    lambda d: CH.task_treemap(A.task_summary(d)),
    lambda d: CH.task_share_compare(A.task_summary(d)),
    CH.framework_box,
    CH.task_framework_heatmap,
    lambda d: CH.engagement_scatter(d, A.engagement_correlation(d)),
    lambda d: CH.residual_by_group(d, C.COL_FRAMEWORK),
    CH.developer_sunburst,
])
def test_charts_build_with_data_and_empty(sample_df, builder):
    assert isinstance(builder(sample_df), go.Figure)
    assert isinstance(builder(sample_df.iloc[0:0]), go.Figure)


def test_scatter_caps_colours_at_three_plus_other(sample_df):
    fig = CH.engagement_scatter(sample_df, A.engagement_correlation(sample_df))
    point_traces = [t for t in fig.data if t.mode == "markers"]
    assert len(point_traces) <= 4


def test_fmt_num():
    assert CH.fmt_num(1_500) == "1.5K"
    assert CH.fmt_num(2_300_000) == "2.3M"
    assert CH.fmt_num(42) == "42"
