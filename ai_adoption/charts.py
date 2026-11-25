"""Plotly figure builders. Every function returns a ``go.Figure``.

Conventions: one y-axis per chart, single-series charts use one hue, categorical
colour is capped at three series + gray "Other" on scatter plots, and every
mark has a hover tooltip.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from . import analysis as A
from . import config as C

FONT = "Inter, Segoe UI, system-ui, sans-serif"
INK = "#52514e"
GRID = "rgba(128,128,128,0.18)"


def _style(fig: go.Figure, height: int = 420, legend: bool = False) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=16, t=16, b=8),
        font=dict(family=FONT, size=13, color=INK),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, title=None),
        hoverlabel=dict(font=dict(family=FONT, size=12)),
        bargap=0.35,
    )
    fig.update_xaxes(gridcolor=GRID, zeroline=False, linecolor=GRID, title_font=dict(size=12))
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor=GRID, title_font=dict(size=12))
    return fig


def _empty(msg: str = "No data for the current filters") -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=msg, showarrow=False, font=dict(size=14, color=INK))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return _style(fig, height=240)


def fmt_num(n: float) -> str:
    n = float(n)
    for unit, div in (("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if abs(n) >= div:
            return f"{n / div:.1f}{unit}"
    return f"{n:,.0f}"


# --------------------------------------------------------------------------- #
# Generic ranked bar
# --------------------------------------------------------------------------- #
def ranked_bar(
    df: pd.DataFrame, label: str, value: str, title_x: str,
    n: int = 15, hover_extra: dict[str, str] | None = None, log_x: bool = False,
) -> go.Figure:
    if df.empty:
        return _empty()
    d = df.nlargest(n, value).iloc[::-1]
    custom_cols = list((hover_extra or {}).keys())
    hover = f"<b>%{{y}}</b><br>{title_x}: %{{x:,.0f}}"
    for i, (col, lab) in enumerate((hover_extra or {}).items()):
        fmt = ":.1%" if "share" in col else ":,.2f" if d[col].dtype.kind == "f" else ":,"
        hover += f"<br>{lab}: %{{customdata[{i}]{fmt}}}"
    fig = go.Figure(go.Bar(
        x=d[value], y=d[label].astype(str), orientation="h",
        marker=dict(color=C.PRIMARY, cornerradius=4),
        customdata=d[custom_cols].to_numpy() if custom_cols else None,
        hovertemplate=hover + "<extra></extra>",
        text=[fmt_num(v) for v in d[value]], textposition="outside",
        textfont=dict(size=11, color=INK), cliponaxis=False,
    ))
    fig.update_xaxes(title=title_x, type="log" if log_x else "linear")
    fig.update_yaxes(title=None, automargin=True)
    return _style(fig, height=max(300, 26 * len(d) + 80))


# --------------------------------------------------------------------------- #
# Overview
# --------------------------------------------------------------------------- #
def download_tier_bar(df: pd.DataFrame) -> go.Figure:
    if df.empty:
        return _empty()
    counts = (df[C.COL_DOWNLOAD_TIER].value_counts()
              .reindex(C.DOWNLOAD_TIER_LABELS, fill_value=0).reset_index())
    counts.columns = ["tier", "models"]
    # Drop empty tiers at either end (the real dataset starts at ~360K downloads)
    nz = counts.index[counts["models"] > 0]
    counts = counts.loc[nz.min():nz.max()]
    fig = go.Figure(go.Bar(
        x=counts["tier"], y=counts["models"],
        marker=dict(color=C.PRIMARY, cornerradius=4),
        hovertemplate="<b>%{x} downloads</b><br>%{y:,} models<extra></extra>",
    ))
    fig.update_xaxes(title="Download tier")
    fig.update_yaxes(title="Models")
    return _style(fig, height=340)


def concentration_chart(df: pd.DataFrame) -> go.Figure:
    curve = A.concentration_curve(df)
    if curve.empty:
        return _empty()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines", line=dict(color=C.OTHER_GRAY, width=1, dash="dot"),
        hoverinfo="skip", name="Equal share",
    ))
    fig.add_trace(go.Scatter(
        x=curve["model_share"], y=curve["download_share"], mode="lines",
        line=dict(color=C.PRIMARY, width=2), name="Actual",
        fill="tozeroy", fillcolor="rgba(42,120,214,0.10)",
        hovertemplate="Top %{x:.1%} of models<br>hold %{y:.1%} of downloads<extra></extra>",
    ))
    fig.update_xaxes(title="Share of models (ranked by downloads)", tickformat=".0%")
    fig.update_yaxes(title="Cumulative share of downloads", tickformat=".0%")
    fig.update_layout(hovermode="x")
    return _style(fig, height=340)


# --------------------------------------------------------------------------- #
# Tasks
# --------------------------------------------------------------------------- #
def task_treemap(tasks: pd.DataFrame) -> go.Figure:
    if tasks.empty:
        return _empty()
    fig = px.treemap(
        tasks, path=[px.Constant("All tasks"), C.COL_TASK], values="downloads",
        color="likes_per_1k", color_continuous_scale=C.SEQUENTIAL_BLUE,
        custom_data=["models", "download_share", "likes_per_1k"],
    )
    fig.update_traces(
        hovertemplate="<b>%{label}</b><br>Downloads: %{value:,.0f}"
                      "<br>Share: %{customdata[1]:.1%}<br>Models: %{customdata[0]}"
                      "<br>Likes / 1K downloads: %{customdata[2]:.2f}<extra></extra>",
        texttemplate="<b>%{label}</b><br>%{customdata[1]:.1%}",
        marker=dict(line=dict(width=2, color="white")),
        root_color="rgba(0,0,0,0)",
    )
    fig.update_layout(coloraxis_colorbar=dict(title="Likes /<br>1K DL", thickness=10))
    return _style(fig, height=460)


def task_share_compare(tasks: pd.DataFrame, n: int = 12) -> go.Figure:
    """Share of models vs share of downloads per task, as small multiples."""
    if tasks.empty:
        return _empty()
    d = tasks.head(n).iloc[::-1]
    from plotly.subplots import make_subplots
    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, horizontal_spacing=0.04,
                        subplot_titles=("Share of models", "Share of downloads"))
    for col, (metric, color) in enumerate((("model_share", C.SERIES[1]),
                                           ("download_share", C.SERIES[0])), start=1):
        fig.add_trace(go.Bar(
            x=d[metric], y=d[C.COL_TASK], orientation="h",
            marker=dict(color=color, cornerradius=4),
            hovertemplate="<b>%{y}</b><br>%{x:.1%}<extra></extra>",
            text=[f"{v:.0%}" for v in d[metric]], textposition="outside",
            textfont=dict(size=11, color=INK), cliponaxis=False,
        ), row=1, col=col)
        fig.update_xaxes(tickformat=".0%", row=1, col=col)
    fig.update_annotations(font=dict(size=12, color=INK))
    return _style(fig, height=max(320, 28 * len(d) + 80))


# --------------------------------------------------------------------------- #
# Frameworks
# --------------------------------------------------------------------------- #
def framework_box(df: pd.DataFrame, n: int = 10) -> go.Figure:
    if df.empty:
        return _empty()
    order = A.framework_summary(df).head(n)[C.COL_FRAMEWORK].tolist()
    sub = df[df[C.COL_FRAMEWORK].isin(order)]
    fig = go.Figure()
    for fw in order:
        s = sub[sub[C.COL_FRAMEWORK] == fw]
        fig.add_trace(go.Box(
            x=s[C.COL_DOWNLOADS].clip(lower=1), name=fw, orientation="h",
            marker=dict(color=C.PRIMARY, size=4, opacity=0.5),
            line=dict(color=C.PRIMARY, width=1.5), fillcolor="rgba(42,120,214,0.15)",
            boxpoints="outliers", customdata=s[[C.COL_NAME]].to_numpy(),
            hovertemplate="<b>%{customdata[0]}</b><br>%{x:,.0f} downloads<extra></extra>",
        ))
    fig.update_xaxes(type="log", dtick=1, title="Downloads per model (log scale)")
    fig.update_yaxes(categoryorder="array", categoryarray=order[::-1], automargin=True)
    return _style(fig, height=max(320, 34 * len(order) + 80))


def task_framework_heatmap(df: pd.DataFrame, value: str = "models") -> go.Figure:
    mat = A.task_framework_matrix(df, value=value)
    if mat.empty:
        return _empty()
    label = "Models" if value == "models" else "Downloads"
    text = [[fmt_num(v) if v else "" for v in row] for row in mat.to_numpy()]
    fig = go.Figure(go.Heatmap(
        z=mat.to_numpy(), x=mat.columns.tolist(), y=mat.index.tolist(),
        colorscale=[[0, "rgba(205,226,251,0.25)"]] + [
            [(i + 1) / len(C.SEQUENTIAL_BLUE), c] for i, c in enumerate(C.SEQUENTIAL_BLUE)],
        xgap=2, ygap=2, text=text, texttemplate="%{text}",
        textfont=dict(size=11),
        colorbar=dict(title=label, thickness=10),
        hovertemplate=f"<b>%{{y}}</b> × <b>%{{x}}</b><br>{label}: %{{z:,}}<extra></extra>",
    ))
    fig.update_yaxes(autorange="reversed", automargin=True)
    fig.update_xaxes(side="top", tickangle=-30)
    return _style(fig, height=max(360, 36 * len(mat) + 120))


# --------------------------------------------------------------------------- #
# Engagement vs downloads
# --------------------------------------------------------------------------- #
def engagement_scatter(df: pd.DataFrame, corr: A.Correlation, color_by: str = C.COL_FRAMEWORK) -> go.Figure:
    sub = df[(df[C.COL_DOWNLOADS] > 0) & (df[C.COL_LIKES] > 0)].copy()
    if sub.empty:
        return _empty()
    # Scatter plots cap categorical colour at 3 hues + gray Other
    sub["_group"] = A.top_n_with_other(sub[color_by], 3)
    groups = [g for g in sub["_group"].value_counts().index if g != "Other"]
    fig = go.Figure()
    if "Other" in set(sub["_group"]):
        groups.append("Other")
    for i, g in enumerate(groups):
        s = sub[sub["_group"] == g]
        color = C.OTHER_GRAY if g == "Other" else C.SERIES[i]
        fig.add_trace(go.Scatter(
            x=s[C.COL_DOWNLOADS], y=s[C.COL_LIKES], mode="markers", name=str(g),
            marker=dict(color=color, size=8, opacity=0.75 if g != "Other" else 0.45,
                        line=dict(color="white", width=1)),
            customdata=s[[C.COL_NAME, C.COL_TASK, C.COL_FRAMEWORK, C.COL_LIKES_PER_1K]].to_numpy(),
            hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[1]} · %{customdata[2]}"
                          "<br>Downloads: %{x:,.0f}<br>Likes: %{y:,.0f}"
                          "<br>Likes / 1K DL: %{customdata[3]:.2f}<extra></extra>",
        ))
    if not np.isnan(corr.slope):
        xs = np.logspace(np.log10(sub[C.COL_DOWNLOADS].min()), np.log10(sub[C.COL_DOWNLOADS].max()), 50)
        ys = 10 ** (corr.slope * np.log10(xs) + corr.intercept)
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines", name=f"Trend (slope {corr.slope:.2f})",
            line=dict(color="#0b0b0b", width=2, dash="dash"), hoverinfo="skip",
        ))
    fig.update_xaxes(type="log", dtick=1, title="Total downloads (log)")
    fig.update_yaxes(type="log", dtick=1, title="Community likes (log)")
    return _style(fig, height=520, legend=True)


def residual_by_group(df: pd.DataFrame, by: str, n: int = 12) -> go.Figure:
    """Median engagement residual per group: who over/under-performs on likes."""
    scored = A.add_engagement_residual(df)
    if scored.empty:
        return _empty()
    top = scored[by].value_counts().head(n).index
    g = (scored[scored[by].isin(top)].groupby(by)["Engagement_Residual"]
         .agg(["median", "count"]).reset_index().sort_values("median"))
    colors = [C.SERIES[0] if v >= 0 else C.SERIES[1] for v in g["median"]]
    fig = go.Figure(go.Bar(
        x=g["median"], y=g[by], orientation="h", marker=dict(color=colors, cornerradius=4),
        customdata=g[["count"]].to_numpy(),
        hovertemplate="<b>%{y}</b><br>Median residual: %{x:+.2f} log10 units"
                      "<br>Models: %{customdata[0]}<extra></extra>",
    ))
    fig.add_vline(x=0, line=dict(color=INK, width=1))
    fig.update_xaxes(title="Likes vs. expected (log10, + = more liked than downloads predict)")
    fig.update_yaxes(automargin=True)
    return _style(fig, height=max(300, 28 * len(g) + 80))


# --------------------------------------------------------------------------- #
# Developers
# --------------------------------------------------------------------------- #
def developer_sunburst(df: pd.DataFrame, n_dev: int = 12) -> go.Figure:
    if df.empty:
        return _empty()
    devs = A.developer_summary(df).head(n_dev)[C.COL_DEV]
    sub = df[df[C.COL_DEV].isin(devs)]
    agg = sub.groupby([C.COL_DEV, C.COL_TASK], as_index=False)[C.COL_DOWNLOADS].sum()
    fig = px.sunburst(agg, path=[C.COL_DEV, C.COL_TASK], values=C.COL_DOWNLOADS,
                      color=C.COL_DOWNLOADS, color_continuous_scale=C.SEQUENTIAL_BLUE)
    fig.update_traces(
        hovertemplate="<b>%{label}</b><br>%{value:,.0f} downloads"
                      "<br>%{percentParent:.1%} of parent<extra></extra>",
        marker=dict(line=dict(color="white", width=1.5)),
        insidetextorientation="radial",
    )
    fig.update_layout(coloraxis_showscale=False)
    return _style(fig, height=560)
