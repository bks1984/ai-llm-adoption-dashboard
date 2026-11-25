"""Global AI & LLM Tools Adoption — interactive Streamlit dashboard.

Run with:  streamlit run app.py
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from ai_adoption import analysis as A
from ai_adoption import charts as CH
from ai_adoption import config as C
from ai_adoption.data_loader import SchemaError, clean, find_source_file, load_dataset

st.set_page_config(
    page_title="AI & LLM Tools Adoption 2026",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .block-container {padding-top: 2rem; padding-bottom: 2rem;}
      [data-testid="stMetricValue"] {font-size: 1.6rem;}
      [data-testid="stMetricLabel"] p {font-size: 0.85rem; opacity: 0.8;}
      .insight {border-left: 3px solid #2a78d6; padding: 0.4rem 0.9rem; margin: 0.2rem 0 1rem;
                background: rgba(42,120,214,0.06); border-radius: 0 6px 6px 0;}
    </style>
    """,
    unsafe_allow_html=True,
)

PLOT_CFG = {"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]}


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner="Loading dataset…")
def _load_default(path: str, mtime: float, is_sample: bool):
    # path + mtime are the cache key, so a new or updated data file is picked up
    data = load_dataset(path)
    return data.df, data.source_path.name, is_sample


@st.cache_data(show_spinner="Reading uploaded file…")
def _load_upload(raw: bytes, name: str):
    import io
    return clean(pd.read_csv(io.BytesIO(raw))), name, False


def insight(text: str) -> None:
    st.markdown(f'<div class="insight">{text}</div>', unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("🤖 AI & LLM Adoption")
    uploaded = st.file_uploader("Load a dataset CSV", type="csv",
                                help="Upload the Kaggle CSV to override the default source.")

try:
    if uploaded is not None:
        df_all, source_name, is_sample = _load_upload(uploaded.getvalue(), uploaded.name)
    else:
        src, sample = find_source_file()
        df_all, source_name, is_sample = _load_default(str(src), src.stat().st_mtime, sample)
except (SchemaError, FileNotFoundError) as exc:
    st.error(str(exc))
    st.stop()

with st.sidebar:
    st.caption(f"Source: `{source_name}` · {len(df_all):,} rows")
    st.divider()
    st.subheader("Filters")

    task_opts = A.task_summary(df_all)[C.COL_TASK].tolist()
    fw_opts = A.framework_summary(df_all)[C.COL_FRAMEWORK].tolist()
    dev_opts = A.developer_summary(df_all)[C.COL_DEV].tolist()

    sel_tasks = st.multiselect("Task category", task_opts, placeholder="All tasks")
    sel_fws = st.multiselect("Framework / library", fw_opts, placeholder="All frameworks")
    sel_devs = st.multiselect("Developer / org", dev_opts, placeholder="All developers")
    tier = st.select_slider("Minimum downloads",
                            options=["0", "1K", "10K", "100K", "1M", "10M"], value="0")
    min_dl = {"0": 0, "1K": 1e3, "10K": 1e4, "100K": 1e5, "1M": 1e6, "10M": 1e7}[tier]
    search = st.text_input("Search model or developer", placeholder="e.g. whisper, llama")
    top_n = st.slider("Items in ranked charts", 5, 30, 15)

    st.divider()
    st.markdown(f"[Dataset on Kaggle]({C.KAGGLE_URL}) · Apache 2.0")

df = A.apply_filters(df_all, sel_tasks, sel_fws, sel_devs, int(min_dl), search)

# --------------------------------------------------------------------------- #
# Header + KPIs
# --------------------------------------------------------------------------- #
st.title("Global AI & LLM Tools Adoption — 2026")
st.caption("Popularity, community engagement and download metrics for open-source AI "
           "models and LLM tools from the Hugging Face Hub.")

if is_sample:
    st.warning(
        "**Showing the synthetic sample data.** The numbers are made up to match the "
        "dataset's shape. To see the real numbers, put the Kaggle CSV in `data/raw/` "
        "(`python scripts/download_data.py`) or upload it in the sidebar.",
        icon="⚠️",
    )

if df.empty:
    st.info("No models match the current filters. Widen them in the sidebar.")
    st.stop()

k = A.compute_kpis(df)
corr = A.engagement_correlation(df)

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Models / tools", f"{k.n_models:,}",
          delta=f"{k.n_models / len(df_all):.0%} of dataset" if len(df) != len(df_all) else None,
          delta_color="off")
c2.metric("Total downloads", CH.fmt_num(k.total_downloads))
c3.metric("Community likes", CH.fmt_num(k.total_likes))
c4.metric("Median downloads", CH.fmt_num(k.median_downloads))
c5.metric("Likes per 1K downloads", f"{k.likes_per_1k:.2f}")
c6.metric("Top-10 download share", f"{k.top10_share:.0%}")

tabs = st.tabs([
    "📊 Overview", "🧭 Task trends", "🧱 Frameworks",
    "💬 Engagement vs downloads", "🏢 Developers", "🔎 Data explorer",
])

# --------------------------------------------------------------------------- #
# 1 · Overview
# --------------------------------------------------------------------------- #
with tabs[0]:
    top = A.top_models(df, 1).iloc[0]
    insight(
        f"<b>{top[C.COL_NAME]}</b> is the most downloaded entry "
        f"({CH.fmt_num(top[C.COL_DOWNLOADS])}, {k.top1_share:.0%} of all downloads in view). "
        f"Downloads are highly concentrated: the top 10 entries hold <b>{k.top10_share:.0%}</b> "
        f"and the Gini coefficient is <b>{A.gini(df[C.COL_DOWNLOADS]):.2f}</b>."
    )
    left, right = st.columns([3, 2])
    with left:
        st.subheader(f"Top {top_n} models by downloads")
        st.plotly_chart(CH.ranked_bar(
            df, C.COL_NAME, C.COL_DOWNLOADS, "Total downloads", n=top_n,
            hover_extra={C.COL_LIKES: "Likes", C.COL_LIKES_PER_1K: "Likes / 1K DL"}),
            config=PLOT_CFG)
    with right:
        st.subheader("How many models reach each tier")
        st.plotly_chart(CH.download_tier_bar(df), config=PLOT_CFG)
        st.subheader("Download concentration")
        st.plotly_chart(CH.concentration_chart(df), config=PLOT_CFG)

# --------------------------------------------------------------------------- #
# 2 · Task trends
# --------------------------------------------------------------------------- #
with tabs[1]:
    tasks = A.task_summary(df)
    lead = tasks.iloc[0]
    most_models = tasks.sort_values("models", ascending=False).iloc[0]
    insight(
        f"<b>{lead[C.COL_TASK]}</b> leads with {lead['download_share']:.0%} of downloads. "
        f"<b>{most_models[C.COL_TASK]}</b> has the most models ({most_models['models']}). "
        "Where a task's download share is larger than its model share, a few models "
        "get most of the use. Where it is smaller, the task has many niche models."
    )
    st.subheader("Download share by task")
    st.caption("Tile size = downloads · colour = likes per 1K downloads")
    st.plotly_chart(CH.task_treemap(tasks), config=PLOT_CFG)
    st.subheader("Model supply vs. download demand")
    st.plotly_chart(CH.task_share_compare(tasks, n=top_n), config=PLOT_CFG)
    with st.expander("Task summary table"):
        st.dataframe(tasks, hide_index=True, column_config={
            "download_share": st.column_config.ProgressColumn("Download share", format="percent",
                                                              min_value=0, max_value=1),
            "model_share": st.column_config.NumberColumn("Model share", format="percent"),
            "likes_per_1k": st.column_config.NumberColumn("Likes / 1K DL", format="%.2f"),
        })

# --------------------------------------------------------------------------- #
# 3 · Frameworks
# --------------------------------------------------------------------------- #
with tabs[2]:
    fws = A.framework_summary(df)
    insight(
        "This view compares Transformers, Sentence-Transformers, GGUF and the rest. "
        f"<b>{fws.iloc[0][C.COL_FRAMEWORK]}</b> carries {fws.iloc[0]['download_share']:.0%} "
        "of downloads. The box plot shows the downloads of a typical model in each "
        "framework, so one blockbuster model doesn't skew the comparison."
    )
    left, right = st.columns(2)
    with left:
        st.subheader("Total downloads by framework")
        st.plotly_chart(CH.ranked_bar(fws, C.COL_FRAMEWORK, "downloads", "Total downloads",
                                      n=top_n, hover_extra={"models": "Models",
                                                            "download_share": "Share"}),
                        config=PLOT_CFG)
    with right:
        st.subheader("Downloads per model (distribution)")
        st.plotly_chart(CH.framework_box(df, n=min(top_n, 12)),
                        config=PLOT_CFG)
    st.subheader("Which frameworks serve which tasks")
    hm_metric = st.radio("Heatmap value", ["models", "downloads"], horizontal=True,
                         format_func=str.title)
    st.plotly_chart(CH.task_framework_heatmap(df, hm_metric),
                    config=PLOT_CFG)

# --------------------------------------------------------------------------- #
# 4 · Engagement vs downloads
# --------------------------------------------------------------------------- #
with tabs[3]:
    if corr.n >= 3:
        insight(
            f"Do the most downloaded models also get the most likes? There is a "
            f"<b>{corr.strength}</b> positive link (Spearman ρ = {corr.spearman:.2f}, "
            f"n = {corr.n:,}). But it is <b>sub-linear</b>: a 10× rise in downloads goes "
            f"with about {10 ** corr.slope:.1f}× more likes (log-log slope {corr.slope:.2f}). "
            "The most downloaded models are often pulled in automatically by pipelines, "
            "so they get few likes per download."
        )
    else:
        insight("Not enough models with both downloads and likes to estimate a relationship.")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Spearman ρ (rank)", f"{corr.spearman:.2f}")
    m2.metric("Pearson r (log-log)", f"{corr.pearson_log:.2f}")
    m3.metric("Elasticity (slope)", f"{corr.slope:.2f}")
    m4.metric("Relationship", corr.strength.title())

    color_dim = st.radio("Colour points by", [C.COL_FRAMEWORK, C.COL_TASK], horizontal=True,
                         format_func=lambda c: "Framework" if c == C.COL_FRAMEWORK else "Task")
    st.plotly_chart(CH.engagement_scatter(df, corr, color_dim),
                    config=PLOT_CFG)
    st.caption("Top 3 groups are coloured; the rest are gray. Dashed line = log-log fit.")

    left, right = st.columns(2)
    show_cols = [C.COL_NAME, C.COL_TASK, C.COL_FRAMEWORK, C.COL_DOWNLOADS, C.COL_LIKES,
                 C.COL_LIKES_PER_1K, "Engagement_Residual"]
    with left:
        st.subheader("💎 Hidden gems")
        st.caption("Liked far more than their downloads would predict (≥ 1K downloads)")
        st.dataframe(A.hidden_gems(df, top_n)[show_cols], hide_index=True)
    with right:
        st.subheader("⚙️ Silent workhorses")
        st.caption("Heavily downloaded but rarely liked (≥ 100K downloads)")
        st.dataframe(A.under_engaged(df, top_n)[show_cols], hide_index=True)

    st.subheader("Which groups over- or under-perform on likes")
    grp = st.radio("Group by", [C.COL_FRAMEWORK, C.COL_TASK, C.COL_DEV], horizontal=True,
                   format_func=lambda c: c.replace("_", " "))
    st.plotly_chart(CH.residual_by_group(df, grp, n=top_n),
                    config=PLOT_CFG)

# --------------------------------------------------------------------------- #
# 5 · Developers
# --------------------------------------------------------------------------- #
with tabs[4]:
    devs = A.developer_summary(df)
    insight(
        f"<b>{devs.iloc[0][C.COL_DEV]}</b> has the most downloads "
        f"({devs.iloc[0]['download_share']:.0%}). There are {k.n_developers:,} publishers "
        "in view. Click a segment in the sunburst to see that developer's tasks."
    )
    metric = st.radio("Rank developers by", ["downloads", "likes", "models"], horizontal=True,
                      format_func=str.title)
    left, right = st.columns([2, 3])
    with left:
        st.plotly_chart(CH.ranked_bar(devs, C.COL_DEV, metric, metric.title(), n=top_n,
                                      hover_extra={"models": "Models",
                                                   "likes_per_1k": "Likes / 1K DL"}),
                        config=PLOT_CFG)
    with right:
        st.plotly_chart(CH.developer_sunburst(df, n_dev=min(top_n, 15)),
                        config=PLOT_CFG)

# --------------------------------------------------------------------------- #
# 6 · Data explorer
# --------------------------------------------------------------------------- #
with tabs[5]:
    st.subheader(f"Filtered records ({len(df):,})")
    st.dataframe(
        df.sort_values(C.COL_DOWNLOADS, ascending=False),
        hide_index=True, height=520,
        column_config={
            C.COL_DOWNLOADS: st.column_config.NumberColumn("Downloads", format="localized"),
            C.COL_LIKES: st.column_config.NumberColumn("Likes", format="localized"),
            C.COL_SCORE: st.column_config.NumberColumn("Engagement score", format="%.4f"),
            C.COL_LIKES_PER_1K: st.column_config.NumberColumn("Likes / 1K DL", format="%.2f"),
        },
    )
    st.download_button(
        "⬇️ Download filtered CSV", df.to_csv(index=False).encode("utf-8"),
        file_name="ai_llm_tools_filtered.csv", mime="text/csv",
    )
    with st.expander("Column descriptions"):
        st.markdown("""
| Column | Description |
|---|---|
| `Record_ID` | Unique numeric identifier |
| `Model_Or_Tool_Name` | Repository / model identifier (e.g. `openai/whisper-large-v3`) |
| `Developer_Org` | Organisation or community account that publishes it |
| `Primary_Task_Category` | Functional domain (text-generation, sentence-similarity, …) |
| `Framework_Library` | Underlying library / format (transformers, GGUF, PyTorch, …) |
| `Total_Downloads` | Cumulative download count |
| `Community_Likes` | Hub likes / stars |
| `Adoption_Engagement_Score` | Engineered engagement-vs-downloads metric (from source) |
| `Data_Source` | Origin of the record |
| `Likes_per_1K_Downloads` | *Derived:* likes ÷ downloads × 1,000 |
| `Download_Tier` | *Derived:* order-of-magnitude download bucket |
""")
