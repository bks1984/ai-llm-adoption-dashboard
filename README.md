# 🤖 Global AI & LLM Tools Adoption — Interactive Dashboard (2026)

An interactive **Streamlit + Plotly** dashboard and analysis toolkit for the Kaggle dataset
[**Global AI & LLM Tools Adoption Dataset (2026)**](https://www.kaggle.com/datasets/muhammadahmeddata/global-ai-and-llm-tools-adoption-dataset-2026).
The dataset covers about 1,000 open-source AI models, ML libraries and LLM tools from the
Hugging Face Hub, with their downloads, community likes and an engagement score.

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/built%20with-Streamlit-ff4b4b)
![License](https://img.shields.io/badge/code-MIT-green)
![Data](https://img.shields.io/badge/data-Apache%202.0-lightgrey)
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ai-llm-adoption-dashboard.streamlit.app)

**▶ Live demo: https://ai-llm-adoption-dashboard.streamlit.app**

---

## 🎯 Use cases covered

The dataset page lists four analysis goals. Each one has its own part of the dashboard:

| Kaggle use case | Where it lives | What you get |
|---|---|---|
| **AI trend analysis**: which tasks dominate open-source AI | 🧭 *Task trends* tab | Treemap of download share by task, *model supply vs. download demand* comparison, summary table |
| **Framework popularity mapping**: Transformers vs Sentence-Transformers vs GGUF | 🧱 *Frameworks* tab | Total downloads per framework, per-model distribution (box plot), task × framework heatmap |
| **Engagement vs. downloads correlation**: do high-download models get the most likes? | 💬 *Engagement* tab | Log-log scatter with fitted trend, Spearman/Pearson/elasticity, *hidden gems* and *silent workhorses*, over/under-performing groups |
| **Data visualisation of top developers & tools** | 📊 *Overview* and 🏢 *Developers* tabs | Top-N rankings, download tiers, concentration (Lorenz) curve, developer → task sunburst |

There is also a 🔎 **Data explorer** tab with a sortable table and a filtered CSV export.
The sidebar filters for task, framework, developer, minimum downloads and free-text search
apply to every tab.

## 🗂️ Project structure

```
ai-llm-adoption-dashboard/
├── app.py                      # Streamlit dashboard (entry point)
├── ai_adoption/                # Reusable analysis package
│   ├── config.py               # Paths, schema, chart palette
│   ├── data_loader.py          # Load → validate → clean → derive metrics
│   ├── analysis.py             # KPIs, group summaries, correlation, rankings
│   └── charts.py               # Plotly figure builders
├── scripts/
│   ├── download_data.py        # Fetch the real dataset via the Kaggle API
│   ├── generate_sample_data.py # Schema-identical synthetic sample (offline demo)
│   └── build_processed.py      # Write cleaned CSV + print headline findings
├── data/
│   ├── raw/                    # real Kaggle CSV (loaded by default)
│   ├── processed/              # cleaned output (git-ignored)
│   └── sample/                 # synthetic fallback for offline use
├── notebooks/
│   └── 01_exploratory_analysis.ipynb
├── tests/                      # pytest: loader, analysis, charts, app smoke tests
├── docs/
│   └── DATA_DICTIONARY.md
├── .streamlit/config.toml      # Theme
├── .github/workflows/ci.yml    # Lint + tests on push / PR
├── requirements.txt
└── Makefile
```

## 🚀 Quick start

```bash
git clone https://github.com/<your-username>/ai-llm-adoption-dashboard.git
cd ai-llm-adoption-dashboard
python -m venv .venv
# Windows: .venv\Scripts\activate   |   macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Then open http://localhost:8501.

### Key findings (real data, 1,000 models)

| Question | Answer from the data |
|---|---|
| Which tasks dominate downloads? | **sentence-similarity** (16%) edges out **text-generation** (14%), even though text-generation has 3× as many models (200 vs 62). Embedding models are pulled in by retrieval pipelines at huge volume. |
| Which framework leads? | **transformers** carries 43% of downloads across 516 models. **sentence-transformers** has 24% from only 75 models. GGUF has 59 models but a small share of downloads. |
| Do the most downloaded models get the most likes? | Only **weakly** (Spearman ρ = 0.31). Likes grow more slowly than downloads (log-log slope ≈ 0.70), so the most downloaded "workhorse" models get few likes per download. |
| How concentrated are downloads? | The top 10 models hold 25% of all downloads (Gini 0.63, *within* a list that already contains only the 1,000 most downloaded models). |
| Who publishes the most models? | **Qwen** (104 models), then unsloth, google, Comfy-Org and facebook. 336 publishers in total. |

### About the data

The real Kaggle CSV is committed at `data/raw/global_ai_llm_adoption_2026.csv`, and the
dashboard loads it by default. Data quirks the loader handles:

- **`Developer_Org` is "Community/Independent" on every row.** When that column has only one
  value, the loader takes the publisher from the repository id instead (`Qwen/Qwen3-8B` → `Qwen`).
- **`Adoption_Engagement_Score` = likes per 1,000 downloads, rounded to 2 decimals.** We checked
  this against all 1,000 rows.
- **Every model has at least ~360K downloads.** The file is a snapshot of the most downloaded
  models on the Hub, not a random sample, so the lower download tiers are empty.
- `General AI` (task) and `Open-Source` (framework) are the source's catch-all labels for
  untagged models.

### Refreshing the data

To pull a fresh copy (or if the committed CSV is removed), pick one option:

1. **Kaggle API:** create a token at *kaggle.com → Settings → Create New Token*, save
   `kaggle.json` to `~/.kaggle/`, then run:
   ```bash
   pip install kaggle
   python scripts/download_data.py
   ```
2. **Manual:** download the zip from the dataset page and drop the CSV into `data/raw/`.
3. **In the app:** upload the CSV with the sidebar file uploader.

The loader uses the first CSV it finds in `data/raw/`. It also accepts headers with spaces
instead of underscores, numbers with thousands separators, and a missing
`Adoption_Engagement_Score` column.

## 🧮 Methodology notes

- **Likes per 1K downloads** is the engagement rate used throughout. The
  `Adoption_Engagement_Score` column is kept as provided, because the dataset page does not
  publish its formula. When the column is missing, the loader falls back to likes per 1K downloads.
- **Correlation** uses Spearman's ρ, which is robust to the heavy-tailed download
  distribution, plus Pearson's r on log10 values. The **log-log slope** is the elasticity of
  likes with respect to downloads. A slope below 1 means likes grow more slowly than downloads.
- **Engagement residual**: how far (in log10 units) a model's likes are from what its
  downloads predict. Positive residuals are *hidden gems* (liked more than downloaded).
  Negative residuals are *silent workhorses*, often dependencies that pipelines pull in
  automatically.
- **Concentration**: a Lorenz-style curve and the Gini coefficient show how few models
  account for most downloads.

## 🧪 Development

```bash
pip install -r requirements-dev.txt
pytest                       # 37 tests: loader, analysis, every chart, headless app run
ruff check .                 # lint
python scripts/build_processed.py   # text report of headline findings
```

`make install | run | test | lint | sample | data` wrap the same commands.

## ☁️ Deploy

The app deploys as-is to **[Streamlit Community Cloud](https://streamlit.io/cloud)**: push to
GitHub, click *New app*, and select `app.py`. It deploys with the committed real data.

## 📄 License & attribution

- Code: [MIT](LICENSE).
- Data: *Global AI & LLM Tools Adoption Dataset (2026)* by Muhammad Ahmad on Kaggle,
  licensed **Apache 2.0**. The dataset page says it was collected from the Hugging Face
  Hub API.
- The real data is redistributed unchanged in `data/raw/` under Apache 2.0.
- `data/sample/ai_llm_tools_sample.csv` is **synthetic**: an offline fallback used only when
  `data/raw/` is empty. Its rows are marked `Data_Source = "Synthetic sample"`.
