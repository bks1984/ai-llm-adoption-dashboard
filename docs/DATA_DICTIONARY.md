# Data dictionary

Source: [Global AI & LLM Tools Adoption Dataset (2026)](https://www.kaggle.com/datasets/muhammadahmeddata/global-ai-and-llm-tools-adoption-dataset-2026)
(Apache 2.0, 131 KB, 1,000 rows, collected via the Hugging Face Hub API).
File: `data/raw/global_ai_llm_adoption_2026.csv`. 41 task categories, 56 frameworks, 336 publishers.
Every model has at least ~360K downloads (it is a snapshot of the most downloaded models).

## Source columns

| Column | Type | Description | Example values |
|---|---|---|---|
| `Record_ID` | int | Unique identifier (1–1000) | `1` |
| `Model_Or_Tool_Name` | str | Official repository or model identifier | `sentence-transformers/all-MiniLM-L6-v2`, `gpt2`, `whisperkit` |
| `Developer_Org` | str | Creator, organisation or community maintainer. **In the published file this is `Community/Independent` on every row**, so the loader takes the publisher from `Model_Or_Tool_Name` instead | `Community/Independent` |
| `Primary_Task_Category` | category | Functional domain of the model | `text-generation`, `sentence-similarity`, `automatic-speech-recognition` |
| `Framework_Library` | category | Underlying framework or file format | `transformers`, `pytorch`, `gguf` |
| `Total_Downloads` | int | Cumulative real-world download count | `1250000` |
| `Community_Likes` | int | Community likes / stars | `3400` |
| `Adoption_Engagement_Score` | float | Engagement relative to downloads. Checked: equals `Community_Likes / Total_Downloads × 1000`, rounded to 2 dp | `0.04` |
| `Data_Source` | category | Origin of the record | `HuggingFace Live Hub API` |

## Derived columns (added by `ai_adoption.data_loader.clean`)

| Column | Definition |
|---|---|
| `Likes_per_1K_Downloads` | `Community_Likes / Total_Downloads × 1000` (0 when downloads = 0) |
| `Download_Tier` | Order-of-magnitude bucket: `<1K`, `1K–10K`, `10K–100K`, `100K–1M`, `1M–10M`, `10M+` (each bucket includes its lower bound) |
| `Engagement_Residual` | *(computed on demand)* log10(likes) minus the value the log-log fit of likes on downloads predicts |

## Cleaning rules

1. Spaces in headers become underscores.
2. Downloads and likes: thousands separators removed; non-numeric values → 0; negatives clipped to 0.
3. If `Developer_Org` has a single value for every row, it is replaced by the `org` prefix of `Model_Or_Tool_Name`.
4. Text fields are trimmed. Blank or null values → `unknown`. Task names are lower-cased.
5. Duplicate (`Model_Or_Tool_Name`, `Developer_Org`) pairs are dropped.
6. A missing `Adoption_Engagement_Score` falls back to `Likes_per_1K_Downloads`.
