"""Clean the source CSV and write ``data/processed/ai_llm_tools_clean.csv``.

Also prints a short text report of the headline findings.

Usage:
    python scripts/build_processed.py [--input PATH]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ai_adoption import analysis as A  # noqa: E402
from ai_adoption import config as C  # noqa: E402
from ai_adoption.data_loader import load_dataset, save_processed  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=None)
    args = ap.parse_args()

    data = load_dataset(args.input)
    df = data.df
    out = save_processed(df)

    k = A.compute_kpis(df)
    corr = A.engagement_correlation(df)
    tasks = A.task_summary(df).head(5)
    fws = A.framework_summary(df).head(5)

    print(f"Source : {data.source_path} {'(SYNTHETIC SAMPLE)' if data.is_sample else ''}")
    print(f"Output : {out}")
    print(f"Models : {k.n_models:,} | Developers: {k.n_developers:,} | "
          f"Tasks: {k.n_tasks} | Frameworks: {k.n_frameworks}")
    print(f"Downloads: {k.total_downloads:,} | Likes: {k.total_likes:,} | "
          f"Top-10 share: {k.top10_share:.1%} | Gini: {A.gini(df[C.COL_DOWNLOADS]):.2f}")
    print(f"Likes~Downloads Spearman ρ = {corr.spearman:.2f} ({corr.strength}), "
          f"log-log slope = {corr.slope:.2f}")
    print("\nTop tasks by downloads:")
    for _, r in tasks.iterrows():
        print(f"  {r[C.COL_TASK]:<32} {r['download_share']:>6.1%}  ({r['models']} models)")
    print("\nTop frameworks by downloads:")
    for _, r in fws.iterrows():
        print(f"  {r[C.COL_FRAMEWORK]:<32} {r['download_share']:>6.1%}  ({r['models']} models)")


if __name__ == "__main__":
    main()
