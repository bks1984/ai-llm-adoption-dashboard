"""Generate a synthetic sample that matches the Kaggle dataset's schema.

The real data lives on Kaggle and needs an account to download, so the repo
ships this generator to let the dashboard run out of the box. Every row is
tagged ``Data_Source = "Synthetic sample"`` so it can never be mistaken for
the real numbers. Replace it by running ``scripts/download_data.py``.

Usage:
    python scripts/generate_sample_data.py [--rows 1000] [--seed 42] [--out PATH]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ai_adoption import config as C  # noqa: E402

# (task, relative weight, typical log10 downloads)
TASKS = [
    ("text-generation", 26, 4.6),
    ("sentence-similarity", 9, 5.4),
    ("feature-extraction", 8, 5.2),
    ("text-classification", 9, 4.5),
    ("automatic-speech-recognition", 6, 4.8),
    ("image-classification", 6, 4.4),
    ("text-to-image", 6, 4.3),
    ("token-classification", 4, 4.3),
    ("fill-mask", 4, 5.6),
    ("translation", 3, 4.1),
    ("summarization", 3, 4.0),
    ("object-detection", 3, 4.2),
    ("image-text-to-text", 4, 4.4),
    ("zero-shot-classification", 2, 4.6),
    ("text-to-speech", 3, 3.9),
    ("question-answering", 2, 4.0),
    ("image-segmentation", 2, 4.0),
]

# task -> framework weights
FRAMEWORKS = {
    "default": {"transformers": 60, "pytorch": 14, "safetensors": 8, "gguf": 6,
                "onnx": 4, "tensorflow": 3, "jax": 2, "mlx": 3},
    "text-generation": {"transformers": 45, "gguf": 25, "pytorch": 8, "safetensors": 10,
                        "mlx": 7, "onnx": 2, "jax": 3},
    "sentence-similarity": {"sentence-transformers": 70, "transformers": 15, "onnx": 10,
                            "pytorch": 5},
    "feature-extraction": {"sentence-transformers": 40, "transformers": 40, "onnx": 12,
                           "pytorch": 8},
    "text-to-image": {"diffusers": 75, "pytorch": 10, "safetensors": 10, "onnx": 5},
    "automatic-speech-recognition": {"transformers": 55, "pytorch": 15, "coreml": 12,
                                     "ctranslate2": 10, "onnx": 8},
    "object-detection": {"ultralytics": 45, "transformers": 35, "pytorch": 12, "onnx": 8},
    "image-segmentation": {"transformers": 50, "pytorch": 30, "onnx": 20},
    "text-to-speech": {"pytorch": 40, "transformers": 35, "onnx": 25},
}

DEVELOPERS = [
    ("meta-llama", 3.0, 0.7), ("google", 3.0, 0.5), ("microsoft", 3.0, 0.4),
    ("openai", 2.0, 0.9), ("sentence-transformers", 2.5, 0.3), ("Qwen", 3.0, 0.6),
    ("mistralai", 2.0, 0.7), ("deepseek-ai", 2.0, 1.0), ("facebook", 2.5, 0.4),
    ("BAAI", 2.0, 0.3), ("stabilityai", 1.5, 1.0), ("nvidia", 2.0, 0.4),
    ("intfloat", 1.2, 0.2), ("FacebookAI", 1.5, 0.4), ("google-bert", 1.5, 0.5),
    ("HuggingFaceTB", 1.0, 0.6), ("TheBloke", 2.5, 0.2), ("bartowski", 2.5, 0.1),
    ("unsloth", 2.0, 0.3), ("mlx-community", 2.0, -0.2), ("argmaxinc", 0.6, 0.0),
    ("Systran", 0.8, 0.1), ("black-forest-labs", 1.0, 1.1), ("ultralytics", 0.8, 0.3),
    ("cardiffnlp", 0.8, 0.0), ("Helsinki-NLP", 1.2, -0.2), ("nomic-ai", 0.8, 0.2),
    ("jinaai", 0.8, 0.2), ("allenai", 1.0, 0.2), ("ibm-granite", 0.8, 0.0),
    ("tiiuae", 0.7, 0.3), ("CohereForAI", 0.7, 0.4), ("01-ai", 0.5, 0.2),
    ("community", 8.0, -0.6),
]

FAMILIES = ["base", "instruct", "chat", "mini", "small", "large", "v2", "v3", "lite",
            "turbo", "xl", "pro", "distilled", "multilingual", "en", "code", "vision"]
SIZES = ["0.5B", "1B", "1.5B", "3B", "7B", "8B", "13B", "14B", "32B", "70B", "", "", ""]
STEMS = {
    "text-generation": ["llama", "qwen", "mistral", "phi", "gemma", "falcon", "granite", "smol"],
    "sentence-similarity": ["all-MiniLM", "bge", "e5", "gte", "mpnet", "nomic-embed"],
    "feature-extraction": ["bge", "e5", "jina-embeddings", "clip", "dinov2"],
    "automatic-speech-recognition": ["whisper", "whisperkit", "wav2vec2", "parakeet"],
    "text-to-image": ["stable-diffusion", "flux", "sdxl", "kandinsky"],
    "fill-mask": ["bert", "roberta", "deberta", "xlm-roberta", "distilbert"],
    "object-detection": ["yolo", "detr", "rt-detr", "owlvit"],
}


def _pick(rng: np.random.Generator, weights: dict[str, float]) -> str:
    keys = list(weights)
    p = np.array(list(weights.values()), dtype=float)
    return str(rng.choice(keys, p=p / p.sum()))


def generate(rows: int = 1000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    task_names = [t[0] for t in TASKS]
    task_w = np.array([t[1] for t in TASKS], dtype=float)
    task_mu = {t[0]: t[2] for t in TASKS}
    dev_names = [d[0] for d in DEVELOPERS]
    dev_w = np.array([d[1] for d in DEVELOPERS], dtype=float)
    dev_boost = {d[0]: d[2] for d in DEVELOPERS}

    records, seen = [], set()
    while len(records) < rows:
        task = str(rng.choice(task_names, p=task_w / task_w.sum()))
        dev = str(rng.choice(dev_names, p=dev_w / dev_w.sum()))
        fw = _pick(rng, FRAMEWORKS.get(task, FRAMEWORKS["default"]))
        if dev in {"TheBloke", "bartowski"}:
            fw = "gguf"
        elif dev == "mlx-community":
            fw = "mlx"

        stem = str(rng.choice(STEMS.get(task, ["model", "tiny", "base-net", "vit", "t5"])))
        size = str(rng.choice(SIZES)) if task == "text-generation" else ""
        family = str(rng.choice(FAMILIES))
        name = "-".join(p for p in (stem, size, family) if p)
        if fw == "gguf":
            name += "-GGUF"
        owner = dev if dev != "community" else f"user{rng.integers(100, 9999)}"
        full = f"{owner}/{name}"
        if full in seen:
            continue
        seen.add(full)

        log_dl = rng.normal(task_mu[task] + dev_boost[dev], 1.05)
        if log_dl > 7.8:  # soft-compress the extreme tail instead of a hard cap
            log_dl = 7.8 + (log_dl - 7.8) * 0.35
        downloads = int(max(50, 10 ** log_dl))
        # likes scale sub-linearly with downloads, with lots of scatter
        log_likes = -1.9 + 0.68 * np.log10(downloads) + rng.normal(0, 0.45)
        if fw in {"gguf", "mlx"}:
            log_likes -= 0.35  # quantised mirrors get downloaded, rarely liked
        if dev in {"black-forest-labs", "deepseek-ai", "stabilityai"}:
            log_likes += 0.4   # hype-driven releases
        likes = int(max(0, round(10 ** log_likes - 1)))

        records.append({
            C.COL_NAME: full,
            C.COL_DEV: dev if dev != "community" else owner,
            C.COL_TASK: task,
            C.COL_FRAMEWORK: fw,
            C.COL_DOWNLOADS: downloads,
            C.COL_LIKES: likes,
        })

    df = pd.DataFrame.from_records(records)
    df = df.sort_values(C.COL_DOWNLOADS, ascending=False, ignore_index=True)
    df.insert(0, C.COL_ID, np.arange(1, len(df) + 1))
    df[C.COL_SCORE] = (df[C.COL_LIKES] / df[C.COL_DOWNLOADS] * 1_000).round(4)
    df[C.COL_SOURCE] = "Synthetic sample"
    return df[C.ALL_COLUMNS]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rows", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=C.SAMPLE_FILE)
    args = ap.parse_args()

    df = generate(args.rows, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df):,} synthetic rows -> {args.out}")


if __name__ == "__main__":
    main()
