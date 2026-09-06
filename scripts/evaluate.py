"""Evaluate trained artifacts on the held-out test set against a lead baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from gru_summarizer.artifacts import verify_manifest, write_manifest
from gru_summarizer.evaluation import lead_baseline, rouge_scores
from gru_summarizer.inference import load_summarizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--test-file", type=Path, default=Path("data/processed/test.csv"))
    parser.add_argument("--limit", type=int, default=0, help="0 evaluates every held-out example")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--examples", type=int, default=6)
    parser.add_argument(
        "--compare-artifacts",
        type=Path,
        help="Optional previous model to score on the identical test rows",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.test_file).dropna(subset=["source", "target"])
    if args.limit > 0:
        frame = frame.iloc[: args.limit]
    texts = frame["source"].astype(str).tolist()
    references = frame["target"].astype(str).tolist()
    summarizer = load_summarizer(args.artifacts_dir)
    predictions: list[str] = []
    for start in range(0, len(texts), args.batch_size):
        predictions.extend(summarizer.summarize_batch(texts[start : start + args.batch_size]))
    baselines = [lead_baseline(text, summarizer.config.max_summary_tokens - 2) for text in texts]
    manifest = verify_manifest(args.artifacts_dir)
    result = {
        "evaluation_type": "held-out test split",
        "examples_evaluated": len(texts),
        "metric": "mean ROUGE F1 with stemming",
        "gru": rouge_scores(references, predictions),
        "lead_words_baseline": rouge_scores(references, baselines),
        "qualitative_examples": [
            {
                "review": texts[i],
                "reference_title": references[i],
                "gru_summary": predictions[i],
                "lead_baseline": baselines[i],
            }
            for i in range(min(args.examples, len(texts)))
        ],
        "limitations": [
            "The dataset is a single anonymized women's clothing retailer domain.",
            "Even with attention, a compact GRU can produce generic or repetitive titles.",
            f"Inputs beyond {summarizer.config.max_input_tokens} normalized tokens are truncated.",
            "ROUGE rewards lexical overlap and does not establish factual correctness.",
        ],
    }
    if args.compare_artifacts:
        previous = load_summarizer(args.compare_artifacts)
        previous_predictions: list[str] = []
        for start in range(0, len(texts), args.batch_size):
            previous_predictions.extend(
                previous.summarize_batch(texts[start : start + args.batch_size])
            )
        result["previous_gru_same_split"] = rouge_scores(references, previous_predictions)
        result["comparison_note"] = (
            "The previous shipped model was re-scored on these exact test rows. "
            "Neither model was trained on them."
        )
        for index, example in enumerate(result["qualitative_examples"]):
            example["previous_gru_summary"] = previous_predictions[index]
    (args.artifacts_dir / "evaluation.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    write_manifest(args.artifacts_dir, manifest["metadata"])
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
