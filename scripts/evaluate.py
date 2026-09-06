"""Evaluate trained artifacts on the held-out test set against a lead baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from gru_summarizer.artifacts import verify_manifest, write_manifest
from gru_summarizer.evaluation import lead_baseline, rouge_scores
from gru_summarizer.inference import load_summarizer


CHALLENGE_REVIEWS = {
    "wireless headphones": (
        "I bought this wireless headphone set last week. The sound quality is amazing, "
        "with deep bass and clear highs. However, the battery life is terrible. It barely "
        "lasts 3 hours on a single charge. Also, the ear cups get very uncomfortable after "
        "about 30 minutes of wearing them. Customer service was helpful when I called, but "
        "overall I wouldn't recommend them for long trips."
    ),
    "restaurant (outside training domain)": (
        "Absolutely incredible dining experience! We ordered the steak and the truffle pasta. "
        "Both were cooked to perfection. The ambiance is very romantic, with dim lighting and "
        "soft jazz playing in the background. The waiter, John, was extremely attentive without "
        "being overbearing. Prices are a bit high, but well worth it for a special occasion. "
        "Will definitely be coming back!"
    ),
    "mobile app": (
        "This latest update completely broke the app for me. It crashes every time I try to "
        "open a new project. I've tried reinstalling, clearing the cache, and restarting my "
        "phone, but nothing works. The user interface is also much more confusing now, hiding "
        "features that used to be one click away. I'm canceling my subscription until they fix "
        "these glaring bugs."
    ),
}


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
    challenge_names = list(CHALLENGE_REVIEWS)
    challenge_texts = [CHALLENGE_REVIEWS[name] for name in challenge_names]
    challenge_predictions = summarizer.summarize_batch(challenge_texts)
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
        "post_training_challenge_examples": [
            {
                "domain": name,
                "review": review,
                "gru_summary": prediction,
            }
            for name, review, prediction in zip(
                challenge_names, challenge_texts, challenge_predictions
            )
        ],
        "challenge_note": (
            "These user-supplied stress tests were not training rows and are excluded from "
            "ROUGE. The restaurant example is outside the consumer-product training domain."
        ),
        "limitations": [
            "Training covers consumer products, not restaurant or other service reviews.",
            "The source corpus contains only positive and negative ratings, not neutral ratings.",
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
