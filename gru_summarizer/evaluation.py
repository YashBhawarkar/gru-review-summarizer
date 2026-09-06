"""ROUGE aggregation and a transparent lead-words baseline."""

from __future__ import annotations

import re
from collections.abc import Sequence

from rouge_score import rouge_scorer

from .preprocessing import normalize_text


def lead_baseline(text: str, max_words: int = 8) -> str:
    first_sentence = re.split(r"(?<=[.!?])\s+", str(text).strip(), maxsplit=1)[0]
    return " ".join(normalize_text(first_sentence).split()[:max_words])


def rouge_scores(references: Sequence[str], predictions: Sequence[str]) -> dict[str, float]:
    if len(references) != len(predictions) or not references:
        raise ValueError("ROUGE needs equally sized, non-empty references and predictions")
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    totals = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    for reference, prediction in zip(references, predictions, strict=True):
        scores = scorer.score(reference, prediction)
        for name in totals:
            totals[name] += scores[name].fmeasure
    return {name: round(value / len(references), 6) for name, value in totals.items()}
